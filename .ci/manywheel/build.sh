#!/usr/bin/env bash

set -ex

SCRIPTPATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
PYTORCH_ROOT="${PYTORCH_ROOT:-$(cd "${SCRIPTPATH}/../.." && pwd)}"

case "${GPU_ARCH_TYPE:-BLANK}" in
    cuda|cuda-aarch64|cpu|cpu-aarch64|cpu-riscv64|cpu-cxx11-abi|xpu|rocm)
        # New pipeline: pyproject-driven build via `python -m build`
        # then patchelf-based wheel repair.
        source "${SCRIPTPATH}/set_desired_python.sh"

        # build_env_setup.py needs its build-flag exports (USE_CUDA,
        # TH_BINARY_BUILD, ...) to reach the wheel build subprocess; it
        # writes them here for us to source.
        ENV_FILE=$(mktemp)
        trap 'rm -f "$ENV_FILE"' EXIT
        python3 "${SCRIPTPATH}/build_env_setup.py" --env-out "$ENV_FILE"
        source "$ENV_FILE"

        python3 "${SCRIPTPATH}/build_install_deps.py" "${PYTORCH_ROOT}"

        : "${PYTORCH_FINAL_PACKAGE_DIR:=/artifacts}"
        mkdir -p "${PYTORCH_FINAL_PACKAGE_DIR}"
        RAW_WHEEL_DIR=$(mktemp -d)

        cd "${PYTORCH_ROOT}"
        python3 "${SCRIPTPATH}/build_wheel.py"  "${RAW_WHEEL_DIR}"

        # native-AOT stage 2, BEFORE the repair so the relinked libtorch_cuda goes
        # through patchelf like the rest of the wheel. The kernel builders import
        # torch, so the raw wheel is installed first.
        #
        # CUDA arch types only: this arm also serves cpu/xpu/rocm, where stage 2 can
        # only skip.
        if [[ "${GPU_ARCH_TYPE}" == cuda* ]]; then
            # nullglob so an empty directory counts as zero, not the pattern.
            naot_wheels=()
            shopt -s nullglob
            naot_wheels=("${RAW_WHEEL_DIR}"/*.whl)
            shopt -u nullglob
            if [[ ${#naot_wheels[@]} -ne 1 ]]; then
              echo "native-AOT: expected one raw wheel in ${RAW_WHEEL_DIR}, found ${#naot_wheels[@]}" >&2
              exit 1
            fi
            RAW_WHEEL="${naot_wheels[0]}"
            # --no-deps --no-index: Requires-Dist would pull GBs of CUDA wheels.
            python3 -m pip install --progress-bar off --no-deps --no-index "${RAW_WHEEL}"
            # Sourced for install_cutlass_dsl, so the DSL pin lives in one place.
            source "${PYTORCH_ROOT}/.ci/pytorch/common_utils.sh"
            # Stage 2 owns the verdict (see .ci/pytorch/build.sh for why a separate
            # probe here was wrong).
            if [[ "$(python3 tools/native_aot/build_stage2.py --print-verdict)" == "RUN" ]]; then
              install_cutlass_dsl
            fi
            python3 tools/native_aot/build_stage2.py --wheel "${RAW_WHEEL}"
        fi  # GPU_ARCH_TYPE == cuda*

        python3 "${SCRIPTPATH}/repair_wheel.py" "${RAW_WHEEL_DIR}" "${PYTORCH_FINAL_PACKAGE_DIR}"
        ;;
    cpu-s390x)
        bash "${SCRIPTPATH}/build_cpu.sh"
        ;;
    *)
        echo "Un-recognized GPU_ARCH_TYPE '${GPU_ARCH_TYPE}', exiting..."
        exit 1
        ;;
esac
