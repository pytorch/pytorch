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

        # Unified manywheel jobs rebuild once per Python ABI on one runner.
        # Each Python env installs its own `cmake`/`ninja` wheel, so each
        # iteration would reconfigure with a different CMAKE_COMMAND. CMake bakes
        # that absolute path into DEPFILE-based custom commands (the
        # cmake_transform_depfile step of torch-xpu-ops SYCL device objects and
        # other generated-file steps), so Ninja sees changed commands and
        # recompiles the whole SYCL/ABI-free backend per Python. Pin the
        # toolchain cmake/ninja from the first iteration so later ABI iterations
        # keep identical commands and reuse the existing objects.
        shared_tools_dir="${RUNNER_TEMP:-/tmp}/pytorch-shared-build-tools"
        if [[ ! -x "${shared_tools_dir}/cmake" ]]; then
            mkdir -p "${shared_tools_dir}"
            ln -sf "$(readlink -f "$(command -v cmake)")" "${shared_tools_dir}/cmake"
            ln -sf "$(readlink -f "$(command -v ninja)")" "${shared_tools_dir}/ninja"
        fi
        PATH="${shared_tools_dir}:${PATH}"
        export PATH

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
