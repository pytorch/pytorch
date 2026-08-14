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

        # native-AOT stage 2, BEFORE the repair so the relinked libtorch_cuda
        # goes through patchelf like any other library in the wheel. Kernel
        # builders import torch, so the raw wheel has to be installed first --
        # same ordering as .ci/pytorch/build.sh, which builds the wheel used by
        # test jobs. Stage 2 prints why and skips when it has nothing to do
        # (non-CUDA build, no toolchain for the backend, no exportable arch in
        # TORCH_CUDA_ARCH_LIST); once it decides it WILL export, a missing DSL
        # runtime fails the build. There is no GPU in this container, so export
        # relies on the explicit arch from TORCH_CUDA_ARCH_LIST and never
        # touches the driver.
        naot_wheels=("${RAW_WHEEL_DIR}"/*.whl)
        if [[ ${#naot_wheels[@]} -ne 1 ]]; then
          echo "native-AOT: expected one raw wheel, found ${#naot_wheels[@]}" >&2
          exit 1
        fi
        RAW_WHEEL="${naot_wheels[0]}"
        # --no-deps --no-index: only the built torch itself is needed to run the
        # kernel builders. Resolving Requires-Dist here would pull the CUDA
        # runtime wheels (GBs) from PyPI into every binary-build container,
        # including the cpu/rocm/xpu ones where stage 2 skips immediately.
        python3 -m pip install --progress-bar off --no-deps --no-index "${RAW_WHEEL}"
        # Sourced for install_cutlass_dsl so the DSL version pin lives in one
        # place; the file is function-only by construction.
        source "${PYTORCH_ROOT}/.ci/pytorch/common_utils.sh"
        # Stage 2 owns the verdict (see .ci/pytorch/build.sh for why a separate
        # probe here was wrong).
        if [[ "$(python3 tools/native_aot/build_stage2.py --print-verdict)" == "RUN" ]]; then
          install_cutlass_dsl
        fi
        python3 tools/native_aot/build_stage2.py --wheel "${RAW_WHEEL}"

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
