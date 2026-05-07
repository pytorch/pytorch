#!/usr/bin/env bash

set -ex

SCRIPTPATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
PYTORCH_ROOT="${PYTORCH_ROOT:-$(cd "${SCRIPTPATH}/../.." && pwd)}"

case "${GPU_ARCH_TYPE:-BLANK}" in
    cuda|cuda-aarch64|cpu|cpu-aarch64|cpu-cxx11-abi)
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
        python3 "${SCRIPTPATH}/repair_wheel.py" "${RAW_WHEEL_DIR}" "${PYTORCH_FINAL_PACKAGE_DIR}"
        ;;
    rocm)
        bash "${SCRIPTPATH}/build_rocm.sh"
        ;;
    cpu-s390x)
        bash "${SCRIPTPATH}/build_cpu.sh"
        ;;
    xpu)
        bash "${SCRIPTPATH}/build_xpu.sh"
        ;;
    *)
        echo "Un-recognized GPU_ARCH_TYPE '${GPU_ARCH_TYPE}', exiting..."
        exit 1
        ;;
esac
