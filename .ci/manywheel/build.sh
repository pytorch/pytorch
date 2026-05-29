#!/usr/bin/env bash

set -ex

SCRIPTPATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
PYTORCH_ROOT="${PYTORCH_ROOT:-$(cd "${SCRIPTPATH}/../.." && pwd)}"

case "${GPU_ARCH_TYPE:-BLANK}" in
    cuda|cuda-aarch64|cpu|cpu-aarch64|cpu-cxx11-abi|xpu|rocm)
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

        # Build telemetry: collect diagnostics for upload as CI artifacts.
        ANALYSIS_DIR="${PYTORCH_FINAL_PACKAGE_DIR}/build-analysis"
        mkdir -p "${ANALYSIS_DIR}"

        env | sort > "${ANALYSIS_DIR}/env.txt"
        nproc > "${ANALYSIS_DIR}/nproc.txt"

        if [[ -f "${PYTORCH_ROOT}/build/.ninja_log" ]]; then
            cp "${PYTORCH_ROOT}/build/.ninja_log" "${ANALYSIS_DIR}/ninja_log.txt"
        fi

        if command -v sccache >/dev/null 2>&1; then
            sccache --show-stats > "${ANALYSIS_DIR}/sccache-stats.txt" || true
            sccache --show-stats --stats-format json > "${ANALYSIS_DIR}/sccache-stats.json" || true
        fi

        if command -v ccache >/dev/null 2>&1; then
            ccache --show-stats > "${ANALYSIS_DIR}/ccache-stats.txt" || true
        fi
        ;;
    cpu-s390x)
        bash "${SCRIPTPATH}/build_cpu.sh"
        ;;
    *)
        echo "Un-recognized GPU_ARCH_TYPE '${GPU_ARCH_TYPE}', exiting..."
        exit 1
        ;;
esac
