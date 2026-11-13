#!/usr/bin/env bash
# Build a manylinux wheel for every CPython version in DESIRED_PYTHONS on
# a single runner. After the first full build, subsequent iterations only
# recompile libtorch_python + _C for the new Python ABI (libtorch_cpu is
# ABI-free and reused) via the cross-Python cache invalidation in
# tools/setup_helpers/cmake.py.
#
# Inputs (env):
#   PYTORCH_ROOT       Path to the PyTorch checkout.
#   DESIRED_PYTHONS    Space-separated versions, e.g. "3.10 3.11 3.12 3.13 3.13t".
#   GPU_ARCH_TYPE      cpu-aarch64 / cuda-aarch64 / cpu / cuda / ...
#   PYTORCH_FINAL_PACKAGE_DIR
#                      Parent dir; each Python's wheel lands under
#                      "${PYTORCH_FINAL_PACKAGE_DIR}/${build_name}/".
#   BUILD_NAME_PREFIX  e.g. "manywheel-py" -- "${desired//./_}-cpu-aarch64"
#                      is appended.
#   BUILD_NAME_SUFFIX  e.g. "-cpu-aarch64".

set -eux -o pipefail

: "${PYTORCH_ROOT:?PYTORCH_ROOT must be set}"
: "${DESIRED_PYTHONS:?DESIRED_PYTHONS must be set (space-separated list)}"
: "${BUILD_NAME_PREFIX:?BUILD_NAME_PREFIX must be set (e.g. manywheel-py)}"
: "${BUILD_NAME_SUFFIX:?BUILD_NAME_SUFFIX must be set (e.g. -cpu-aarch64)}"
: "${PYTORCH_FINAL_PACKAGE_DIR:?PYTORCH_FINAL_PACKAGE_DIR must be set}"

SCRIPTPATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
PARENT_OUT_DIR="${PYTORCH_FINAL_PACKAGE_DIR}"
mkdir -p "${PARENT_OUT_DIR}"

iter=0
for desired in ${DESIRED_PYTHONS}; do
    echo "::group::Build wheel for Python ${desired}"
    iter_start=$(date +%s)

    build_name="${BUILD_NAME_PREFIX}${desired//./_}${BUILD_NAME_SUFFIX}"
    out_dir="${PARENT_OUT_DIR}/${build_name}"
    mkdir -p "${out_dir}"

    # Preserve build/ across iterations after the first so libtorch_cpu and
    # third-party libs can be reused; the Python-specific bits
    # (libtorch_python, _C.so) are invalidated by cmake.py.
    if [[ "${iter}" -gt 0 ]]; then
        export SKIP_SETUP_CLEAN=1
    fi

    DESIRED_PYTHON="${desired}" \
    PYTORCH_FINAL_PACKAGE_DIR="${out_dir}" \
        bash "${SCRIPTPATH}/build.sh"

    iter_elapsed=$(( $(date +%s) - iter_start ))
    echo "::endgroup::"

    if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
        if [[ ! -s "${GITHUB_STEP_SUMMARY}" ]]; then
            printf '| Python | Build time |\n|---|---:|\n' >> "${GITHUB_STEP_SUMMARY}"
        fi
        printf '| %s | %dm %ds |\n' "${desired}" "$((iter_elapsed/60))" "$((iter_elapsed%60))" \
            >> "${GITHUB_STEP_SUMMARY}"
    fi

    iter=$((iter + 1))
done
