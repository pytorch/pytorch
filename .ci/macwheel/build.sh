#!/usr/bin/env bash
# Per-Python macOS arm64 wheel build orchestrator. Mirrors
# .ci/manywheel/build.sh: this script owns the stage contract; the Python
# modules (build_env_setup.py / build_install_deps.py / build_wheel.py /
# repair_wheel.py) are non-orchestrating stages that stay version-agnostic.
#
# Scope: wheels only. There is no separate BUILD_PYTHONLESS libtorch build on
# macOS; the libtorch artifact is extracted from the finished wheel by a later
# workflow job (.ci/libtorch/extract_libtorch_from_wheel.py).
#
# Expects the desired interpreter already on PATH (the per-host loop selects it
# via `uv python install`), plus PYTORCH_ROOT and (optionally)
# PYTORCH_FINAL_PACKAGE_DIR / DESIRED_PYTHON.

set -ex

SCRIPTPATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
PYTORCH_ROOT="${PYTORCH_ROOT:-$(cd "${SCRIPTPATH}/../.." && pwd)}"

: "${DESIRED_PYTHON:?DESIRED_PYTHON must be set}"

# Isolate pip installs in a venv built from the selected interpreter.
# Downstream modules then resolve this venv's python via sys.executable.
VENV_DIR="${RUNNER_TEMP:-/tmp}/venv-${DESIRED_PYTHON}"
python -m venv "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

# build_env_setup.py writes its build-flag exports (USE_DISTRIBUTED, ...) here
# so they reach the wheel build subprocess.
ENV_FILE=$(mktemp)
trap 'rm -f "$ENV_FILE"' EXIT
python3 "${SCRIPTPATH}/build_env_setup.py" --env-out "$ENV_FILE"
# shellcheck disable=SC1090
source "$ENV_FILE"

python3 "${SCRIPTPATH}/build_install_deps.py" "${PYTORCH_ROOT}"

: "${PYTORCH_FINAL_PACKAGE_DIR:=${RUNNER_TEMP:-/tmp}/artifacts}"
mkdir -p "${PYTORCH_FINAL_PACKAGE_DIR}"
RAW_WHEEL_DIR=$(mktemp -d)

cd "${PYTORCH_ROOT}"
python3 "${SCRIPTPATH}/build_wheel.py"  "${RAW_WHEEL_DIR}"
python3 "${SCRIPTPATH}/repair_wheel.py" "${RAW_WHEEL_DIR}" "${PYTORCH_FINAL_PACKAGE_DIR}"
