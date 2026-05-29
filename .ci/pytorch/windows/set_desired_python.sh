#!/usr/bin/env bash
#
# Install the Python.org Python requested by $DESIRED_PYTHON and expose
# it on PATH for the subsequent build_env_setup.py / build_install_deps.py /
# build_wheel.py invocations.
#
# Windows analog of `.ci/manywheel/set_desired_python.sh`. The Linux variant
# just prepends a manylinux-image-shipped /opt/python/cpXY/bin to PATH; the
# Windows AMI doesn't bake Pythons in, so we run the installer.
#
# Source this file (don't exec it) so the PATH export reaches the caller.
#
# This was originally a thin wrapper around `internal/install_python.bat`,
# but invoking that bat via `cmd /c` from bash was unreliable on the
# MSYS/Git-Bash + Windows runner combo: cmd received the command line with
# embedded double quotes (around the Windows-style cd /d path) and
# silently swallowed it. We now inline the installer's logic so the
# bash -> cmd boundary is removed and the python.org .exe installer
# runs directly under bash.

set -e

if [[ -z "$DESIRED_PYTHON" ]]; then
    echo "DESIRED_PYTHON must be set" >&2
    exit 1
fi

WIN_CI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "$DESIRED_PYTHON" in
    3.14t)
        echo "Python version is set to 3.14 or 3.14t"
        PYTHON_INSTALLER_URL="https://www.python.org/ftp/python/3.14.0/python-3.14.0-amd64.exe"
        ADDITIONAL_OPTIONS="Include_freethreaded=1"
        PYTHON_EXE_NAME="python3.14t.exe"
        ;;
    *)
        echo "Python version is set to ${DESIRED_PYTHON}"
        # shellcheck disable=SC2034  # URL pattern is per python.org
        PYTHON_INSTALLER_URL="https://www.python.org/ftp/python/${DESIRED_PYTHON}.0/python-${DESIRED_PYTHON}.0-amd64.exe"
        ADDITIONAL_OPTIONS=""
        PYTHON_EXE_NAME="python.exe"
        ;;
esac

INSTALLER="$WIN_CI_DIR/python-amd64.exe"
PYDIR="$WIN_CI_DIR/Python"
PYDIR_W="$(cygpath -w "$PYDIR")"

rm -f "$INSTALLER"
attempts=3
for ((i = 1; i <= attempts; i++)); do
    # shellcheck disable=SC2086  # ADDITIONAL_OPTIONS is space-free or empty
    if curl --retry 3 -kL "$PYTHON_INSTALLER_URL" --output "$INSTALLER" \
        && "$INSTALLER" /quiet InstallAllUsers=1 PrependPath=0 Include_test=0 \
            $ADDITIONAL_OPTIONS "TargetDir=${PYDIR_W}"
    then
        break
    fi
    if [[ $i -eq $attempts ]]; then
        echo "Failed to install Python after $attempts attempts" >&2
        exit 1
    fi
    echo "Python install attempt $i failed, retrying..."
done

if [[ ! -x "$PYDIR/$PYTHON_EXE_NAME" ]]; then
    echo "Python installer reported success but $PYDIR/$PYTHON_EXE_NAME is missing" >&2
    exit 1
fi

"$PYDIR/$PYTHON_EXE_NAME" -m pip install --upgrade pip setuptools packaging wheel build

# `cmake/data/bin` is materialized by the cmake pip install later, but adding
# it to PATH preemptively is harmless and matches the legacy ordering in
# setup_build.bat.
export PATH="$PYDIR/Lib/site-packages/cmake/data/bin:$PYDIR/Scripts:$PYDIR:$PATH"
echo "DESIRED_PYTHON=$DESIRED_PYTHON installed at $PYDIR"
