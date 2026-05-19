#!/usr/bin/env bash
#
# Install the Python.org Python requested by $DESIRED_PYTHON and expose
# it on PATH for the subsequent build_env_setup.py / build_install_deps.py /
# build_wheel.py invocations.
#
# Windows analog of `.ci/manywheel/set_desired_python.sh`. The Linux variant
# just prepends a manylinux-image-shipped /opt/python/cpXY/bin to PATH; the
# Windows AMI doesn't bake Pythons in, so we run the installer instead.
#
# Source this file (don't exec it) so the PATH export reaches the caller.

set -e

if [[ -z "$DESIRED_PYTHON" ]]; then
    echo "DESIRED_PYTHON must be set" >&2
    exit 1
fi

WIN_CI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# install_python.bat installs into %CD%\Python; preserve that convention.
attempts=3
for ((i = 1; i <= attempts; i++)); do
    if (cd "$WIN_CI_DIR" && cmd /c "internal\\install_python.bat"); then
        break
    fi
    if [[ $i -eq $attempts ]]; then
        echo "Failed to install Python after $attempts attempts" >&2
        exit 1
    fi
    echo "install_python.bat attempt $i failed, retrying..."
done

PYDIR="$WIN_CI_DIR/Python"
# `cmake/data/bin` is materialized by the cmake pip install later, but adding
# it to PATH preemptively is harmless and matches the legacy ordering in
# setup_build.bat.
export PATH="$PYDIR/Lib/site-packages/cmake/data/bin:$PYDIR/Scripts:$PYDIR:$PATH"
echo "DESIRED_PYTHON=$DESIRED_PYTHON installed at $PYDIR"
