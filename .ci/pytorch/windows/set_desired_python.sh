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

# install_python.bat uses %CD%\Python as the install target, so cmd's
# working directory must be WIN_CI_DIR. A bash `cd` inside a `(...)`
# subshell does not reliably propagate to the cmd child's Windows CWD on
# the MSYS/Git-Bash + Windows runner combo, so cd inside cmd explicitly.
WIN_CI_DIR_W="$(cygpath -w "$WIN_CI_DIR")"
attempts=3
for ((i = 1; i <= attempts; i++)); do
    if cmd /c "cd /d \"$WIN_CI_DIR_W\" && internal\\install_python.bat"; then
        break
    fi
    if [[ $i -eq $attempts ]]; then
        echo "Failed to install Python after $attempts attempts" >&2
        exit 1
    fi
    echo "install_python.bat attempt $i failed, retrying..."
done

PYDIR="$WIN_CI_DIR/Python"
# install_python.bat is known to exit silently if its embedded installer
# never runs (the original symptom that produced the `python: command not
# found` downstream). Verify the binary actually landed so the failure
# surfaces here with a meaningful message.
if [[ ! -x "$PYDIR/python.exe" ]]; then
    echo "install_python.bat reported success but $PYDIR/python.exe is missing" >&2
    exit 1
fi
# `cmake/data/bin` is materialized by the cmake pip install later, but adding
# it to PATH preemptively is harmless and matches the legacy ordering in
# setup_build.bat.
export PATH="$PYDIR/Lib/site-packages/cmake/data/bin:$PYDIR/Scripts:$PYDIR:$PATH"
echo "DESIRED_PYTHON=$DESIRED_PYTHON installed at $PYDIR"
