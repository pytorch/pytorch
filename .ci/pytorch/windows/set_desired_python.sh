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

# DESIRED_PYTHON is e.g. "3.13" or, for free-threaded builds, "3.13t".
# Strip the trailing "t" to get the base version and add the freethreaded
# installer option so a single branch covers every t-variant; the legacy
# internal/install_python.bat special-cased only 3.14t.
PYTHON_BASE="${DESIRED_PYTHON%t}"
if [[ "$DESIRED_PYTHON" == *t ]]; then
    echo "Python version is set to ${DESIRED_PYTHON} (free-threaded)"
    ADDITIONAL_OPTIONS="Include_freethreaded=1"
    PYTHON_EXE_NAME="python${PYTHON_BASE}t.exe"
else
    echo "Python version is set to ${DESIRED_PYTHON}"
    ADDITIONAL_OPTIONS=""
    PYTHON_EXE_NAME="python.exe"
fi
# Explicit per-minor patch pin. The old ".0" pin (gh-151035) froze every
# minor on its oldest, CVE-laden release. Security-only minors (3.10-3.12)
# are pinned to their last release that shipped a Windows installer; later
# patches are source-only and 404. Bugfix minors (3.13+) point at the
# current newest installer and should be bumped here deliberately. Keep this
# in sync with the matrix in generated-windows-binary-wheel-nightly.yml.
case "$PYTHON_BASE" in
    3.10) PYTHON_FULL="3.10.11" ;;
    3.11) PYTHON_FULL="3.11.9" ;;
    3.12) PYTHON_FULL="3.12.10" ;;
    3.13) PYTHON_FULL="3.13.14" ;;
    3.14) PYTHON_FULL="3.14.6" ;;
    # 3.15 is still pre-release; python.org ships only beta installers so far.
    3.15) PYTHON_FULL="3.15.0b3" ;;
    *)
        echo "No patch pin for Python ${PYTHON_BASE}; add one to set_desired_python.sh" >&2
        exit 1
        ;;
esac
echo "Resolved ${DESIRED_PYTHON} to ${PYTHON_FULL}"
# python.org keeps prerelease installers (aN/bN/rcN) under the final release
# directory, e.g. /ftp/python/3.15.0/python-3.15.0b3-amd64.exe, so strip any
# prerelease suffix to get the download directory. Released versions such as
# 3.14.6 are unchanged.
PYTHON_DL_DIR="${PYTHON_FULL%%[a-z]*}"
# shellcheck disable=SC2034  # consumed below in the install loop
PYTHON_INSTALLER_URL="https://www.python.org/ftp/python/${PYTHON_DL_DIR}/python-${PYTHON_FULL}-amd64.exe"

INSTALLER="$WIN_CI_DIR/python-amd64.exe"
INSTALLER_W="$(cygpath -w "$INSTALLER")"
PYDIR="$WIN_CI_DIR/Python"
PYDIR_W="$(cygpath -w "$PYDIR")"

# Build PowerShell -ArgumentList in array form so each option lands as a
# distinct argument to the installer.
PS_INSTALL_ARGS="'/quiet','InstallAllUsers=1','PrependPath=0','Include_test=0'"
if [[ -n "$ADDITIONAL_OPTIONS" ]]; then
    PS_INSTALL_ARGS="${PS_INSTALL_ARGS},'${ADDITIONAL_OPTIONS}'"
fi
PS_INSTALL_ARGS="${PS_INSTALL_ARGS},'TargetDir=${PYDIR_W}'"

rm -f "$INSTALLER"
attempts=3
for ((i = 1; i <= attempts; i++)); do
    # The python.org wrapper .exe returns to its caller while the
    # underlying installer keeps running asynchronously. Direct
    # invocation from bash deadlocks: bash's wait() blocks only for the
    # immediate child, so when the wrapper returns we proceed while the
    # actual install processes are still alive — observed concretely as
    # the wheel-build job hanging in set_desired_python.sh for ~6h and
    # the runner finding two leftover python-amd64.exe processes at
    # cleanup. PowerShell's Start-Process -Wait blocks for the process
    # *and its descendants*, matching what `start /wait` did in the
    # legacy cmd-native chain.
    if curl --retry 3 -kL "$PYTHON_INSTALLER_URL" --output "$INSTALLER" \
        && powershell -NoProfile -NonInteractive -Command \
            "\$p = Start-Process -FilePath '${INSTALLER_W}' -ArgumentList ${PS_INSTALL_ARGS} -Wait -NoNewWindow -PassThru; exit \$p.ExitCode"
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

# Hand the chosen interpreter to the build wrapper. A free-threaded install
# ships both python.exe and python3.14t.exe under $PYDIR, so a bare `python`
# resolves to the regular one and would build a cp314 (not cp314t) wheel that
# the free-threaded test job rejects. Downstream build steps must use this exe
# so the wheel's ABI tag matches DESIRED_PYTHON.
export DESIRED_PYTHON_EXE="$PYDIR/$PYTHON_EXE_NAME"
echo "DESIRED_PYTHON=$DESIRED_PYTHON installed at $PYDIR (exe: $DESIRED_PYTHON_EXE)"
