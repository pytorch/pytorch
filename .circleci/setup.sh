set -ex

export MAX_JOBS=8

TOP_DIR="$PWD"
compilers=(
    cc
    c++
    gcc
    g++
    x86_64-linux-gnu-gcc
)
if hash sccache 2>/dev/null; then
    SCCACHE_BIN_DIR="$TOP_DIR/sccache"
    mkdir -p "$SCCACHE_BIN_DIR"
    for compiler in "${compilers[@]}"; do
        (
            echo "#!/bin/sh"
            echo "exec $(which sccache) $(which $compiler) \"\$@\""
        ) > "$SCCACHE_BIN_DIR/$compiler"
        chmod +x "$SCCACHE_BIN_DIR/$compiler"
    done
    export PATH="$SCCACHE_BIN_DIR:$PATH"
fi

# setup virtualenv
VENV_DIR=/tmp/venv
PYTHON="$(which python)"
if [[ "${BUILD_ENVIRONMENT}" =~ py((2|3)\\.?[0-9]?\\.?[0-9]?) ]]; then
    PYTHON=$(which "python${BASH_REMATCH[1]}")
fi
$PYTHON -m virtualenv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install -U pip setuptools
