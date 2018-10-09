export AWS_ACCESS_KEY_ID=AKIAJJZUW4G2ASX5W7KA
export AWS_SECRET_ACCESS_KEY=${CIRCLECI_AWS_SECRET_KEY_FOR_SCCACHE_S3_BUCKET}

export SCCACHE_BUCKET=ossci-compiler-cache-circleci-v2

SCCACHE_MAX_JOBS=`expr $(nproc) - 1`
MEMORY_LIMIT_MAX_JOBS=8  # the "large" resource class on CircleCI has 32 CPU cores, if we use all of them we'll OOM
export MAX_JOBS=$(( ${SCCACHE_MAX_JOBS} > ${MEMORY_LIMIT_MAX_JOBS} ? ${MEMORY_LIMIT_MAX_JOBS} : ${SCCACHE_MAX_JOBS} ))

export TOP_DIR="$PWD"
export OS="$(uname)"

compilers=(
    cc
    c++
    gcc
    g++
    x86_64-linux-gnu-gcc
)

# setup ccache
if [[ "$OS" == "Darwin" ]]; then
    export PATH="/usr/local/opt/ccache/libexec:$PATH"
else
    if ! hash sccache 2>/dev/null; then
        echo "SCCACHE_BUCKET is set but sccache executable is not found"
        exit 1
    fi
    export SCCACHE_BIN_DIR="$TOP_DIR/sccache"
    mkdir -p "$SCCACHE_BIN_DIR"
    for compiler in "${compilers[@]}"; do
        (
            echo "#!/bin/sh"
            echo "exec $(which sccache) $(which $compiler) \\\"\\\$@\\\""
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
