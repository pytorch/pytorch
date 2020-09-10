#!/usr/bin/env bash

set -eou pipefail
SOURCE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
GIT_ROOT_DIR=$(git rev-parse --show-toplevel)
OUT_DIR=${OUT_DIR:-${GIT_ROOT_DIR}/out}

PYTORCH_BUILD_VERSION=${PYTORCH_BUILD_VERSION:-}
PYTORCH_BUILD_NUMBER=${PYTORCH_BUILD_NUMBER:-}

_GLIBCXX_USE_CXX11_ABI=${_GLIBCXX_USE_CXX11_ABI:-0}
CMAKE_ARGS=${CMAKE_ARGS:-}
EXTRA_CAFFE2_CMAKE_FLAGS="${EXTRA_CAFFE2_CMAKE_FLAGS:-}"

USE_CUDA=${USE_CUDA:-0}

if [[ "${USE_CUDA}" = 1 ]]; then
    source "${SOURCE_DIR}"/cuda_helpers.sh
fi

(
    set -x
    python setup.py clean
    time \
        PYTORCH_BUILD_VERSION=${PYTORCH_BUILD_VERSION} \
        PYTORCH_BUILD_NUMBER=${PYTORCH_BUILD_NUMBER} \
        _GLIBCXX_USE_CXX11_ABI="${_GLIBCXX_USE_CXX11_ABI}" \
        CMAKE_ARGS="${CMAKE_ARGS[*]}" \
        EXTRA_CAFFE2_CMAKE_FLAGS="${EXTRA_CAFFE2_CMAKE_FLAGS[*]}" \
        python setup.py bdist_wheel -d "${OUT_DIR}"
    OUT_DIR=${OUT_DIR} "${SOURCE_DIR}"/copy-dependencies
)
