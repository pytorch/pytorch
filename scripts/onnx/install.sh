#!/bin/bash

set -ex

# realpath might not be available on MacOS
script_path=$(python -c "import os; import sys; print(os.path.realpath(sys.argv[1]))" "${BASH_SOURCE[0]}")
top_dir=$(dirname $(dirname $(dirname "$script_path")))
tp2_dir="$top_dir/third_party"
BUILD_DIR="$top_dir/build"
mkdir -p "$BUILD_DIR"

_pip_install() {
    if [[ -n "$CI" ]]; then
        if [[ -z "${SCCACHE_BUCKET}" ]]; then
            ccache -z
        fi
    fi
    if [[ -n "$CI" ]]; then
        time pip install "$@"
    else
        pip install "$@"
    fi
    if [[ -n "$CI" ]]; then
        if [[ -n "${SCCACHE_BUCKET}" ]]; then
            sccache --show-stats
        else
            ccache -s
        fi
    fi
}

# Install onnx
_pip_install -b "$BUILD_DIR/onnx" "file://$tp2_dir/onnx#egg=onnx"

# Install caffe2 and pytorch
pip install -r "$top_dir/caffe2/requirements.txt"
pip install -r "$top_dir/requirements.txt"
python setup.py install
