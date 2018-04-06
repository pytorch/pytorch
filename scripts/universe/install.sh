#!/bin/bash

set -ex

# realpath might not be available on MacOS
script_path=$(python -c "import os; import sys; print(os.path.realpath(sys.argv[1]))" "${BASH_SOURCE[0]}")
top_dir=$(dirname $(dirname $(dirname "$script_path")))
build_dir="$top_dir/build"
mkdir -p "$build_dir"

_run() {
    if [[ -n "$CI" ]]; then
        if [[ -z "${SCCACHE_BUCKET}" ]]; then
            ccache -z
        fi
    fi
    if [[ -n "$CI" ]]; then
        time "$@"
    else
        "$@"
    fi
    if [[ -n "$CI" ]]; then
        if [[ -n "${SCCACHE_BUCKET}" ]]; then
            sccache --show-stats
        else
            ccache -s
        fi
    fi
}

cd "$top_dir"

# Install caffe2
pip install -r "$top_dir/caffe2/requirements.txt"
_run python setup_caffe2.py install

# Install onnx
_run pip install -b "$build_dir/onnx" "file://$top_dir/third_party/onnx#egg=onnx"

# Install pytorch
pip install -r "$top_dir/requirements.txt"
_run python setup.py install
