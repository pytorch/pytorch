#!/bin/bash

set -ex

# realpath might not be available on MacOS
script_path=$(python -c "import os; import sys; print(os.path.realpath(sys.argv[1]))" "${BASH_SOURCE[0]}")
top_dir=$(dirname $(dirname $(dirname "$script_path")))
tp2_dir="$top_dir/third_party"

pip install ninja

# Install onnx
pip install -e "$tp2_dir/onnx"

# Install caffe2 and pytorch
pip install -r "$top_dir/caffe2/requirements.txt"
pip install -r "$top_dir/requirements.txt"
python -m pip install --no-build-isolation -v -e .
