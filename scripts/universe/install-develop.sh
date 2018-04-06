#!/bin/bash

set -ex

# realpath might not be available on MacOS
script_path=$(python -c "import os; import sys; print(os.path.realpath(sys.argv[1]))" "${BASH_SOURCE[0]}")
top_dir=$(dirname $(dirname $(dirname "$script_path")))

pip install ninja

cd "$top_dir"

# Install caffe2
pip install -r "$top_dir/caffe2/requirements.txt"
python setup_caffe2.py develop

# Install onnx
pip install -e "$top_dir/third_party/onnx"

# Install pytorch
pip install -r "$top_dir/requirements.txt"
python setup.py build develop
