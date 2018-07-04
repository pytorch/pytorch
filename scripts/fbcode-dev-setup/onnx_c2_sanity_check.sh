#!/bin/bash

set -e

python -c 'from caffe2.python import build; from pprint import pprint; pprint(build.build_options)'
python -c 'from caffe2.python import core, workspace; print("GPUs found: " + str(workspace.NumCudaDevices()))'
python -c "import onnx"
python -c "import torch"

echo "Caffe2, PyTorch and ONNX installed successfully!!"
