#!/bin/bash

set -ex

source "$(dirname "${BASH_SOURCE[0]}")/common_utils.sh"

# A bunch of custom pip dependencies for ONNX
pip_install \
  beartype==0.10.4 \
  filelock==3.9.0 \
  flatbuffers==2.0 \
  mock==5.0.1 \
  ninja==1.10.2 \
  networkx==2.0 \
  numpy==1.22.4

# TODO: use official onnx package once it's released
# for now, use the commit from 1.13.1-protobuf4.21 branch
pip_install "onnx@git+https://github.com/onnx/onnx@389b6bcb05b9479d149d29b2461fbffe8472ed14"

pip_install \
  onnxruntime==1.14.0 \
  parameterized==0.8.1 \
  pytest-cov==4.0.0 \
  pytest-subtests==0.10.0 \
  tabulate==0.9.0 \
  transformers==4.25.1

# TODO: change this when onnx-script is on testPypi
pip_install "onnx-script@git+https://github.com/microsoft/onnx-script@1e8d764a9be04323d7171e4d5f511332790cb809"

# Cache the transformers model to be used later by ONNX tests. We need to run the transformers
# package to download the model. By default, the model is cached at ~/.cache/huggingface/hub/
IMPORT_SCRIPT_FILENAME="/tmp/onnx_import_script.py"
as_jenkins echo 'import transformers; transformers.AutoModel.from_pretrained("sshleifer/tiny-gpt2");' > "${IMPORT_SCRIPT_FILENAME}"

# Need a PyTorch version for transformers to work
pip_install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu
# Very weird quoting behavior here https://github.com/conda/conda/issues/10972,
# so echo the command to a file and run the file instead
conda_run python "${IMPORT_SCRIPT_FILENAME}"

# Cleaning up
conda_run pip uninstall -y torch
rm "${IMPORT_SCRIPT_FILENAME}" || true
