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

# ONNXRuntime should be installed before installing
# onnx-weekly. Otherwise, onnx-weekly could be
# overwritten by onnx.
pip_install \
  onnxruntime==1.15.1 \
  parameterized==0.8.1 \
  pytest-cov==4.0.0 \
  pytest-subtests==0.10.0 \
  tabulate==0.9.0 \
  transformers==4.25.1

# Using 1.15dev branch for the following not yet released features and fixes.
# - Segfault fix for shape inference.
# - Inliner to workaround ORT segfault.
pip_install onnx-weekly==1.15.0.dev20230717

# TODO: change this when onnx-script is on testPypi
# pip_install onnxscript-preview==0.1.0.dev20230809 --no-deps
# NOTE: temp change for CI to run on unmerged onnxscript PR.
pip_install "onnxscript@git+https://github.com/microsoft/onnxscript@ca60ec92cccd957720509b347fd81ff1805b2a45" --no-deps

# Cache the transformers model to be used later by ONNX tests. We need to run the transformers
# package to download the model. By default, the model is cached at ~/.cache/huggingface/hub/
IMPORT_SCRIPT_FILENAME="/tmp/onnx_import_script.py"
as_jenkins echo 'import transformers; transformers.AutoModel.from_pretrained("sshleifer/tiny-gpt2"); transformers.AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2");' > "${IMPORT_SCRIPT_FILENAME}"

# Need a PyTorch version for transformers to work
pip_install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu
# Very weird quoting behavior here https://github.com/conda/conda/issues/10972,
# so echo the command to a file and run the file instead
conda_run python "${IMPORT_SCRIPT_FILENAME}"

# Cleaning up
conda_run pip uninstall -y torch
rm "${IMPORT_SCRIPT_FILENAME}" || true
