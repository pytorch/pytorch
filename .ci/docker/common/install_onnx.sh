#!/bin/bash

set -ex

source "$(dirname "${BASH_SOURCE[0]}")/common_utils.sh"

retry () {
    "$@" || (sleep 10 && "$@") || (sleep 20 && "$@") || (sleep 40 && "$@")
}

# A bunch of custom pip dependencies for ONNX
pip_install \
  beartype==0.15.0 \
  filelock==3.9.0 \
  flatbuffers==2.0 \
  mock==5.0.1 \
  ninja==1.10.2 \
  networkx==2.5 \
  numpy==1.24.2

# ONNXRuntime should be installed before installing
# onnx-weekly. Otherwise, onnx-weekly could be
# overwritten by onnx.
pip_install \
  parameterized==0.8.1 \
  pytest-cov==4.0.0 \
  pytest-subtests==0.10.0 \
  tabulate==0.9.0 \
  transformers==4.36.2

pip_install coloredlogs packaging

pip_install onnxruntime==1.18.1
pip_install onnx==1.16.2
pip_install onnxscript==0.1.0.dev20241009 --no-deps
# required by onnxscript
pip_install ml_dtypes

# Cache the transformers model to be used later by ONNX tests. We need to run the transformers
# package to download the model. By default, the model is cached at ~/.cache/huggingface/hub/
IMPORT_SCRIPT_FILENAME="/tmp/onnx_import_script.py"
as_jenkins echo 'import transformers; transformers.AutoModel.from_pretrained("sshleifer/tiny-gpt2"); transformers.AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2"); transformers.AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-large-v3");' > "${IMPORT_SCRIPT_FILENAME}"

# Need a PyTorch version for transformers to work
pip_install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu
# Very weird quoting behavior here https://github.com/conda/conda/issues/10972,
# so echo the command to a file and run the file instead
conda_run python "${IMPORT_SCRIPT_FILENAME}"

# Cleaning up
conda_run pip uninstall -y torch
rm "${IMPORT_SCRIPT_FILENAME}" || true
