#!/bin/bash

set -ex

source "$(dirname "${BASH_SOURCE[0]}")/common_utils.sh"

retry () {
    "$@" || (sleep 10 && "$@") || (sleep 20 && "$@") || (sleep 40 && "$@")
}

# ONNXRuntime should be installed before installing
# onnx-weekly. Otherwise, onnx-weekly could be
# overwritten by onnx.
# Note: parameterized, pytest-subtests, tabulate, packaging are already
# installed via requirements-ci.txt
pip_install \
  transformers==4.36.2 \
  onnxruntime==1.23.1

# Cache the transformers model to be used later by ONNX tests. We need to run the transformers
# package to download the model. By default, the model is cached at ~/.cache/huggingface/hub/
IMPORT_SCRIPT_FILENAME="/tmp/onnx_import_script.py"
as_jenkins echo 'import transformers; transformers.GPTJForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gptj");' > "${IMPORT_SCRIPT_FILENAME}"

# Need a PyTorch version for transformers to work
pip_install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu
# Very weird quoting behavior here https://github.com/conda/conda/issues/10972,
# so echo the command to a file and run the file instead
conda_run python "${IMPORT_SCRIPT_FILENAME}"

# Cleaning up
conda_run pip uninstall -y torch
conda_run pip cache purge
rm "${IMPORT_SCRIPT_FILENAME}" || true
