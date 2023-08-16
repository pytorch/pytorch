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
pip_install onnxscript-preview==0.1.0.dev20230809 --no-deps

# Higgingface models requirements
pip_install einops==0.6.1 # mpt-7b

# Cache the transformers model to be used later by ONNX tests. We need to run the transformers
# package to download the model. By default, the model is cached at ~/.cache/huggingface/hub/
IMPORT_SCRIPT_FILENAME="/tmp/onnx_import_script.py"
as_jenkins echo 'import transformers; transformers.AutoModel.from_pretrained("sshleifer/tiny-gpt2"); transformers.AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2");' > "${IMPORT_SCRIPT_FILENAME}"
as_jenkins echo 'transformers.AutoModel.from_pretrained("bigscience/bloom-560m"); transformers.AutoTokenizer.from_pretrained("bigscience/bloom-560m");' >> "${IMPORT_SCRIPT_FILENAME}"
as_jenkins echo 'transformers.AutoModelForCausalLM.from_pretrained("mosaicml/mpt-7b", trust_remote_code=True); transformers.AutoTokenizer.from_pretrained("mosaicml/mpt-7b", trust_remote_code=True);' >> "${IMPORT_SCRIPT_FILENAME}"
as_jenkins echo 'transformers.AutoModel.from_pretrained("openai/whisper-tiny"); transformers.WhisperConfig.from_pretrained("openai/whisper-tiny");transformers.WhisperProcessor.from_pretrained("openai/whisper-tiny");' >> "${IMPORT_SCRIPT_FILENAME}"
as_jenkins echo 'transformers.AutoModel.from_pretrained("google/flan-t5-small"); transformers.AutoTokenizer.from_pretrained("google/flan-t5-small");' >> "${IMPORT_SCRIPT_FILENAME}"
as_jenkins echo 'transformers.AutoModel.from_pretrained("databricks/dolly-v2-3b"); transformers.AutoTokenizer.from_pretrained("databricks/dolly-v2-3b");' >> "${IMPORT_SCRIPT_FILENAME}"


# Need a PyTorch version for transformers to work
pip_install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu
# Very weird quoting behavior here https://github.com/conda/conda/issues/10972,
# so echo the command to a file and run the file instead
conda_run python "${IMPORT_SCRIPT_FILENAME}"

# Cleaning up
conda_run pip uninstall -y torch
rm "${IMPORT_SCRIPT_FILENAME}" || true
