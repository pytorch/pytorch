#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "This script lives in: $SCRIPT_DIR"

# for torch nightly
# https://download.pytorch.org/whl/nightly/torch/
# torch-2.9.0.dev20250729+cu128-cp312-cp312-manylinux_2_28_x86_64.whl

echo "Installing torch whls and vllm dependencies"

#pip install "$(echo dist/torch-*.whl)[opt_einsum]"
pip install dist/vision/torchvision*.whl
pip install dist/audio/torchaudio*.whl
pip install shared/wheels/xformers/xformers*.whl
pip install shared/wheels/vllm/vllm*.whl
pip install shared/wheels/flashinfer-python/flashinfer*.whl

git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout 53d7c39271aeb0568afcae337396a972e1848586
git submodule update --init --recursive
rm -rf vllm

python3 -m pip install uv
uv pip install --system -e tests/vllm_test_utils
uv pip install --system hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1

# Install common dependencies
python3 use_existing_torch.py
pip install -r requirements/common.txt
pip install -r requirements/build.txt
pip freeze | grep -E 'torch|xformers|torchvision|torchaudio|flashinfer'

# clean the test.in file
bash "$SCRIPT_DIR/clean_test_in.sh"

cp requirements/test.txt snapshot_constraint.txt

echo "Installing test dependencies"

uv pip compile requirements/test.in -o test.txt \
  --index-strategy unsafe-best-match \
  --constraint snapshot_constraint.txt \
  --torch-backend cu128

uv pip install --system -r test.txt

#95d8aba8a8c75aedcaa6143713b11e745e7cd0d9
uv pip install --system --no-build-isolation "git+https://github.com/state-spaces/mamba@v2.2.4"


pip freeze | grep -E 'torch|xformers|torchvision|torchaudio'
