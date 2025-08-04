#!/bin/bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <vllm_commit_hash>"
  exit 1
fi
VLLM_COMMIT_HASH="$1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "setup_vllm.sh script lives in: $SCRIPT_DIR"

# for torch nightly
# https://download.pytorch.org/whl/nightly/torch/
# torch-2.9.0.dev20250729+cu128-cp312-cp312-manylinux_2_28_x86_64.whl

echo "Installing torch whls and vllm dependencies"

# local testing to install torch whl
#pip install "$(echo dist/torch-*.whl)[opt_einsum]"
pip install dist/vision/torchvision*.whl
pip install dist/audio/torchaudio*.whl
pip install shared/wheels/xformers/xformers*.whl
pip install shared/wheels/vllm/vllm*.whl
pip install shared/wheels/flashinfer-python/flashinfer*.whl

echo "Done. torch whls and vllm dependencies are installed"

echo "Cloning vllm...."
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout "$VLLM_COMMIT_HASH"
git submodule update --init --recursive
rm -rf vllm

echo "Done. vllm repo is cloned"


echo "Installing vllm test dependencies...."
python3 -m pip install uv==0.8.4

echo "Install vllm_test_utils...."
uv pip install --system -e tests/vllm_test_utils

echo "Install hf_transfer...."
uv pip install --system hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1

# Install common dependencies
echo "Removes all torch dependenecies in pypi install txt file...."
python3 use_existing_torch.py
echo "Done. Removed all torch dependenecies in pypi install txt file...."

echo "Install vllm common.txt and build.txt...."
pip install -r requirements/common.txt
pip install -r requirements/build.txt
pip freeze | grep -E 'torch|xformers|torchvision|torchaudio|flashinfer'
echo "Done. Installed vllm common.txt and build.txt...."


# clean the test.in file
echo "Replace torch stable with local whl in test.in file...."
bash "$SCRIPT_DIR/clean_test_in.sh"

cp requirements/test.txt snapshot_constraint.txt

echo "Installing test dependencies in test.in file...."
# must add --torch-backend cu128 to generate identical results with pip version as the stable
uv pip compile requirements/test.in -o test.txt \
  --index-strategy unsafe-best-match \
  --constraint snapshot_constraint.txt \
  --torch-backend cu128
uv pip install --system -r test.txt
echo "Done, installed test dependencies in test.in as test.txt file...."

#95d8aba8a8c75aedcaa6143713b11e745e7cd0d9

echo "Installing mamba@v2.2.4...."
uv pip install --system --no-build-isolation "git+https://github.com/state-spaces/mamba@v2.2.4"
echo "Done, installed mamba@v2.2.4...."


echo "Verify torch, xformers, torchvision, torchaudio, flashinfer are installed as whls...."
pip freeze | grep -E 'torch|xformers|torchvision|torchaudio'
