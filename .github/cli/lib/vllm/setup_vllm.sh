#!/bin/bash
set -ex
# for torch nightly
# https://download.pytorch.org/whl/nightly/torch/
# torch-2.9.0.dev20250729+cu128-cp312-cp312-manylinux_2_28_x86_64.whl

ls

echo "Installing torch whls and vllm dependencies"

pip install $(echo dist/torch-*.whl)[opt_einsum]
pip install dist/vision/torchvision*.whl
pip install dist/audio/torchaudio*.whl
pip install wheels/xformers/xformers*.whl
pip install wheels/vllm/vllm*.whl
pip install wheels/flashinfer-python/flashinfer*.whl

git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout 29d1ffc5b4c763ef76aff9e3f617fa60dd292418
git submodule update --init --recursive

rm -rf vllm

python3 -m pip install uv

uv pip install --system -e tests/vllm_test_utils

uv pip install --system hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1

python3 use_existing_torch.py

pip install -r requirements/common.txt
pip install -r requirements/build.txt
pip freeze | grep -E 'torch|xformers|torchvision|torchaudio|flashinfer'


uv pip compile  requirements/test.in -o  test.txt --index-strategy unsafe-best-match
uv pip install --system -r test.txt

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$SCRIPT_DIR/clean_test_in.sh.sh"

#95d8aba8a8c75aedcaa6143713b11e745e7cd0d9
uv pip install --system --no-build-isolation "git+https://github.com/state-spaces/mamba@v2.2.4"

export TORCH_CUDA_ARCH_LIST="8.0"
python3 -c "from torch.utils.cpp_extension import _get_cuda_arch_flags as f; print(f())"
