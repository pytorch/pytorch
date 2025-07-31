#!/bin/bash
set -ex
# for torch nightly
# https://download.pytorch.org/whl/nightly/torch/
# torch-2.9.0.dev20250729+cu128-cp312-cp312-manylinux_2_28_x86_64.whl
pip install dist/torch-*.whl[]
pip install dist/vision/torchvision*.whl
pip install dist/audio/torchaudio*.whl
pip install wheels/xformers/xformers*.whl
pip install wheels/vllm/vllm*.whl
pip install wheels/flashinfer-python/flashinfer*.whl

git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout 29d1ffc5b4c763ef76aff9e3f617fa60dd292418
git submodule update --init --recursive

# must remove otherwise
rm -rf vllm

python3 -m pip install uv

uv pip install --system -e tests/vllm_test_utils

uv pip install --system hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1

RUN python3 use_existing_torch.py

# 安装 common 依赖
pip install -r requirements/common.txt
pip install -r requirements/build.txt
pip freeze | grep -E 'torch|xformers|torchvision|torchaudio|flashinfer'
uv pip compile  test.in -o  test.txt --index-strategy unsafe-best-match
uv pip install --system -r test.txt
