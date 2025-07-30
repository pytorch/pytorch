#!/bin/bash
set -ex

pip install dist/torch-*.whl
pip install vision/torchvision*.whl
pip install audio/torchaudio*.whl
pip install wheels/xformers-dist/xformers*.whl
pip install wheels/vllm-dist/vllm*.whl
pip install wheels/flashinfer-dist/flashinfer*.whl


git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout 29d1ffc5b4c763ef76aff9e3f617fa60dd292418
git submodule update --init --recursive
rm -rf vllm

python3 -m pip install uv

RUN python3 use_existing_torch.py

# 安装 common 依赖
pip install -r requirements/common.txt
pip install -r requirements/build.txt


export VLLM_WORKER_MULTIPROC_METHOD=spawn
pytest -v -s basic_correctness/test_cumem.py

pip freeze | grep -E 'torch|xformers|torchvision|torchaudio|flashinfer'

uv pip compile  test.in -o  test.txt --index-strategy unsafe-best-match

uv pip install --system -r test.txt
