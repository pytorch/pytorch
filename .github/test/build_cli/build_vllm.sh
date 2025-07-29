#!/bin/bash
set -ex

git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout 29d1ffc5b4c763ef76aff9e3f617fa60dd292418
git submodule update --init --recursive


# 安装 PyTorch
pip install /torch_wheels/torch*.whl
pip install --pre torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# 安装 common 依赖
pip install -r requirements/common.txt
pip install -r requirements/build.txt

# --- build xformers ---
pip install /torch_wheels/xformers*.whl

# --- build vllm ---
python3 setup.py bdist_wheel --dist-dir /wheels/vllm-dist

# --- build flashinfer ---
pip install /wheels/flashinfer-dist
