#!/bin/bash
set -ex

# 安装 PyTorch
if ls /torch_wheels/torch*.whl 2>/dev/null; then
  pip install /torch_wheels/torch*.whl
else
  pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
fi

# 安装 common 依赖
pip install -r requirements/common.txt
pip install -r requirements/build.txt

# --- build xformers ---
git clone https://github.com/facebookresearch/xformers.git --recursive
cd xformers
git checkout f2de641ef670510cadab099ce6954031f52f191c
python3 setup.py bdist_wheel --dist-dir /wheels/xformers-dist
cd ..
rm -rf xformers

# --- build vllm ---
python3 setup.py bdist_wheel --dist-dir /wheels/vllm-dist

# --- build flashinfer ---
git clone https://github.com/flashinfer-ai/flashinfer.git --recursive
cd flashinfer
git checkout v0.2.2.post1
FLASHINFER_ENABLE_AOT=1 python3 setup.py bdist_wheel --dist-dir /wheels/flashinfer-dist
