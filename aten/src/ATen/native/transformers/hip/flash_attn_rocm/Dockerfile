# BSD 3 Clause
# Copyright 2023 Advanced Micro Devices, Inc.
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FROM rocm/pytorch:rocm5.3.2_ubuntu20.04_py3.7_pytorch_1.12.1
WORKDIR /flash_attn

USER root

ENV DEBIAN_FRONTEND noninteractive
ENV TZ "Asia/Shanghai"

RUN apt-get update \
    && apt install -y git-all \
    && git clone https://<your github username>:<your github token>@github.com/ROCmSoftwarePlatform/flash-attention_private \
    && cd /flash_attn/flash-attention_private \
    && git checkout flash_attention_for_rocm \
    && cd /flash_attn/flash-attention_private/csrc/flash_attn_rocm/composable_kernel \
    && git submodule init \
    && git submodule update \
    && cd /flash_attn/flash-attention_private/csrc/flash_attn_rocm \
    && mkdir build \
    && cd build \
    && cmake .. \
    && cd /flash_attn/flash-attention_private/csrc/flash_attn_rocm/build \
    && make -j64

