# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# Copyright 2019 Kakao Brain
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from this
#    software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
"""Per-layer profilers."""
import copy
import time
from typing import Generator, List, Tuple, Union

import torch
from torch import Tensor
import torch.nn as nn

from ..microbatch import Batch

__all__: List[str] = []


Device = Union[torch.device, int, str]

Tensors = Tuple[Tensor, ...]
TensorOrTensors = Union[Tensor, Tensors]


def layerwise_sandbox(module: nn.Sequential, device: torch.device,) -> Generator[nn.Module, None, None]:
    """Copies layers for ease to profile. It doesn't modify the given
    module.
    """
    for layer in module:
        layer_copy = copy.deepcopy(layer)
        layer_copy.to(device)
        layer_copy.train()
        yield layer_copy


def detach(batch: Batch) -> None:
    """Detaches from autograd graph."""
    for i, x in enumerate(batch):
        batch[i] = x.detach().requires_grad_(x.requires_grad)


def profile_times(module: nn.Sequential, sample: TensorOrTensors, timeout: float, device: torch.device,) -> List[int]:
    """Profiles elapsed times per layer."""
    if any(p.grad is not None for p in module.parameters()):
        raise ValueError("some parameter already has gradient")

    _batch = Batch(sample)
    for i, x in enumerate(_batch):
        _batch[i] = x.detach().to(device).requires_grad_(x.requires_grad)

    time_bufs: List[List[float]] = [[] for _ in module]
    begun_at = time.time()

    while time.time() - begun_at < timeout:
        batch = _batch

        for i, layer in enumerate(layerwise_sandbox(module, device)):
            detach(batch)

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            tick = time.time()

            # Forward
            batch = batch.call(layer)

            # Backward
            backward_tensors = tuple(y for y in batch if y.requires_grad)
            if backward_tensors:
                torch.autograd.backward(backward_tensors, backward_tensors)

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            tock = time.time()

            time_bufs[i].append(tock - tick)

    us = 1_000_000
    return [sum(int(t * us) for t in buf) for buf in time_bufs]


def profile_sizes(
    module: nn.Sequential, input: TensorOrTensors, chunks: int, param_scale: float, device: torch.device,
) -> List[int]:
    """Profiles CUDA memory usage per layer."""
    if device.type != "cuda":
        raise ValueError("size profiler supports only CUDA device")

    batch = Batch(input)
    sizes: List[int] = []

    latent_scale = batch[0].size(0) / chunks
    for i, x in enumerate(batch):
        batch[i] = x[:1].detach().to(device).requires_grad_(x.requires_grad)

    for layer in layerwise_sandbox(module, device):
        detach(batch)

        # Detect memory usage at forward.
        memory_before = torch.cuda.memory_allocated(device)
        batch = batch.call(layer)
        memory_after = torch.cuda.memory_allocated(device)
        latent_size = memory_after - memory_before

        # Analyze size of parameters.
        param_size = sum(p.storage().size() * p.storage().element_size() for p in layer.parameters())

        # Combine size of parameters and activations with normalize scales.
        size = latent_size * latent_scale + param_size * param_scale
        sizes.append(int(size))

    return sizes
