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
import torch
from torch import nn

from torch.distributed._pipeline.sync import Pipe


def test_simple_linears():
    def sum_grad(parameters):
        return sum([p.grad.sum() for p in parameters if p.grad is not None])

    def zero_grad(parameters):
        for p in parameters:
            p.grad = None

    inputs = torch.rand(8, 1)
    model = nn.Sequential(nn.Linear(1, 2), nn.Linear(2, 4), nn.Linear(4, 2), nn.Linear(2, 1),)

    # Without Pipe
    outputs = model(inputs)
    loss = outputs.mean()
    loss.backward()

    grad_without_pipe = sum_grad(model.parameters())

    zero_grad(model.parameters())

    # With Pipe
    model = Pipe(model, [2, 2], devices=["cpu", "cpu"], chunks=4)

    outputs = model(inputs)
    loss = outputs.mean()
    loss.backward()

    grad_with_pipe = sum_grad(model.parameters())

    # Both grads should be identical.
    assert torch.allclose(grad_with_pipe, grad_without_pipe)
