# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# Copyright 2019 Kakao Brain
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
