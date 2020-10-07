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

from torch.distributed._pipeline.sync.phony import get_phony


def test_phony_size():
    p = get_phony(torch.device("cpu"), requires_grad=False)
    assert p.size() == (0,)


def test_phony_requires_grad():
    p1 = get_phony(torch.device("cpu"), requires_grad=True)
    p2 = get_phony(torch.device("cpu"), requires_grad=False)
    assert p1.requires_grad
    assert not p2.requires_grad


def test_cached_phony():
    p1 = get_phony(torch.device("cpu"), requires_grad=True)
    p2 = get_phony(torch.device("cpu"), requires_grad=True)
    assert p1 is p2

    p3 = get_phony(torch.device("cpu"), requires_grad=False)
    p4 = get_phony(torch.device("cpu"), requires_grad=False)
    assert p3 is p4

    assert p1 is not p3


def test_phony_in_autograd_function():
    class Phonify(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            phony = get_phony(input.device, requires_grad=False)
            return phony.detach()

    x = torch.rand(1, requires_grad=True)

    p1 = Phonify.apply(x)
    p2 = get_phony(torch.device("cpu"), requires_grad=True)

    assert p1 is not p2
    assert p1.grad_fn is not None
    assert p2.grad_fn is None
