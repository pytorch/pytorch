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
import weakref

import pytest
import torch

from torch.distributed._pipeline.sync.dependency import Fork, Join, fork, join


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")
def test_fork_join():
    logs = []

    class Log(torch.autograd.Function):
        @staticmethod
        def forward(ctx, number, tensor):
            ctx.number = number
            return tensor.detach()

        @staticmethod
        def backward(ctx, grad):
            logs.append(ctx.number)
            return None, grad

    a = torch.rand(1, device="cpu", requires_grad=True)
    b = torch.rand(1, device="cuda", requires_grad=True)

    a = Log.apply(1, a)

    a, phony = fork(a)
    b = join(a, phony)

    b = Log.apply(2, b)
    b = b.to("cpu")

    (a + b).backward()

    assert logs == [2, 1]


def test_fork_join_enable_grad():
    x = torch.rand(1, requires_grad=True)

    with torch.enable_grad():
        x2, p = fork(x)

    assert p.requires_grad
    assert x2 is not x
    x = x2

    assert x.requires_grad
    assert p.requires_grad
    assert x.grad_fn.__class__ is Fork._backward_cls
    assert p.grad_fn.__class__ is Fork._backward_cls

    with torch.enable_grad():
        x2 = join(x, p)

    assert x2 is not x
    x = x2

    assert x.requires_grad
    assert x.grad_fn.__class__ is Join._backward_cls


def test_fork_join_no_grad(monkeypatch):
    def do_not_apply(*args):
        raise AssertionError("Function.apply called")

    monkeypatch.setattr("torch.autograd.Function.apply", do_not_apply)

    x = torch.rand(1, requires_grad=True)

    with torch.no_grad():
        x2, p = fork(x)

    assert not p.requires_grad
    assert x2 is x
    x = x2

    with torch.no_grad():
        x2 = join(x, p)

    assert x2 is x
    x = x2


def test_fork_leak():
    leak = None

    class F(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            return input

        @staticmethod
        def backward(ctx, grad):
            nonlocal leak
            leak = weakref.ref(ctx)
            return grad

    x = torch.rand(1, requires_grad=True)
    x = F.apply(x)
    x, phony = fork(x)
    x = join(x, phony)

    x.backward()
    del x, phony

    assert leak() is None


def test_join_when_fork_not_requires_grad():
    x = torch.rand(2, 1)
    a, b = x.chunk(2)

    assert not a.requires_grad
    a, p = fork(a)
    assert not a.requires_grad
    assert not p.requires_grad

    assert not b.requires_grad
    b = join(b, p)
    assert not b.requires_grad


def test_join_when_fork_requires_grad():
    x = torch.rand(2, 1)
    a, b = x.chunk(2)

    a.requires_grad_()
    assert a.requires_grad
    a, p = fork(a)
    assert a.requires_grad
    assert p.requires_grad

    assert not b.requires_grad
    b = join(b, p)
    assert b.requires_grad
