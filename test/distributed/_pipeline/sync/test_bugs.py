# Copyright 2019 Kakao Brain
#
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
import pytest
import torch
from torch import nn
import torch.nn.functional as F

from torch.distributed._pipeline.sync import Pipe


def test_python_autograd_function():
    # A Python autograd function might fail with this error:
    #
    #   RuntimeError: Returning Variables sharing storage with other Variables
    #   that require grad is not supported in Python functions. Please submit a
    #   feature request if you hit this error.
    #
    # It doesn't look like an essential restriction. But it happens on the
    # current PyTorch version. To avoid it, we should detach the tensor before
    # returning by identity autograd functions, such as Wait, Fork, and Join.
    #
    class Identity(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            return input

        @staticmethod
        def backward(ctx, grad):
            return grad

    class M(nn.Module):
        def forward(self, input):
            return Identity.apply(input)

    model = nn.Sequential(M(), M())
    model = Pipe(model, [1, 1], devices=["cpu", "cpu"], checkpoint="always")

    x = torch.rand(42)
    y = model(x)
    assert torch.allclose(x, y)


def test_exception_no_hang():
    # In v0.0.2, once a failed partition receives a normal message
    # (non-closing) for the next micro-batch, a hang occured. The reason was
    # that a failed partition didn't call in_queue.task_done() on a normal
    # message. So the former partition was blocked at out_queue.join() for the
    # next of next micro-batch.
    class ExpectedException(Exception):
        pass

    class Pass(nn.Module):
        def forward(self, x):
            return x

    class Raise(nn.Module):
        def forward(self, x):
            raise ExpectedException()

    model = nn.Sequential(Pass(), Pass(), Raise())
    model = Pipe(model, [1, 1, 1], devices=["cpu", "cpu", "cpu"], chunks=3)

    with pytest.raises(ExpectedException):
        model(torch.rand(3))


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="2 cuda devices required")
def test_tuple_wait(cuda_sleep):
    # In v0.0.3, Wait is applied to only the first tensor on a micro-batch.
    # Under this behavior, if checkpointing was disabled, there's a possibility
    # that gradient accumulations on other tensors are not synchronized
    # properly to the copy stream.
    class Sleep(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return x.detach()

        @staticmethod
        def backward(ctx, grad):
            with torch.cuda.device(grad.device):
                cuda_sleep(0.05)
            return grad

    class Layer1(nn.Module):
        def forward(self, pair):
            a, b = pair
            return a * 1, b * 2, b * 3

    class Layer2(nn.Module):
        def forward(self, triple):
            a, b, c = triple
            b = Sleep.apply(b)
            return a + b + c

    model = nn.Sequential(Layer1(), Layer2())
    model = Pipe(model, [1, 1], devices=[0, 1], chunks=32, checkpoint="never")

    a = torch.rand(1024, 3, 32, 32, device=0, requires_grad=True)
    b = torch.rand(1024, 3, 32, 32, device=0, requires_grad=True)

    y = model((a, b))
    y.norm().backward()

    torch.cuda.synchronize(0)
    torch.cuda.synchronize(1)

    assert torch.isclose(b.grad.norm().cpu(), torch.tensor(5.000))


def test_parallel_randoms():
    class Dropouts(nn.Module):
        def forward(self, x):
            for _ in range(100):
                x = F.dropout(x, p=0.001)
            return x

    model = nn.Sequential(Dropouts(), Dropouts())

    x = torch.rand(10, 10, requires_grad=True)
    model = Pipe(model, [1, 1], devices=["cpu", "cpu"], chunks=10, checkpoint="always")
    y = model(x)
    y.norm().backward()

    assert y.to(torch.bool).tolist() == x.grad.to(torch.bool).tolist()
