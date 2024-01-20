# Owner(s): ["oncall: distributed"]

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

from torch.distributed.pipeline.sync import Pipe
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.common_cuda import TEST_MULTIGPU


def test_python_autograd_function(setup_rpc):
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
    model = Pipe(model, checkpoint="always")

    x = torch.rand(42)
    y = model(x)
    assert torch.allclose(x, y.local_value())


def test_exception_no_hang(setup_rpc):
    # In v0.0.2, once a failed partition receives a normal message
    # (non-closing) for the next micro-batch, a hang occurred. The reason was
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
    model = Pipe(model, chunks=3)

    with pytest.raises(ExpectedException):
        model(torch.rand(3))


@pytest.mark.skipif(not TEST_MULTIGPU, reason="2 cuda devices required")
def test_tuple_wait(cuda_sleep, setup_rpc):
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
        def __init__(self):
            super().__init__()
            self.ones = nn.Parameter(torch.ones(32, 3, 32, 32, requires_grad=True))

        def forward(self, a, b):
            a = a * self.ones
            return a * 1, b * 2, b * 3

    class Layer2(nn.Module):
        def __init__(self):
            super().__init__()
            self.ones = nn.Parameter(torch.ones(32, 3, 32, 32, requires_grad=True))

        def forward(self, a, b, c):
            a = a * self.ones
            b = Sleep.apply(b)
            return a + b + c

    model = nn.Sequential(Layer1().cuda(0), Layer2().cuda(1))
    model = Pipe(model, chunks=32, checkpoint="never")

    a = torch.rand(1024, 3, 32, 32, device=0, requires_grad=True)
    b = torch.rand(1024, 3, 32, 32, device=0, requires_grad=True)

    y = model(a, b)
    y.local_value().norm().backward()

    torch.cuda.synchronize(0)
    torch.cuda.synchronize(1)

    assert torch.isclose(b.grad.norm().cpu(), torch.tensor(5.000))


def test_parallel_randoms(setup_rpc):
    class Dropouts(nn.Module):
        def forward(self, x):
            for _ in range(100):
                x = F.dropout(x, p=0.001)
            return x

    model = nn.Sequential(Dropouts(), Dropouts())

    x = torch.rand(10, 10, requires_grad=True)
    model = Pipe(model, chunks=10, checkpoint="always")
    y = model(x)
    y = y.local_value()
    y.norm().backward()

    assert y.to(torch.bool).tolist() == x.grad.to(torch.bool).tolist()


if __name__ == "__main__":
    run_tests()
