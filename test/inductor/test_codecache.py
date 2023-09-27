# Owner(s): ["module: inductor"]
import functools
import unittest

import torch
from torch._inductor.codecache import AsyncCompile
from torch.testing._internal.inductor_utils import HAS_CUDA

requires_cuda = functools.partial(unittest.skipIf, not HAS_CUDA, "requires cuda")


class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 10)

    def forward(self, inp):
        return self.fc1(inp)


def _run_codecache_test(start_method):
    torch._inductor.config.worker_start_method = start_method
    torch._inductor.config.compile_threads = 16
    AsyncCompile.warm_pool()

    model = MyModel().cuda()
    model = torch.compile(model)
    inp = torch.rand(10, 10).cuda()
    model(inp).sum().backward()


@requires_cuda()
def test_codecache_spawn():
    _run_codecache_test("spawn")


@requires_cuda()
def test_codecache_fork():
    _run_codecache_test("fork")
