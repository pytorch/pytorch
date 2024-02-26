"""
Produced AOT graphs: https://gist.github.com/yf225/3a9ef062a750dd02d7e08b87aa4bc864

Notice that there is no copy_ to copy the mutated values back to the input. Why?
"""

"""
git pull && TORCH_COMPILE_DEBUG=1 python3 test_foreach_copy.py >output.txt 2>&1
"""
import contextlib
import logging
import os
import sys
import traceback

import torch
import torch._dynamo
import torch.distributed as dist
import torch.nn as nn
from torch._dynamo import compiled_autograd
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.distributed._tensor.placement_types import Shard
from torch.distributed.device_mesh import DeviceMesh
# from torchviz import make_dot

hidden_dim = 12340

device_type = "cuda"


def func(x1, x2, x3, x4):
    x5 = x1[:]
    x6 = x2[:]
    with torch.no_grad():
        torch._foreach_copy_([x5, x6], [x3, x4])
    return torch.matmul(x1, x2)

if __name__ == "__main__":
    x1 = torch.randn(4, 4, requires_grad=True)
    x2 = torch.randn(4, 4, requires_grad=True)
    x3 = torch.randn(4, 4, requires_grad=True)
    x4 = torch.randn(4, 4, requires_grad=True)
    def compiler_fn(gm):
        print("Compiling autograd?")
        return torch.compile(gm, backend="inductor", fullgraph=True)
    with compiled_autograd.enable(compiler_fn):
        out1 = torch.compile(func, backend="inductor", fullgraph=True)(x1, x2, x3, x4)
        out1.sum().backward()
