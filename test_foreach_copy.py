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
    torch._foreach_copy_([x1, x2], [x3, x4])
    return x1 + x2

if __name__ == "__main__":
    x1 = torch.randn(3, 4)
    x2 = torch.randn(3, 4)
    x3 = torch.randn(3, 4)
    x4 = torch.randn(3, 4)
    out1 = torch.compile(func)(x1, x2, x3, x4)
    out1.sum().backward()