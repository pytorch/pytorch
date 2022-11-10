#!/bin/bash python3
import torch
import functorch
import torch._dispatch.python as dispatch
from torch._dynamo.optimizations.training import aot_nvprims_nvfuser, aot_nvprims_aten
from torch._prims.context import TorchRefsNvfuserCapabilityMode
optimize = torch._dynamo.optimize(aot_nvprims_aten)

class Fusion(torch.nn.Module):
  def __init__(self) :
    super(Fusion, self).__init__()
    self.conv = torch.nn.Conv2d(16, 16, (1, 1), bias=False)
    with dispatch.enable_python_dispatcher():
        self.norm = torch.nn.BatchNorm2d(16, track_running_stats=True)

  def forward(self, inp) :
    out = self.conv(inp)
    out = out.relu()
    out = self.norm(out)
    return out

model = Fusion()

input1 = torch.randn(2, 16, 8, 8)

optimized_model = optimize(model)
for _ in range(5):
    out = optimized_model(input1)
    out.sum().backward()