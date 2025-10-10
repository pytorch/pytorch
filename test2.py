import torch
from torch import nn
from torch.fx.graph_module import _forward_from_src
import torch.profiler
from collections import defaultdict
import json

def f(self, x):

    # No stacktrace found for following nodes
    _rf_net1_weight = torch.profiler.record_function('net1_weight'); _rf_net1_weight.__enter__()
    net1_weight = self.net1.weight
    _rf_net1_weight.__exit__(None, None, None)
    _rf_net1_bias = torch.profiler.record_function('net1_bias'); _rf_net1_bias.__enter__()
    net1_bias = self.net1.bias
    _rf_net1_bias.__exit__(None, None, None)
    _rf_net2_weight = torch.profiler.record_function('net2_weight'); _rf_net2_weight.__enter__()
    net2_weight = self.net2.weight
    _rf_net2_weight.__exit__(None, None, None)
    _rf_net2_bias = torch.profiler.record_function('net2_bias'); _rf_net2_bias.__enter__()
    net2_bias = self.net2.bias
    _rf_net2_bias.__exit__(None, None, None)

     # File: /data/users/shangdiy/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
    _rf_linear = torch.profiler.record_function('linear'); _rf_linear.__enter__()
    linear = torch.ops.aten.linear.default(x, net1_weight, net1_bias);  x = net1_weight = net1_bias = None
    _rf_linear.__exit__(None, None, None)

     # File: /data/users/shangdiy/pytorch/torch/nn/modules/activation.py:144 in forward, code: return F.relu(input, inplace=self.inplace)
    _rf_relu = torch.profiler.record_function('relu'); _rf_relu.__enter__()
    relu = torch.ops.aten.relu.default(linear);  linear = None
    _rf_relu.__exit__(None, None, None)

     # File: /data/users/shangdiy/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
    _rf_linear_1 = torch.profiler.record_function('linear_1'); _rf_linear_1.__enter__()
    linear_1 = torch.ops.aten.linear.default(relu, net2_weight, net2_bias);  relu = net2_weight = net2_bias = None
    _rf_linear_1.__exit__(None, None, None)
    return linear_1


class MLPModule(nn.Module):
    def __init__(self, device, bias=True):
        super().__init__()
        torch.manual_seed(5)
        self.net1 = nn.Linear(10, 16, bias=bias, device=device)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(16, 10, bias=bias, device=device)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    # schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
) as prof:
    f(MLPModule("cuda"), x=torch.rand(10).to("cuda"))
prof.export_chrome_trace("trace.json")
