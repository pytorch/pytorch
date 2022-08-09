from unicodedata import decomposition
import torch
from torch.profiler import profile, ProfilerActivity
from torch.fx.experimental.proxy_tensor import make_fx
from functorch.compile import draw_graph, ts_compile



import os
import sys
from os.path import abspath
from os.path import exists


import importlib
from torchvision.models import resnet18
import torch.utils._pytree as pytree
from torch.nn.utils.stateless import functional_call
from functorch.compile import default_decompositions
import torch.utils._pytree as pytree
from torchdynamo.testing import reduce_to_scalar_loss

from functorch._src.aot_autograd import aot_function
from functorch.compile import min_cut_rematerialization_partition, print_compile


from typing import Optional

import torch.nn as nn
from functorch.compile import memory_efficient_fusion

def _fn(
    x: torch.Tensor,
    bias: Optional[torch.nn.parameter.Parameter],
    activation: nn.Module,
    prob: float,
) -> torch.Tensor:
    if bias is not None:
        x = torch.add(x, bias)
    y = activation(x)
    return torch.nn.functional.dropout(y, prob) if prob > 0.0 else y


class NVFusedBiasActivationDropout(torch.nn.Module):
    """
    A layer which fuses the computation of Dropout(Activation(x + Bias))
    with AOTAutograd and nvFuser
    """

    def __init__(
        self,
        p: float,
        activation = None,
        bias_shape: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.p = float(p)
        self.requires_residual = False
        self.activation = activation

        self.bias = (
            nn.Parameter(torch.zeros(bias_shape)) if bias_shape is not None else None
        )

        assert (
            self.p < 1.0
        ), f"We don't want to drop all the values, most probably p={self.p} is not properly set"

    def init_weights(self, *args, **kwargs):
        with torch.no_grad():
            if self.bias is not None:
                self.bias.fill_(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Train/inference
        p = self.p if self.training else 0.0

        # Catch a non-cuda setup, fallback to pytorch
        if not x.is_cuda:
            raise NotImplementedError("Only implemented for CUDA")

        # AOTAutograd, NVFuser backed path
        aot_fn = memory_efficient_fusion(_fn, static_argnums=(2, 3))
        return aot_fn(x, self.bias, self.pytorch_activation, p)

g_gpu = torch.Generator(device='cuda')
g_gpu.manual_seed(214748364)
inp = torch.randn(2**20, device='cuda', generator=g_gpu, requires_grad=True)

aot_f = aot_function(_fn, fw_compiler=print_compile, partition_fn=min_cut_rematerialization_partition, 
                     decompositions=default_decompositions, static_argnums=(2, 3))

activation = nn.GELU()
bias = torch.nn.Parameter(torch.zeros(2**20, device="cuda"))
aot_f(inp, bias, activation, 0.8)

exit(1)
def f(y):
    x = y.cos().cos().cos()
    b = x.sum()
    return b
    # b.backward()
    # return y.grad


g_gpu = torch.Generator(device='cuda')
g_gpu.manual_seed(214748364)
inp = torch.randn(2**20, device='cuda', generator=g_gpu, requires_grad=True)

aot_f = aot_function(f, fw_compiler=print_compile, partition_fn=min_cut_rematerialization_partition, 
                     decompositions=default_decompositions)
aot_f(inp)
# traced_graph = make_fx(f, decomposition_table=default_decompositions)(inp)
# traced_graph.graph.eliminate_dead_code()
# draw_graph(traced_graph, "traced")
# print(traced_graph.graph)

exit(1)
def trace_model(model, inputs):
    """
    Get the full graph (both forward and backward) of `model` on `inputs`
    The moddel should have a single forward and a single backward graph
    """
    def f(params, inp):
        out = functional_call(model, params, inp)
        loss = reduce_to_scalar_loss(out)
        loss.backward()
        return [param.grad for param in params.values()]
    
    params = dict(model.named_parameters())
    traced_graph = make_fx(f, decomposition_table=default_decompositions)(params, inputs)
    return traced_graph, params

inputs = torch.rand(1, 3, 224, 224, device="cuda")
model = resnet18(pretrained=True)
model.to("cuda")


model.eval()
traced_graph, params = trace_model(model, inputs)
traced_graph.graph.set_codegen(torch.fx.graph.CodeGen())  # avoid recursive pytree
arg_list, spec  = pytree.tree_flatten([params, inputs])
# print(traced_graph)
script_f = ts_compile(traced_graph, 0)
with torch.no_grad():
    script_f(*arg_list)
    script_f(*arg_list)
        
exit(1)
def get_cuda_time(timing):
    """
    Get the total cuda time from torch profiler timings
    """
    cuda_time_total = 0
    for e in timing:
        cuda_time_total = cuda_time_total + e.cuda_time_total
    return cuda_time_total


def bench_GPU_time(f, inp):
    
    itr = 10
    with torch.no_grad():
        for _ in range(5):
            f(inp)
            torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(itr):
                torch.cuda.reset_peak_memory_stats()
                f(inp)
                torch.cuda.synchronize()
                print(torch.cuda.max_memory_allocated())

    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))
    print(get_cuda_time(prof.key_averages()) / itr) 


def bench_scripted_time(f, inp):
    traced = make_fx(f, decomposition_table=default_decompositions)(inp)
    scripted_f = ts_compile(traced, inp)
    bench_GPU_time(scripted_f, inp)

inp = torch.randn(2**22, device="cuda", requires_grad=False)


def f(a):
    b = a.cos().cos()
    return b

def f2(a):
    b = a.cos().cos().cos()
    return b



# 
# bench_GPU_time(f, inp)
bench_scripted_time(f, inp)
# print(torch.cuda.max_memory_allocated())
# torch.cuda.reset_peak_memory_stats()

# bench_GPU_time(f2, inp)
bench_scripted_time(f2, inp)
# print(torch.cuda.max_memory_allocated())
