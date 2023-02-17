import torch
import torch._dynamo
import torch._inductor
from torch._inductor.compile_fx import aot_compile_fx

torch._inductor.config.aot_codegen_output_prefix = "aot_inductor_output"


def func(x):
    return (torch.sigmoid(torch.sin(x)), torch.sigmoid(torch.cos(x)))


inp = torch.randn((8, 4, 16, 16), device="cpu")
fx_graph, _ = torch._dynamo.export(func, inp)
aot_compile_fx(fx_graph, inp)
