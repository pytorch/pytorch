import torch
import torch._dynamo
import torch._inductor.config


def func(x):
    return (torch.sigmoid(torch.sin(x)), torch.sigmoid(torch.cos(x)))


inp = torch.randn((8, 4, 16, 16), device="cpu")

torch._dynamo.export(func, inp, aot_inductor=True)
