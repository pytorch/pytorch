# import torch
# from torch._dynamo.testing import EagerAndRecordGraphs, normalize_gm

# flag = True
# torch._dynamo.config.enable_einops_tracing=flag

# import einops

# backend = EagerAndRecordGraphs()

# def f(x):
#     y = einops.reduce(x, 'a b c -> a b', "min")
#     return y.sin()

# x = torch.randn(3, 4, 5)
# expected = f(x)
# got = torch.compile(f, backend=backend, fullgraph=True)(x)

# assert torch.allclose(expected, got)
# assert len(backend.graphs) == 1, len(backend.graphs)
# graph = backend.graphs[0]
# print(normalize_gm(graph.print_readable(print_output=False)))


import torch

fromhex = float.fromhex

@torch.compile(backend="eager", fullgraph=True)
def fn(x):
    a = fromhex('0.')
    return x.sin()

fn(torch.randn(2))