# import torch
# import torch.nn as nn
# import torch.utils.checkpoint as checkpoint


# class CheckpointedModel(nn.Module):
#     def __init__(self):
#         super(CheckpointedModel, self).__init__()
#         self.layer1 = nn.Linear(1024, 1024)
#         self.layer2 = nn.Linear(1024, 1024)
#         self.layer3 = nn.Linear(1024, 1024)
#         self.layer4 = nn.Linear(1024, 1024)

#     def forward(self, x):
#         # No checkpoint on layer1
#         def fn(x):
#             q = self.layer1(x)
#             # print(q.shape)
#             k = self.layer2(x)
#             v = self.layer3(x)
#             return q * k * v
#         return checkpoint.checkpoint(fn, x, use_reentrant=False)


# # Testing the model
# model = CheckpointedModel()
# x = torch.randn(1, 1024)

# opt_model = torch.compile(model, backend="eager")

# with torch.autocast("cpu", dtype=torch.bfloat16):
#     out = opt_model(x)


# import contextlib
# import torch


# @contextlib.contextmanager
# def whoo(x):
#     try:
#         yield x.sin()
#     except:
#         pass

# def g(x):
#     return x.cos() + x.tan()

# torch._dynamo.config.enable_trace_contextlib = True

# @torch.compile(backend='eager', fullgraph=False)
# def fn(x):
#     with whoo(x) as y:
#         return g(x)

# x = torch.randn(2)
# fn(x)

# import torch

# class DummyModule(torch.nn.Module):
#     def __init__(self):
#         super(DummyModule, self).__init__()
#         self.a = torch.nn.ModuleDict(
#             {
#                 "b": torch.nn.ModuleDict(
#                     {
#                         "c": torch.nn.ModuleDict(
#                             {
#                                 "d": torch.nn.ModuleDict(
#                                     {
#                                         "e": torch.nn.Linear(10, 10, bias=False)
#                                     }
#                                 )
#                             }
#                         )
#                     }
#                 )
#             }
#         )
#     def forward(self, x):
#         return self.a.b.c.d.e(x)

# model = DummyModule()

# @torch.compile(fullgraph=True, backend='eager')
# def f(model, states, x):
#     return torch.func.functional_call(model, states, x)

# states = model.state_dict()
# x = torch.randn(10, 10)
# f(model, states, x)


# import torch
# from torch._dynamo.testing import EagerAndRecordGraphs

# def h(x):
#     return x.cos()

# @torch._dynamo.disable
# def g(x):
#     return x.sin() + h(x)

# eager = EagerAndRecordGraphs()
# @torch.compile(backend=eager)
# def f(x):
#     return g(x)

# x = torch.randn(2)
# y = f(x)
# assert torch.all(y == (x.sin() + x.cos()))
# assert len(eager.graphs) == 0

# ###############################################

# def k(x):
#     return x.tan() + h(x)

# @torch.compile(backend=eager)
# def l(x):
#     @torch._dynamo.disable
#     def u(x):
#         return k(x)
#     return u(x)

# x = torch.randn(2)
# y = l(x)
# assert torch.all(y == (x.tan() + x.cos()))
# assert len(eager.graphs) == 0


# import torch
# import itertools

# def subgen(t):
#     yield t+1
#     yield t+2

# def main_gen(t):
#     yield from subgen(t)
#     yield t+3

# @torch.compile(backend="eager", fullgraph=True)
# def fn(t):
#     gen = main_gen(t)
#     return list(gen)

# t = torch.tensor([1.0])
# y = fn(t)
# print(y)


import torch
import itertools

def subgen(t):
    yield t+1
    yield t+2

def main_gen(t):
    yield from subgen(t)
    yield t+3

@torch.compile(backend="eager", fullgraph=True)
def fn(t):
    gen = main_gen(t)
    return max(gen)

t = torch.tensor([1.0])
y = fn(t)
print(y)
