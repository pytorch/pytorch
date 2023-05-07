# Owner(s): ["module: dynamo"]

import torch
import torch._dynamo


def inner_func():
    return torch.is_grad_enabled()


def outer_func(func):
    def wrapped(*args):
        a = func(*args)
        torch._dynamo.graph_break()
        return torch.sin(a + 1), inner_func()

    return wrapped
