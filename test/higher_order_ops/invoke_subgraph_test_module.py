import torch


tensor = None
_region = torch.compiler.nested_compile_region(lambda x, t: x + tensor + t)


def call(x, t):
    return _region(x, t)
