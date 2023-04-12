# This module contains functions that *will be allowed* by dynamo

import functools

import torch.nn


def is_compiling():
    return False


def wrap_inline(fn):
    """
    Create an extra frame around fn that is not in skipfiles
    """
    if isinstance(fn, torch.nn.Module):
        return WrapperModule(fn)

    @functools.wraps(fn)
    def inner(*args, **kwargs):
        return fn(*args, **kwargs)

    return inner


class WrapperModule(torch.nn.Module):
    def __init__(self, mod):
        super().__init__()
        self.mod = mod

    def __call__(self, *args, **kwargs):
        return self.mod(*args, **kwargs)
