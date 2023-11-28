# This module contains functions that *will be allowed* by dynamo

import functools


def is_compiling() -> bool:
    return False


def wrap_inline(fn):
    """
    Create an extra frame around fn that is not in skipfiles
    """

    @functools.wraps(fn)
    def inner(*args, **kwargs):
        return fn(*args, **kwargs)

    return inner


def call_hook(hook, *args):
    """
    Used by compiled autograd to handle hook returning None
    """
    result = hook(*args)
    if result is None:
        return args[0]
    return result

def call_backward(backward_obj, ctx, *inputs):
    """
    Here just to make the graph call_function logs look identical to hook's implementation
    """
    return backward_obj.apply(ctx, inputs)
