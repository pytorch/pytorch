import torch
from torch._ops import HigherOrderOperator


class PythonFallback(HigherOrderOperator):
    def __init__(self):
        super().__init__("python_fallback")

    def __call__(self, func, args, kwargs):
        return super().__call__(func, args, kwargs)


python_fallback_op = PythonFallback()

def python_fallback():

    def wrapped(fn):
        def inner(*args, **kwargs):
            return torch.ops.higher_order.python_fallback(fn, args, kwargs)
        return inner

    return wrapped


@python_fallback_op.py_autograd_impl
def python_fallback_autograd_impl(func, args, kwargs):
    # cannot use autograd.Function because we need the autograd engine
    # to see what happened. Then we need a way to store the intermediates.
    return func(*args, **kwargs)
