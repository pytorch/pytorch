from functools import wraps
import torch

def catch_exceptions(function):
    @wraps(function)
    def safe_call(*args, **kwargs):
        args = (function,) + args
        torch._C._safe_call(*args, **kwargs)
    return safe_call

