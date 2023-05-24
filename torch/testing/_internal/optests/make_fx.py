import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_methods_invocations import wrapper_set_seed
from .common import TestFrameworkError


def make_fx_check(func, args, kwargs, tracing_mode, assertion_fn=torch.testing._comparison.assert_close):
    def f(args, kwargs, extra_args, extra_kwargs):
        if extra_args:
            for i, t in extra_args:
                args[i] = t.size()
        if extra_kwargs:
            for k, t in extra_kwargs.items():
                kwargs[k] = t.size()

        return func(*args, **kwargs)

    # If any argument is a torch.Size(), maybe get dynamic shapes for it by:
    # - Create a temporary Tensor whose size is the torch.Size() we want. Note that
    #   we use an expanded Tensor as we cannot pass "meta" Tensors to make_fx.
    # - Pass it to make_fx such that it is is converted to a proxy Tensor
    # - Unpack the size in the wrapper to get a torch.Size with dynamic shapes (in
    #   symbolic mode, a no-op otherwise)
    extra_args = []
    extra_kwargs = {}
    for i, arg in enumerate(args):
        if isinstance(arg, torch.Size):
            extra_args.append((i, torch.empty(arg, device="cpu")))
    for key, value in kwargs.items():
        if isinstance(value, torch.Size):
            extra_kwargs[key] = torch.empty(value, device="cpu")

    new_f = make_fx(f, tracing_mode=tracing_mode)(args, kwargs, extra_args, extra_kwargs)
    for arg in args:
        if isinstance(arg, torch.Tensor) and arg.dtype == torch.float:
            with torch.no_grad():
                arg.uniform_(0, 1)
    try:
        old_out = f(args, kwargs, extra_args, extra_kwargs)
    except Exception:
        raise TestFrameworkError("Attempted to generate new args and invoke func ",
                                 "but that led to an exception")

    new_out = wrapper_set_seed(new_f, args, kwargs, extra_args, extra_kwargs)
    assertion_fn(new_out, old_out)
