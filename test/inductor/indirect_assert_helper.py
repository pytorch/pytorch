import sys

import torch
from torch.testing._internal.inductor_utils import GPU_TYPE


def first_arg(x, y):
    return x[y]


def second_arg(x, y):
    return x[:, y]


def same_pm_one(x, y):
    return x[y + 1, y - 1]


def same_pp_one(x, y):
    return x[y + 1, y + 1]


def store(x, y, z):
    x[y + 1, y + 1] = z


def upper1(x):
    b = torch.arange(4, device=x.device)
    return x[b]


def lower1(x):
    b = x.new_full((), -4, dtype=torch.int64)
    return x[b]


def upper2(x):
    b = x.new_full((), 4, dtype=torch.int64)
    return x[b]


def lower2(x):
    b = x.new_zeros((), dtype=torch.int64)
    return x[b - 4]


if __name__ == "__main__":
    fns = [
        name
        for name, obj in locals().items()
        if callable(obj) and obj.__module__ == __name__
    ]

    _, fn_name, dims, dyn_shape, one_size = sys.argv
    if fn_name not in fns:
        raise AssertionError(f"Unknown function: {fn_name}")
    if one_size not in ("True", "False"):
        raise AssertionError(f"one_size must be 'True' or 'False', got {one_size}")
    one_size = one_size == "True"
    if dims not in ("2", "3"):
        raise AssertionError(f"dims must be '2' or '3', got {dims}")
    shape_x = [3, 2, 4] if dims == "3" else [3, 2]
    if one_size:
        if fn_name != "first_arg":
            raise AssertionError(
                "only first_arg can be tested for a special case of 1-size tensor"
            )
        shape_x[0] = 1
    if dyn_shape not in ("True", "False"):
        raise AssertionError(f"dyn_shape must be 'True' or 'False', got {dyn_shape}")
    dynamic_shapes = dyn_shape == "True"

    x = torch.randn(shape_x, device=GPU_TYPE)
    y = torch.arange(4, device=GPU_TYPE)
    fn = vars()[fn_name]
    fn = torch.compile(dynamic=dynamic_shapes)(fn)
    if fn_name == "store":
        shape = (y.numel(),) + x.shape[2:]
        z = torch.randn(shape, device=GPU_TYPE)
        fn(x, y, z)
        # On Windows, Python will optimize away a function call if its updated value is not used.
        # Touch the memory of x so that the fn(x, y, z) will not be optimized away
        print(x)
    elif fn_name in ("upper1", "upper2", "lower1", "lower2"):
        print(fn(x))
    else:
        print(fn(x, y))
