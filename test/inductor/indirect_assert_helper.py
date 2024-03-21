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


if __name__ == "__main__":
    _, fn_name, dims, dyn_shape, one_size = sys.argv
    assert fn_name in ("first_arg", "second_arg", "same_pm_one", "same_pp_one", "store")
    assert one_size in ("True", "False")
    one_size = one_size == "True"
    assert dims in ("2", "3")
    shape_x = [3, 2, 4] if dims == "3" else [3, 2]
    if one_size:
        assert (
            fn_name == "first_arg"
        ), "only first_arg can be tested for a special case of 1-size tensor"
        shape_x[0] = 1
    assert dyn_shape in ("True", "False")
    dynamic_shapes = dyn_shape == "True"

    x = torch.randn(shape_x, device=GPU_TYPE)
    y = torch.arange(4, device=GPU_TYPE)
    fn = vars()[fn_name]
    fn = torch.compile(dynamic=dynamic_shapes)(fn)
    if fn_name == "store":
        shape = (y.numel(),) + x.shape[2:]
        z = torch.randn(shape, device=GPU_TYPE)
        fn(x, y, z)
    else:
        fn(x, y)
