import torch
import sys

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
    _, fn_name, dims, dyn_shape = sys.argv
    assert fn_name in ("first_arg", "second_arg", "same_pm_one", "same_pp_one", "store")
    assert dims in ("2", "3")
    shape_x = (3, 2, 4) if dims == "3" else (3, 2)
    assert dyn_shape in ("True", "False")
    dynamic_shapes = dyn_shape == "True"

    x = torch.randn(shape_x, device="cuda")
    y = torch.arange(4, device="cuda")
    fn = vars()[fn_name]
    fn = torch.compile(dynamic=dynamic_shapes)(fn)
    if fn_name == "store":
        shape = (y.numel(),) + x.shape[2:]
        z = torch.randn(shape, device="cuda")
        fn(x, y, z)
    else:
        fn(x, y)
