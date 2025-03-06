from pathlib import Path

import torch


so_files = list(Path(__file__).parent.glob("_C*.so"))
assert len(so_files) == 1, f"Expected one _C*.so file, found {len(so_files)}"
torch.ops.load_library(so_files[0])

from . import ops


# ----------------------------------------------------------------------------- #
# We've reached the end of what is normal in __init__ files.
# The following is used to assert the sgd op is properly loaded and
# calculates correct results upon import of this extension.


def test_slow_sgd():
    param = torch.rand(5, device="cpu")
    grad = torch.rand_like(param)
    weight_decay = 0.01
    lr = 0.001
    maximize = False

    new_param = ops.sgd_out_of_place(param, grad, weight_decay, lr, maximize)
    torch._fused_sgd_(
        (param,),
        (grad,),
        (),
        weight_decay=weight_decay,
        momentum=0.0,
        lr=lr,
        dampening=0.0,
        nesterov=False,
        maximize=maximize,
        is_first_step=False,
    )
    assert torch.equal(new_param, param), f"{new_param=}, {param=}"
    print(param)


test_slow_sgd()
test_slow_sgd()
test_slow_sgd()
