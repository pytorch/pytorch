import torch
from torch.testing._internal.subclasses import BaseWithOverride
from torch._subclasses.base import BaseTensorSubclass as BaseTSC

@BaseWithOverride.torch_dispatch_override(
    ops={torch.ops.aten.mul.Tensor},
)
def torch_disp_mul(cls, func, types, args=(), kwargs=None):  # noqa: B902
    print(f"{cls}.torch_disp_mull {func} {types}")
    return BaseTSC.default_torch_dispatch(cls, func, types, args, kwargs)
