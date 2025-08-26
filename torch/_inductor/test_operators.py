from typing import Any

import torch.library
from torch import Tensor
from torch.autograd import Function


_test_lib_def = torch.library.Library("_inductor_test", "DEF")
_test_lib_def.define("realize(Tensor self) -> Tensor", tags=torch.Tag.pt2_compliant_tag)

_test_lib_impl = torch.library.Library("_inductor_test", "IMPL")
for dispatch_key in ("CPU", "CUDA", "MPS", "Meta"):
    _test_lib_impl.impl("realize", lambda x: x.clone(), dispatch_key)


class Realize(Function):
    @staticmethod
    def forward(ctx: object, x: Tensor) -> Tensor:
        return torch.ops._inductor_test.realize(x)

    @staticmethod
    # types need to stay consistent with _SingleLevelFunction
    def backward(ctx: Any, *grad_output: Any) -> Any:
        return grad_output[0]


def realize(x: Tensor) -> Tensor:
    return Realize.apply(x)
