# mypy: allow-untyped-defs
import torch.library
from torch import Tensor
from torch.autograd import Function


if not torch._running_with_deploy():
    _test_lib_def = torch.library.Library("_inductor_test", "DEF")
    _test_lib_def.define(
        "realize(Tensor self) -> Tensor", tags=torch.Tag.pt2_compliant_tag
    )

    _test_lib_impl = torch.library.Library("_inductor_test", "IMPL")
    for dispatch_key in ("CPU", "CUDA", "Meta"):
        _test_lib_impl.impl("realize", lambda x: x.clone(), dispatch_key)

    class Realize(Function):
        @staticmethod
        def forward(ctx, x):
            return torch.ops._inductor_test.realize(x)

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output

    def realize(x: Tensor) -> Tensor:
        return Realize.apply(x)
