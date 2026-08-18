# A typed `nn.Module[params, return]` subclass gives `model(...)` the same
# typed signature/return as `forward`, while a bare `nn.Module` is unchanged
# (any arguments, `Any` return). See torch/nn/modules/module.py.
from typing import Any
from typing_extensions import assert_type, ParamSpec, TypeVar

from torch import nn, Tensor


_P = ParamSpec("_P")
_R = TypeVar("_R")


class TensorModule(nn.Module[[Tensor], Tensor]):
    def forward(self, x: Tensor) -> Tensor:
        return x


class UntypedModule(nn.Module):  # bare Module -> Module[..., Any]
    def forward(self, x: Tensor) -> Tensor:
        return x


def call_generic(module: nn.Module[_P, _R], *args: _P.args, **kwargs: _P.kwargs) -> _R:
    return module(*args, **kwargs)


def check(x: Tensor) -> None:
    assert_type(TensorModule()(x), Tensor)  # typed __call__
    assert_type(TensorModule().forward(x), Tensor)  # typed forward
    assert_type(call_generic(TensorModule(), x), Tensor)  # generic over a Module
    assert_type(UntypedModule()(x), Any)  # bare Module keeps `Any`
