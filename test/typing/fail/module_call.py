# flake8: noqa
# Opting into a call signature via `nn.Module[params, return]` rejects an
# incompatible `forward` override and mistyped `model(...)` / helper calls.
from typing_extensions import ParamSpec, TypeVar

from torch import nn, Tensor


_P = ParamSpec("_P")
_R = TypeVar("_R")


class BadModule(nn.Module[[Tensor], Tensor]):
    def forward(self, x: str) -> int:  # E: [override]
        return 1


class TensorModule(nn.Module[[Tensor], Tensor]):
    def forward(self, x: Tensor) -> Tensor:
        return x


def call_generic(module: nn.Module[_P, _R], *args: _P.args, **kwargs: _P.kwargs) -> _R:
    return module(*args, **kwargs)


def check(t: TensorModule) -> None:
    t("bad")  # E: incompatible type "str"; expected "Tensor"
    call_generic(t, "bad")  # E: incompatible type "str"; expected "Tensor"
