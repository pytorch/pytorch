from __future__ import annotations

from typing import Optional

import torch
import torch.ao.nn.intrinsic as nni
import torch.ao.nn.qat as nnqat
import torch.nn.functional as F
from torch.ao.nn.intrinsic.modules.fused import _FusedModule


__all__ = ["LinearReLU"]


class LinearReLU(nnqat.Linear, _FusedModule):
    r"""
    A LinearReLU module fused from Linear and ReLU modules, attached with
    FakeQuantize modules for weight, used in
    quantization aware training.

    We adopt the same interface as :class:`torch.nn.Linear`.

    Similar to `torch.ao.nn.intrinsic.LinearReLU`, with FakeQuantize modules initialized to
    default.

    Attributes:
        weight: fake quant module for weight

    Examples::

        >>> # xdoctest: +SKIP
        >>> m = nn.qat.LinearReLU(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """

    # pyrefly: ignore  # bad-override
    _FLOAT_MODULE = nni.LinearReLU

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        qconfig: Optional[object] = None,
    ) -> None:
        super().__init__(in_features, out_features, bias, qconfig)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.relu(F.linear(input, self.weight_fake_quant(self.weight), self.bias))

    @classmethod
    def from_float(
        cls,
        mod: torch.nn.Module,
        use_precomputed_fake_quant: bool = False,
    ) -> LinearReLU:
        return super().from_float(mod, use_precomputed_fake_quant)  # type: ignore[no-untyped-call,no-any-return]

    def to_float(self) -> nni.LinearReLU:
        linear = torch.nn.Linear(
            self.in_features, self.out_features, self.bias is not None
        )
        linear.weight = torch.nn.Parameter(self.weight.detach())
        if self.bias is not None:
            linear.bias = torch.nn.Parameter(self.bias.detach())
        relu = torch.nn.ReLU()
        return torch.ao.nn.intrinsic.LinearReLU(linear, relu)  # type: ignore[no-untyped-call]
