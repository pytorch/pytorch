from typing import Any
from typing_extensions import Self

import torch
import torch.ao.nn.intrinsic as nni
import torch.ao.nn.quantized.dynamic as nnqd


__all__ = ["LinearReLU"]


class LinearReLU(nnqd.Linear):
    r"""
    A LinearReLU module fused from Linear and ReLU modules that can be used
    for dynamic quantization.
    Supports both, FP16 and INT8 quantization.

    We adopt the same interface as :class:`torch.ao.nn.quantized.dynamic.Linear`.

    Attributes:
        Same as torch.ao.nn.quantized.dynamic.Linear

    Examples::

        >>> # xdoctest: +SKIP
        >>> m = nn.intrinsic.quantized.dynamic.LinearReLU(20, 30)
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
        dtype: torch.dtype = torch.qint8,
    ) -> None:
        super().__init__(in_features, out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._packed_params.dtype == torch.qint8:
            # TODO check if we should set reduce_rage = True by default here
            Y = torch.ops.quantized.linear_relu_dynamic(
                x, self._packed_params._packed_params, reduce_range=True
            )
        elif self._packed_params.dtype == torch.float16:
            Y = torch.ops.quantized.linear_relu_dynamic_fp16(
                x, self._packed_params._packed_params
            )
        else:
            raise RuntimeError("Unsupported dtype on dynamic quantized linear relu!")
        return Y.to(x.dtype)

    def _get_name(self) -> str:
        return "DynamicQuantizedLinearReLU"

    @classmethod
    def from_float(
        cls, mod: torch.nn.Module, use_precomputed_fake_quant: bool = False
    ) -> Self:
        return super().from_float(
            mod, use_precomputed_fake_quant=use_precomputed_fake_quant
        )

    @classmethod
    def from_reference(cls, ref_qlinear_relu: Any) -> Self:  # type: ignore[override]
        return super().from_reference(ref_qlinear_relu[0])
