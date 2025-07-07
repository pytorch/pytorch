# mypy: allow-untyped-defs
import numpy as np

import torch
from torch import Tensor
from torch.ao.quantization.experimental.apot_utils import (
    apot_to_float_tensor,
    float_to_apot_tensor,
    quant_dequant_tensor,
)


# class to store APoT quantizer and
# implement quantize and dequantize
class APoTQuantizer:
    alpha: torch.Tensor
    gamma: torch.Tensor
    quantization_levels: torch.Tensor
    level_indices: torch.Tensor
    quantization_partitions: torch.Tensor

    def __init__(
        self,
        alpha: torch.Tensor,
        gamma: torch.Tensor,
        quantization_levels: torch.Tensor,
        level_indices: torch.Tensor,
        quantization_partitions: torch.Tensor,
    ) -> None:
        self.alpha = alpha
        self.gamma = gamma
        self.quantization_levels = quantization_levels
        self.level_indices = level_indices
        self.quantization_partitions = quantization_partitions

    r""" Quantizes fp Tensor to integer APoT representation.
    Conversion is based on the qparams from a specified APoT non-uniform observer.
    The approach follows the method outlined in the APoT paper: https://arxiv.org/pdf/1909.13144.pdf.
    Args:
        tensor2quantize: fp Tensor
    Returns:
        result: APoT Tensor representation of tensor2quantize
    """

    def quantize(self, tensor2quantize: Tensor):
        result = torch.tensor([])

        tensor2quantize = float_to_apot_tensor(tensor2quantize, self.quantization_levels, self.level_indices, self.quantization_partitions, self.alpha)

        # convert to APoT int representation for dtype
        tensor2quantize = tensor2quantize.int()

        from torch.ao.quantization.experimental.APoT_tensor import TensorAPoT

        result = TensorAPoT(self, tensor2quantize)  # type: ignore[assignment]

        return result

    r""" Dequantizes integer Tensor to floating point (fp) representation
    based on the calculated quantization levels from a specified APoT non-uniform observer.
    The approach follows the method outlined in the APoT paper: https://arxiv.org/pdf/1909.13144.pdf.
    Args:
        tensor2quantize: fp Tensor
    Returns:
        result: fp reduced precision representation of input Tensor
    """

    def dequantize(self, apot_tensor) -> Tensor:

        result = apot_to_float_tensor(apot_tensor.data, self.quantization_levels, self.level_indices)

        return result

    r""" Returns result of quantize -> dequantize on a fp Tensor (reduced precision)
    based on the calculated quantization levels from a specified APoT non-uniform observer.
    The approach follows the method outlined in the APoT paper: https://arxiv.org/pdf/1909.13144.pdf.
    Args:
        apot_tensor: quantized APoT Tensor to dequantize
    Returns:
        result: fp representation of input Tensor
    """

    def quant_dequant(self, tensor2quantize: Tensor) -> Tensor:
        self.quantization_levels = quantization_levels
        self.level_indices = level_indices
        self.quantization_partitions = quantization_partitions

        result = quant_dequant_util(x, quantization_levels, level_indices, quantization_partitions)

        return result

    def q_apot_alpha(self) -> float:
        raise NotImplementedError


r""" Global method to create quantizer and call quantizer quantize_APoT
    Args:
        tensor2quantize: fp Tensor to quantize
        alpha: Tensor qparam alpha (clipping level)
        gamma: Tensor qparam gamma (scale factor for quantization levels)
        quantization levels: Tensor with fp quantization levels
        level indices: Tensor with integer quantization level indices
    Returns:
        result: ApoT Tensor representation of tensor2quantize
"""


def quantize_APoT(
    tensor2quantize: Tensor,
    alpha: Tensor,
    gamma: Tensor,
    quantization_levels: Tensor,
    level_indices: Tensor,
    quantization_partitions: Tensor,
):
    quantizer = APoTQuantizer(
        alpha=alpha,
        gamma=gamma,
        quantization_levels=quantization_levels,
        level_indices=level_indices,
        quantization_partitions=quantization_partitions,
    )
    result = quantizer.quantize(tensor2quantize)
    return result


r""" Global method to create quantizer and call quantizer dequantize_APoT
    Args:
        apot_tensor: APoT Tensor to dequantize
    Returns:
        result: fp Tensor dequantized from apot_tensor
"""


def dequantize_APoT(apot_tensor) -> Tensor:
    quantizer = apot_tensor.quantizer
    result = quantizer.dequantize(apot_tensor)
    return result


r""" Global method to create quantizer and call quantizer quant_dequant
    Args:
        tensor2quantize: fp Tensor to quantize
        alpha: Tensor qparam alpha (clipping level)
        gamma: Tensor qparam gamma (scale factor for quantization levels)
        quantization levels: Tensor with fp quantization levels
        level indices: Tensor with integer quantization level indices
    Returns:
        result: fp reduced precision Tensor from tensor2quantize
"""


def quant_dequant_APoT(
    tensor2quantize: Tensor,
    alpha: Tensor,
    gamma: Tensor,
    quantization_levels: Tensor,
    level_indices: Tensor,
    quantization_partitions: Tensor,
) -> Tensor:
    quantizer = APoTQuantizer(
        alpha=alpha,
        gamma=gamma,
        quantization_levels=quantization_levels,
        level_indices=level_indices,
        quantization_partitions=quantization_partitions,
    )
    result = quantizer.quant_dequant(tensor2quantize)
    return result
