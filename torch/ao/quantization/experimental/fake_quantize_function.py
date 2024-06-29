# mypy: allow-untyped-defs
import torch
from torch import Tensor
from torch.ao.quantization.experimental.quantizer import quantize_APoT, dequantize_APoT

class fake_quantize_function(torch.autograd.Function):
    @staticmethod
    def forward(ctx,  # type: ignore[override]
                x: Tensor,
                alpha: Tensor,
                gamma: Tensor,
                quantization_levels: Tensor,
                level_indices: Tensor) -> Tensor:
        quantized_result = quantize_APoT(x, alpha, gamma, quantization_levels, level_indices)

        # calculate mask tensor
        mask = x.detach().apply_(lambda x: (x <= alpha and x >= -alpha))

        result = dequantize_APoT(quantized_result)

        ctx.save_for_backward(mask)

        return result

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tensor:  # type: ignore[override]
        mask = ctx.saved_tensors
        return grad_output * mask
