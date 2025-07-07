import torch
from torch import Tensor
from torch.ao.quantization.experimental.quantizer import dequantize_APoT, quantize_APoT


class fake_quantize_function(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        x: Tensor,
        alpha: Tensor,
        gamma: Tensor,
        quantization_levels: Tensor,
        level_indices: Tensor,
        quantization_partitions: Tensor,
    ) -> Tensor:
        quantized_result = quantize_APoT(
            x, alpha, gamma, quantization_levels, level_indices, quantization_partitions
        )

        # calculate mask tensor
        mask = torch.where(x <= alpha, 1, 0) & torch.where(x >= -alpha, 1, 0)

        result = dequantize_APoT(quantized_result)

        ctx.save_for_backward(mask)

        return result

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: Tensor,
    ) -> Tensor:
        mask = ctx.saved_tensors[0]  # type: ignore[attr-defined]
        return grad_output * mask, None, None, None, None, None
