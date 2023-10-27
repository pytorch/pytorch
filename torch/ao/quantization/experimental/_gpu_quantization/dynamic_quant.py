import torch
import torch.nn as nn
from torch.ao.quantization.experimental._gpu_quantization.quant_primitives import (
    dynamically_quantize_per_channel,
    quant_int8_dynamic_per_token_linear,
)

__all__ = ["DynamicallyPerAxisQuantizedLinear"]


class DynamicallyPerAxisQuantizedLinear(torch.nn.Linear):
    """
    This class is a replacement for `torch.nn.Linear`, implementing dynamic quantization on
    the input across all axes except for the last axis.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        use_fused_int_mm=False,
    ) -> None:
        super().__init__(in_features, out_features, bias)
        self.use_fused_int_mm = use_fused_int_mm
        # note: enabling use_fused_int_mm = True has best perf when additionally setting
        # torch._inductor.config.force_fuse_int_mm_with_mul = True

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the quantized linear layer.

        This method applies dynamic quantization to the input tensor across all axes except
        the last axis using the `quant_int8_dynamic_per_token_linear` function.

        Args:
            X (torch.Tensor): The input tensor to the quantized linear layer.

        Returns:
            torch.Tensor: The output tensor after the quantized matmul and rescale.

        """
        # The following line mimics the behavior of SmoothFakeDynamicallyQuantizedLinear
        if not self.use_fused_int_mm:
            X = X / self.fake_rescale
        # somehow the inductor fusion that occurs for most transformer models
        # when this module has an additional div op is faster than when it doesn't
        # have it although the memory usage is slightly higher. fake_rescale is scalar 1
        # so it doesn't affect accuracy
        Y = quant_int8_dynamic_per_token_linear(
            X, self.W_int_repr_t, self.W_scales, self.bias, X.dtype
        )
        return Y

    @classmethod
    def from_float(
        cls, mod: torch.nn.Linear, use_fused_int_mm=False
    ) -> "DynamicallyPerAxisQuantizedLinear":
        """
        Converts a `mod` of class `torch.nn.Linear` to the dynamically quantized version of it.

        Note: this class does not require calibration.

        Args:
            mod (torch.nn.Linear): The original `torch.nn.Linear` module to convert.

        Returns:
            DynamicallyPerAxisQuantizedLinear: The converted quantized linear module.

        """

        # create the new module with a toy size to ensure initialization is fast
        fake_in_features, fake_out_features = 8, 8
        new_mod = cls(
            fake_in_features,
            fake_out_features,
            bias=mod.bias is not None,
            use_fused_int_mm=use_fused_int_mm,
        )
        new_mod.in_features = mod.in_features
        new_mod.out_features = mod.out_features
        W_int_repr, W_scales, _W_zps = dynamically_quantize_per_channel(
            mod.weight, -128, 127, torch.int8
        )
        new_mod.register_buffer("W_int_repr_t", W_int_repr.contiguous().t())
        new_mod.W_scales = nn.Parameter(W_scales)
        new_mod.bias = mod.bias
        if not use_fused_int_mm:
            new_mod.fake_rescale = torch.tensor(
                [1.0], dtype=mod.weight.dtype, device=mod.weight.device
            )
        del new_mod.weight

        device_to_use = next(mod.parameters()).device
        new_mod.to(device_to_use)
        return new_mod
