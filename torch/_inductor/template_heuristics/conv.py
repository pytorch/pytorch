from __future__ import annotations

from typing import Any, TYPE_CHECKING

import torch

from ..kernel.conv import aten_convolution, conv2d_template, conv3d_template
from ..kernel_inputs import ConvKernelInputs
from ..utils import is_ones, sympy_product
from ..virtualized import V
from .base import TemplateConfigHeuristics
from .registry import register_template_heuristic
from .triton import (
    CPUConfigHeuristic,
    CUDAConfigHeuristic,
    MTIAConfigHeuristic,
    ROCmConfigHeuristic,
    XPUConfigHeuristic,
)


if TYPE_CHECKING:
    from collections.abc import Generator

    from ..kernel_inputs import KernelInputs


class ConvTemplateConfigMixin(TemplateConfigHeuristics):
    """
    Mixin for conv templates that converts config lists to template kwargs.
    Similar to MMTemplateConfigMixin but for convolutions.

    This handles generating both the static template kwargs (KERNEL_H, STRIDE_H, etc.)
    and the per-config kwargs (BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_warps).
    """

    # Type hint for methods from BaseConfigHeuristic
    get_conv_configs: Any

    def get_extra_kwargs(
        self,
        kernel_inputs: KernelInputs,
        op_name: str,
    ) -> dict[str, Any]:
        """
        Return template kwargs that don't change per-config.
        These are derived from kernel_inputs and must include all template parameters.

        Args:
            kernel_inputs: ConvKernelInputs containing input tensors and conv params
            op_name: Operation name (e.g., "convolution")

        Returns:
            Dict of static template kwargs (KERNEL_H, STRIDE_H, GROUPS, etc.)
        """
        assert isinstance(kernel_inputs, ConvKernelInputs), (
            f"ConvTemplateConfigMixin requires ConvKernelInputs, got {type(kernel_inputs)}"
        )

        x, weight, bias = kernel_inputs.get_x_weight_bias()

        # Extract kernel shape from weight: [out_chan, in_chan, *kernel_shape]
        weight_size = V.graph.sizevars.guard_int_seq(weight.get_size())
        kernel_shape = weight_size[2:]  # Skip out_chan, in_chan
        ndim = len(kernel_shape)

        # Extract scalars
        stride = kernel_inputs.get_scalar("stride")
        padding = kernel_inputs.get_scalar("padding")
        groups = kernel_inputs.get_scalar("groups")

        # Check if we should unroll (only for 1x1 kernels)
        unroll = is_ones(kernel_shape)

        # Build kwargs dict based on ndim
        kwargs = {
            "GROUPS": groups,
            "UNROLL": unroll,
            "ALLOW_TF32": torch.backends.cudnn.allow_tf32,
        }

        if ndim == 2:
            kwargs.update(
                {
                    "KERNEL_H": kernel_shape[0],
                    "KERNEL_W": kernel_shape[1],
                    "STRIDE_H": stride[0],
                    "STRIDE_W": stride[1],
                    "PADDING_H": padding[0],
                    "PADDING_W": padding[1],
                }
            )
        elif ndim == 3:
            kwargs.update(
                {
                    "KERNEL_D": kernel_shape[0],
                    "KERNEL_H": kernel_shape[1],
                    "KERNEL_W": kernel_shape[2],
                    "STRIDE_D": stride[0],
                    "STRIDE_H": stride[1],
                    "STRIDE_W": stride[2],
                    "PADDING_D": padding[0],
                    "PADDING_H": padding[1],
                    "PADDING_W": padding[2],
                }
            )

        return kwargs

    def _get_template_configs_impl(
        self,
        kernel_inputs: KernelInputs,
        op_name: str,
    ) -> Generator[dict[str, Any], None, None]:
        """
        Yield per-config kwargs (BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_warps).

        Args:
            kernel_inputs: ConvKernelInputs containing input tensors
            op_name: Operation name

        Yields:
            Dict of per-config kwargs for each configuration to try
        """
        assert isinstance(kernel_inputs, ConvKernelInputs), (
            "ConvTemplateConfigMixin requires ConvKernelInputs"
        )

        x, weight, bias = kernel_inputs.get_x_weight_bias()

        # Calculate dimensions for heuristics
        weight_size = weight.get_size()
        out_chan = weight_size[0]
        in_chan = weight_size[1]

        # Batch * spatial dimensions product
        x_size = x.get_size()
        batch_spatial_product = sympy_product([x_size[0], *x_size[2:]])

        # Get conv config generator from self (which is a BaseConfigHeuristic subclass)
        conv_configs_generator = self.get_conv_configs()

        dtype_size = x.get_dtype().itemsize

        # Generate configs (reusing mm preprocess_mm_configs machinery)
        for c in conv_configs_generator(
            batch_spatial_product,
            out_chan,
            in_chan,
            dtype_size=dtype_size,
            op_name="conv",
        ):
            # Yield per-config kwargs
            yield {
                "BLOCK_M": c.kwargs.get("BLOCK_M"),
                "BLOCK_N": c.kwargs.get("BLOCK_N"),
                "BLOCK_K": c.kwargs.get("BLOCK_K"),
                "num_stages": c.num_stages,
                "num_warps": c.num_warps,
            }


# ATEN convolution heuristic (no per-config tuning)
@register_template_heuristic(aten_convolution.uid, None)
class ATenConvConfigHeuristic(TemplateConfigHeuristics):
    """
    Pseudo heuristic for ATen convolution.
    ATen doesn't have configs to tune - it's a single choice.
    """

    def _get_template_configs_impl(
        self,
        kernel_inputs: KernelInputs,
        op_name: str,
    ) -> Generator[dict[str, Any], None, None]:
        # ATen doesn't have per-config kwargs to tune
        yield dict()

    def get_extra_kwargs(
        self,
        kernel_inputs: KernelInputs,
        op_name: str,
    ) -> dict[str, Any]:
        """
        ATen gets stride, padding, etc. as ordered kwargs for the C++ kernel.
        """
        assert isinstance(kernel_inputs, ConvKernelInputs)

        # Extract scalar values from kernel_inputs
        stride = kernel_inputs.get_scalar("stride")
        padding = kernel_inputs.get_scalar("padding")
        dilation = kernel_inputs.get_scalar("dilation")
        transposed = kernel_inputs.get_scalar("transposed")
        output_padding = kernel_inputs.get_scalar("output_padding")
        groups = kernel_inputs.get_scalar("groups")

        # Check if bias is None to match old behavior
        # When bias is None: input_nodes = [x, weight], add 'bias' to kwargs and ordered list
        # When bias is present: input_nodes = [x, weight, bias], don't add 'bias' to kwargs
        x, weight, bias = kernel_inputs.get_x_weight_bias()

        kwargs = {
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "transposed": transposed,
            "output_padding": output_padding,
            "groups": groups,
        }

        if bias is None:
            # When bias is None, torch.convolution expects it as a kwarg
            kwargs["bias"] = None
            kwargs["ordered_kwargs_for_cpp_kernel"] = [
                "bias",
                "stride",
                "padding",
                "dilation",
                "transposed",
                "output_padding",
                "groups",
            ]
        else:
            # When bias is present, it's passed as a positional arg (3rd in input_nodes)
            kwargs["ordered_kwargs_for_cpp_kernel"] = [
                "stride",
                "padding",
                "dilation",
                "transposed",
                "output_padding",
                "groups",
            ]

        return kwargs


# CUDA Conv2D/Conv3D heuristics
@register_template_heuristic(
    conv2d_template.uid,
    "cuda",
    register=torch.version.hip is None,
)
@register_template_heuristic(
    conv3d_template.uid,
    "cuda",
    register=torch.version.hip is None,
)
class CUDAConvTemplateConfigHeuristic(ConvTemplateConfigMixin, CUDAConfigHeuristic):
    """Conv template heuristic for CUDA."""


# ROCm Conv2D/Conv3D heuristics
@register_template_heuristic(
    conv2d_template.uid,
    "cuda",
    register=torch.version.hip is not None,
)
@register_template_heuristic(
    conv3d_template.uid,
    "cuda",
    register=torch.version.hip is not None,
)
class ROCmConvTemplateConfigHeuristic(ConvTemplateConfigMixin, ROCmConfigHeuristic):
    """Conv template heuristic for ROCm."""


# CPU Conv2D/Conv3D heuristics
@register_template_heuristic(conv2d_template.uid, "cpu")
@register_template_heuristic(conv3d_template.uid, "cpu")
class CPUConvTemplateConfigHeuristic(ConvTemplateConfigMixin, CPUConfigHeuristic):
    """Conv template heuristic for CPU."""


# XPU Conv2D/Conv3D heuristics
@register_template_heuristic(conv2d_template.uid, "xpu")
@register_template_heuristic(conv3d_template.uid, "xpu")
class XPUConvTemplateConfigHeuristic(ConvTemplateConfigMixin, XPUConfigHeuristic):
    """Conv template heuristic for XPU."""


# MTIA Conv2D/Conv3D heuristics
@register_template_heuristic(conv2d_template.uid, "mtia")
@register_template_heuristic(conv3d_template.uid, "mtia")
class MTIAConvTemplateConfigHeuristic(ConvTemplateConfigMixin, MTIAConfigHeuristic):
    """Conv template heuristic for MTIA."""
