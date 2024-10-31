# mypy: allow-untyped-defs
import torch
import torch.nn.functional as F

from .conv_utils import (
    conv_args_and_kwargs,
    conv_backward,
    conv_input_for_string_padding,
    conv_picker,
)
from .expanded_weights_impl import ExpandedWeight, implements_per_sample_grads
from .expanded_weights_utils import forward_helper


@implements_per_sample_grads(F.conv1d)
@implements_per_sample_grads(F.conv2d)
@implements_per_sample_grads(F.conv3d)
class ConvPerSampleGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kwarg_names, conv_fn, *expanded_args_and_kwargs):
        expanded_args, expanded_kwargs = conv_args_and_kwargs(
            kwarg_names, expanded_args_and_kwargs
        )
        orig_input = expanded_args[0]
        was_same_padding = expanded_kwargs["padding"] == "same"

        if isinstance(expanded_kwargs["padding"], str):
            # if padding is a string, we'll do the necessary padding (slowly) using F.pad
            kernel_size = expanded_args[1].shape[2:]
            padding, dilation = expanded_kwargs["padding"], expanded_kwargs["dilation"]
            input = conv_input_for_string_padding(
                conv_fn, padding, expanded_args[0], dilation, kernel_size
            )
            expanded_args = (input, expanded_args[1])
            # since we've already done the padding, don't need any more
            expanded_kwargs["padding"] = 0

        output = forward_helper(conv_fn, expanded_args, expanded_kwargs)
        input, weight = expanded_args
        batched_dim_size = conv_picker(conv_fn, 3, 4, 5)
        if input.dim() != batched_dim_size:
            raise RuntimeError(
                f"Expanded Weights only support convolution with batched input, got {conv_fn} with an"
                f"unbatched input of dim {input.dim()}, expected input of dim {batched_dim_size}"
            )

        ctx.conv_fn = conv_fn

        ctx.batch_size = orig_input.shape[0]
        ctx.input_required_grad = orig_input.requires_grad
        ctx.orig_input_shape = orig_input.shape
        ctx.was_same_padding = was_same_padding
        ctx.stride, ctx.padding = expanded_kwargs["stride"], expanded_kwargs["padding"]
        ctx.dilation, ctx.groups = (
            expanded_kwargs["dilation"],
            expanded_kwargs["groups"],
        )

        if isinstance(weight, ExpandedWeight):
            ctx.input = input
        ctx.weight = weight
        ctx.bias = expanded_kwargs["bias"]

        return output

    @staticmethod
    def backward(ctx, grad_output):
        return conv_backward(ctx.conv_fn, ctx, grad_output)
