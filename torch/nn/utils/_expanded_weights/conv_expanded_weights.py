import torch
import torch.nn.functional as F

from .conv_utils import conv_backward, conv_args_and_kwargs
from .expanded_weights_impl import implements_per_sample_grads
from .expanded_weights_utils import forward_helper

THRESHOLD = 257

@implements_per_sample_grads(F.conv1d)
class Conv1dPerSampleGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *expanded_args_and_kwargs):
        if any([isinstance(i, str) for i in expanded_args_and_kwargs]):
            raise RuntimeError("Expanded Weights does not support convolution padding as a string. "
                               "Please file an issue to prioritize support")
        expanded_args, expanded_kwargs = conv_args_and_kwargs(expanded_args_and_kwargs)
        output, aux_outputs = forward_helper(F.conv1d, expanded_args, expanded_kwargs, 1)
        ctx.args = expanded_args
        ctx.kwargs = expanded_kwargs
        ctx.aux_outputs = aux_outputs
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return conv_backward(F.conv1d, ctx, grad_output)


@implements_per_sample_grads(F.conv2d)
class Conv2dPerSampleGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *expanded_args_and_kwargs):
        if any([isinstance(i, str) for i in expanded_args_and_kwargs]):
            raise RuntimeError("Expanded Weights does not support convolution padding as a string. "
                               "Please file an issue to prioritize support")
        expanded_args, expanded_kwargs = conv_args_and_kwargs(expanded_args_and_kwargs)
        output, aux_outputs = forward_helper(F.conv2d, expanded_args, expanded_kwargs, 1)
        ctx.args = expanded_args
        ctx.kwargs = expanded_kwargs
        ctx.aux_outputs = aux_outputs
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return conv_backward(F.conv2d, ctx, grad_output)


@implements_per_sample_grads(F.conv3d)
class Conv3dPerSampleGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *expanded_args_and_kwargs):
        if any([isinstance(i, str) for i in expanded_args_and_kwargs]):
            raise RuntimeError("Expanded Weights does not support convolution padding as a string. "
                               "Please file an issue to prioritize support")
        expanded_args, expanded_kwargs = conv_args_and_kwargs(expanded_args_and_kwargs)
        output, aux_outputs = forward_helper(F.conv3d, expanded_args, expanded_kwargs, 1)
        ctx.args = expanded_args
        ctx.kwargs = expanded_kwargs
        ctx.aux_outputs = aux_outputs
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return conv_backward(F.conv3d, ctx, grad_output)
