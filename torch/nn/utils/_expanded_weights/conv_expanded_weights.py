import torch
import torch.nn.functional as F

from .conv_utils import conv_backward, conv_args_and_kwargs
from .expanded_weights_impl import ExpandedWeight, implements_per_sample_grads
from .expanded_weights_utils import forward_helper

@implements_per_sample_grads(F.conv1d)
@implements_per_sample_grads(F.conv2d)
@implements_per_sample_grads(F.conv3d)
class ConvPerSampleGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kwarg_names, conv_fn, *expanded_args_and_kwargs):
        if any([isinstance(i, str) for i in expanded_args_and_kwargs]):
            raise RuntimeError("Expanded Weights does not support convolution padding as a string. "
                               "Please file an issue to prioritize support")
        expanded_args, expanded_kwargs = conv_args_and_kwargs(kwarg_names, expanded_args_and_kwargs)
        output = forward_helper(conv_fn, expanded_args, expanded_kwargs)
        input, weight = expanded_args

        ctx.conv_fn = conv_fn

        ctx.batch_size = input.shape[0]
        ctx.input_required_grad = input.requires_grad
        ctx.stride, ctx.padding = expanded_kwargs['stride'], expanded_kwargs['padding']
        ctx.dilation, ctx.groups = expanded_kwargs['dilation'], expanded_kwargs['groups']

        if isinstance(weight, ExpandedWeight):
            ctx.input = input
        ctx.weight = weight
        ctx.bias = expanded_kwargs['bias']

        return output

    @staticmethod
    def backward(ctx, grad_output):
        return conv_backward(ctx.conv_fn, ctx, grad_output)
