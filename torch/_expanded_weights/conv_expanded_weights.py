import torch
import torch.nn.functional as F

from torch._expanded_weights.conv_utils import conv_backward
from .expanded_weights_impl import implements_per_sample_grads
from .expanded_weights_utils import forward_helper

THRESHOLD = 257

@implements_per_sample_grads(F.conv1d)
class Conv1dPerSampleGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *expanded_args):
        if any([isinstance(i, str) for i in expanded_args]):
            raise RuntimeError("Expanded Weights does not support convolution padding as a string. "
                               "Please file an issue to prioritize support")
        return forward_helper(F.conv1d, ctx, expanded_args, num_true_outs=1)

    @staticmethod
    def backward(ctx, grad_output):
        return conv_backward(F.conv1d, ctx, grad_output)


@implements_per_sample_grads(F.conv2d)
class Conv2dPerSampleGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *expanded_args):
        if any([isinstance(i, str) for i in expanded_args]):
            raise RuntimeError("Expanded Weights does not support convolution padding as a string. "
                               "Please file an issue to prioritize support")
        return forward_helper(F.conv2d, ctx, expanded_args, num_true_outs=1)

    @staticmethod
    def backward(ctx, grad_output):
        return conv_backward(F.conv2d, ctx, grad_output)


@implements_per_sample_grads(F.conv3d)
class Conv3dPerSampleGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *expanded_args):
        if any([isinstance(i, str) for i in expanded_args]):
            raise RuntimeError("Expanded Weights does not support convolution padding as a string. "
                               "Please file an issue to prioritize support")
        return forward_helper(F.conv3d, ctx, expanded_args, num_true_outs=1)

    @staticmethod
    def backward(ctx, grad_output):
        return conv_backward(F.conv3d, ctx, grad_output)
