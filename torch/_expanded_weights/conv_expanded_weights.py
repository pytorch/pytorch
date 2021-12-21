import torch
import torch.nn.functional as F
from .expanded_weights_impl import forward_helper, implements_per_sample_grads
from .expanded_weights_utils import grad_if_exists, grad_if_exists_for_input, unpack_expanded_weight_or_tensor

import numpy as np

THRESHOLD = 257

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
        def compute_input_grad():
            height = input.shape[2]
            width = input.shape[3]

            # compute output padding
            output_padding_height = (2 * padding[0] + height - (kernel_height * dilation[0] - dilation[0] + 1)) % stride[0]
            output_padding_width = (2 * padding[1] + width - (kernel_width * dilation[1] - dilation[1] + 1)) % stride[1]
            output_padding = (output_padding_height, output_padding_width)
            weight_ = unpack_expanded_weight_or_tensor(weight)
            return F.conv_transpose2d(grad_output, weight_, None, stride, padding, output_padding, groups, dilation)

        def weight_grad_sample(weight):
            if (batch_size < THRESHOLD and groups == 1):
                # TODO (samdow) conv_group to deal with groups
                return conv_group_weight_grad_sample(input, grad_output, weight, stride, padding, dilation, batch_size)
            else:
                return conv_unfold_weight_grad_sample(input, grad_output, weight, kernel_size, stride, padding, dilation, groups)

        def expand(param):
            if isinstance(param, int):
                return (param, param)
            else:
                return param

        (input, weight, bias, stride, padding, dilation, groups) = ctx.args
        stride, padding, dilation = expand(stride), expand(padding), expand(dilation)

        kernel_height = weight.shape[2]
        kernel_width = weight.shape[3]

        kernel_size = (kernel_height, kernel_width)
        batch_size = input.shape[0]
        results = []

        results.append(grad_if_exists_for_input(input, compute_input_grad))
        results.append(grad_if_exists(weight, weight_grad_sample))
        results.append(grad_if_exists(bias, lambda _: grad_output.reshape(*grad_output.shape[:2], -1).sum(dim=2)))

        # no other arguments are differentiable
        results = results + [None] * (len(ctx.args) - 3)
        return tuple(results)

def conv_unfold_weight_grad_sample(input, grad_output, weight, kernel_size, stride, padding, dilation, groups):
    n = input.shape[0]
    in_channels = input.shape[1]

    input = torch.nn.functional.unfold(
        input,
        kernel_size,
        padding=padding,
        stride=stride,
        dilation=dilation,
    )
    grad_output = grad_output.reshape(n, -1, input.shape[-1])

    # n=batch_sz; o=num_out_channels; p=(num_in_channels/groups)*kernel_sz
    weight_grad_sample = torch.einsum("noq,npq->nop", grad_output, input)
    # rearrange the above tensor and extract diagonals.
    weight_grad_sample = weight_grad_sample.view(
        n,
        groups,
        -1,
        groups,
        int(in_channels / groups),
        np.prod(kernel_size),
    )
    weight_grad_sample = torch.einsum("ngrg...->ngr...", weight_grad_sample).contiguous()
    shape = [n] + list(weight.shape)
    weight_grad_sample = weight_grad_sample.view(shape)
    return weight_grad_sample

def conv_group_weight_grad_sample(input, grad_output, weight, stride, padding, dilation, batch_size):
    I = input.shape[1]
    O = grad_output.shape[1]

    input_ = input.transpose(0, 1)
    grad_output_ = grad_output.view(grad_output.shape[0] * grad_output.shape[1], 1, grad_output.shape[2], grad_output.shape[3])

    weight_grad_sample = F.conv2d(input_, grad_output_, None, stride=dilation, padding=padding, dilation=stride, groups=batch_size)
    weight_grad_sample = weight_grad_sample.narrow(2, 0, weight.shape[2]).narrow(3, 0, weight.shape[3])
    weight_grad_sample = weight_grad_sample.view(I, batch_size, O, *weight_grad_sample.shape[-2:])
    weight_grad_sample = weight_grad_sample.movedim(0, 2)
    return weight_grad_sample
