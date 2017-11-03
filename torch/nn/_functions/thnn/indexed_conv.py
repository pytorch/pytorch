from torch.autograd.function import Function, once_differentiable
from torch._thnn import type2backend

from . import _all_functions


class IndexedConv(Function):

    @staticmethod
    def forward(ctx, input, indices, weight, bias):

        input = input.contiguous()

        ctx._backend = type2backend[type(input)]

        ctx.indices = indices

        output = input.new()
        columns = input.new()
        ones = input.new()

        ctx._backend.IndexedConvolution_updateOutput(ctx._backend.library_state,
                                                     input, output,
                                                     weight, bias,
                                                     indices, columns, ones)

        ctx.save_for_backward(input, weight, bias)

        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):

        input, weight, bias = ctx.saved_tensors
        input = input.contiguous()

        indices = ctx.indices

        columns = input.new()
        ones = input.new()

        grad_input = grad_output.new()
        grad_columns = columns.new()

        ctx._backend.IndexedConvolution_updateGradInput(ctx._backend.library_state,
                                                        input, grad_output,
                                                        grad_input,
                                                        weight, indices,
                                                        grad_columns)

        grad_weight = weight.new().resize_as_(weight).zero_()
        grad_bias = bias.new().resize_as_(bias).zero_()

        scale = 1.0

        ctx._backend.IndexedConvolution_accGradParameters(ctx._backend.library_state,
                                                          input, grad_output,
                                                          grad_weight, grad_bias,
                                                          indices,
                                                          columns, ones,
                                                          scale)

        return grad_input, None, grad_weight, grad_bias


_all_functions.append(IndexedConv)
