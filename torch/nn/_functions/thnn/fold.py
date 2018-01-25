from torch.autograd.function import Function, once_differentiable
from torch._thnn import type2backend

from . import _all_functions


class Col2Row(Function):

    @staticmethod
    def forward(ctx, input, output_size, kernel_size, dilation, padding, stride):

        ctx.output_size = output_size
        ctx.kernel_size = kernel_size
        ctx.dilation = dilation
        ctx.padding = padding
        ctx.stride = stride

        ctx._backend = type2backend[type(input)]

        output = input.new()

        ctx._backend.Col2Row_updateOutput(ctx._backend.library_state,
                                          input, output,
                                          output_size[0],
                                          kernel_size[0],
                                          dilation[0],
                                          padding[0],
                                          stride[0])
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):

        grad_input = grad_output.new()

        ctx._backend.Col2Row_updateGradInput(ctx._backend.library_state,
                                             grad_output,
                                             grad_input,
                                             ctx.kernel_size[0],
                                             ctx.dilation[0],
                                             ctx.padding[0],
                                             ctx.stride[0])
        return grad_input, None, None, None, None, None


class Row2Col(Function):

    @staticmethod
    def forward(ctx, input, kernel_size, dilation, padding, stride):

        assert input.dim() == 3

        ctx.kernel_size = kernel_size
        ctx.dilation = dilation
        ctx.padding = padding
        ctx.stride = stride
        ctx.input_size = (input.size(2), )

        ctx._backend = type2backend[type(input)]

        output = input.new()

        ctx._backend.Vol2Col_updateOutput(ctx._backend.library_state,
                                          input, output,
                                          kernel_size[0],
                                          dilation[0],
                                          padding[0],
                                          stride[0])
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):

        grad_input = grad_output.new()

        ctx._backend.Vol2Col_updateGradInput(ctx._backend.library_state,
                                             grad_output,
                                             grad_input,
                                             ctx.input_size[0],
                                             ctx.kernel_size[0],
                                             ctx.dilation[0],
                                             ctx.padding[0],
                                             ctx.stride[0])
        return grad_input, None, None, None, None


class Col2Im(Function):

    @staticmethod
    def forward(ctx, input, output_size, kernel_size, dilation, padding, stride):

        ctx.output_size = output_size
        ctx.kernel_size = kernel_size
        ctx.dilation = dilation
        ctx.padding = padding
        ctx.stride = stride

        ctx._backend = type2backend[input.type()]

        output = input.new()

        ctx._backend.Col2Im_updateOutput(ctx._backend.library_state,
                                         input, output,
                                         output_size[0], output_size[1],
                                         kernel_size[0], kernel_size[1],
                                         dilation[0], dilation[1],
                                         padding[0], padding[1],
                                         stride[0], stride[1])
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):

        grad_input = grad_output.new()

        ctx._backend.Col2Im_updateGradInput(ctx._backend.library_state,
                                            grad_output,
                                            grad_input,
                                            ctx.kernel_size[0], ctx.kernel_size[1],
                                            ctx.dilation[0], ctx.dilation[1],
                                            ctx.padding[0], ctx.padding[1],
                                            ctx.stride[0], ctx.stride[1])
        return grad_input, None, None, None, None, None


class Im2Col(Function):

    @staticmethod
    def forward(ctx, input, kernel_size, dilation, padding, stride):

        assert input.dim() == 4

        ctx.kernel_size = kernel_size
        ctx.dilation = dilation
        ctx.padding = padding
        ctx.stride = stride
        ctx.input_size = (input.size(2), input.size(3))

        ctx._backend = type2backend[input.type()]

        output = input.new()

        ctx._backend.Im2Col_updateOutput(ctx._backend.library_state,
                                         input, output,
                                         kernel_size[0], kernel_size[1],
                                         dilation[0], dilation[1],
                                         padding[0], padding[1],
                                         stride[0], stride[1])
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):

        grad_input = grad_output.new()

        ctx._backend.Im2Col_updateGradInput(ctx._backend.library_state,
                                            grad_output,
                                            grad_input,
                                            ctx.input_size[0], ctx.input_size[1],
                                            ctx.kernel_size[0], ctx.kernel_size[1],
                                            ctx.dilation[0], ctx.dilation[1],
                                            ctx.padding[0], ctx.padding[1],
                                            ctx.stride[0], ctx.stride[1])
        return grad_input, None, None, None, None


class Col2Vol(Function):

    @staticmethod
    def forward(ctx, input, output_size, kernel_size, dilation, padding, stride):

        ctx.output_size = output_size
        ctx.kernel_size = kernel_size
        ctx.dilation = dilation
        ctx.padding = padding
        ctx.stride = stride

        ctx._backend = type2backend[type(input)]

        output = input.new()

        ctx._backend.Col2Vol_updateOutput(ctx._backend.library_state,
                                          input, output,
                                          output_size[0], output_size[1], output_size[2],
                                          kernel_size[0], kernel_size[1], kernel_size[2],
                                          dilation[0], dilation[1], dilation[2],
                                          padding[0], padding[1], padding[2],
                                          stride[0], stride[1], stride[2])
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):

        grad_input = grad_output.new()

        ctx._backend.Col2Vol_updateGradInput(ctx._backend.library_state,
                                             grad_output,
                                             grad_input,
                                             ctx.kernel_size[0], ctx.kernel_size[1], ctx.kernel_size[2],
                                             ctx.dilation[0], ctx.dilation[1], ctx.dilation[2],
                                             ctx.padding[0], ctx.padding[1], ctx.padding[2],
                                             ctx.stride[0], ctx.stride[1], ctx.stride[2])
        return grad_input, None, None, None, None, None


class Vol2Col(Function):

    @staticmethod
    def forward(ctx, input, kernel_size, dilation, padding, stride):

        assert input.dim() == 5

        ctx.kernel_size = kernel_size
        ctx.dilation = dilation
        ctx.padding = padding
        ctx.stride = stride
        ctx.input_size = (input.size(2), input.size(3), input.size(4))

        ctx._backend = type2backend[type(input)]

        output = input.new()

        ctx._backend.Vol2Col_updateOutput(ctx._backend.library_state,
                                          input, output,
                                          kernel_size[0], kernel_size[1], kernel_size[2],
                                          dilation[0], dilation[1], dilation[2],
                                          padding[0], padding[1], padding[2],
                                          stride[0], stride[1], stride[2])
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):

        grad_input = grad_output.new()

        ctx._backend.Vol2Col_updateGradInput(ctx._backend.library_state,
                                             grad_output,
                                             grad_input,
                                             ctx.input_size[0], ctx.input_size[1], ctx.input_size[2],
                                             ctx.kernel_size[0], ctx.kernel_size[1], ctx.kernel_size[2],
                                             ctx.dilation[0], ctx.dilation[1], ctx.dilation[2],
                                             ctx.padding[0], ctx.padding[1], ctx.padding[2],
                                             ctx.stride[0], ctx.stride[1], ctx.stride[2])
        return grad_input, None, None, None, None
