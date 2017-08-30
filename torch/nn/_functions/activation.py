from torch.autograd.function import Function


class Softsign(Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        buffer = input.clone().abs_().add_(1)
        return input.div(buffer)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_variables
        buffer = input.abs().add_(1)
        return grad_output.div(buffer.mul(buffer))
