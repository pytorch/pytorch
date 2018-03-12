import torch
from torch.autograd import Function, Variable
from torch.autograd.function import once_differentiable


class MarginRankingLoss(Function):
    @staticmethod
    def forward(ctx, input1, input2, y, margin, size_average):
        ctx.margin = margin
        ctx.size_average = size_average
        _output = input1.clone()
        _output.add_(-1, input2)
        _output.mul_(-1).mul_(y)
        _output.add_(ctx.margin)
        _output.clamp_(min=0)
        output = _output.sum()

        if ctx.size_average:
            output = output / y.size(0)

        ctx.save_for_backward(input1, input2, y)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input1, input2, y = ctx.saved_variables
        grad_input1 = Variable(input1.data.new(input1.size()).zero_())
        grad_input2 = Variable(input1.data.new(input1.size()).zero_())

        dist = ((input1 - input2).mul_(-1) * y).add_(ctx.margin)
        mask = dist.ge(0)

        grad_input1.masked_fill_(mask, 1)
        grad_input1 = grad_input1.mul_(-1) * y
        grad_input2.masked_fill_(mask, 1) * y
        grad_input2 = grad_input2 * y

        if ctx.size_average:
            grad_input1.div_(y.size(0))
            grad_input2.div_(y.size(0))

        return grad_input1 * grad_output, grad_input2 * grad_output, None, None, None
