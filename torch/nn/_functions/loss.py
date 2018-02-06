import torch
from torch.autograd import Function, Variable
from torch.autograd.function import once_differentiable


class CosineEmbeddingLoss(Function):
    @staticmethod
    def forward(ctx, input1, input2, y, margin, size_average):
        ctx.margin = margin
        ctx.size_average = size_average
        ctx.w1 = input1.new()
        ctx.w22 = input1.new()
        ctx.w = input1.new()
        ctx.w32 = input1.new()
        ctx._outputs = input1.new()

        _idx = input1.new().byte()

        buffer = torch.mul(input1, input2)
        torch.sum(buffer, 1, out=ctx.w1, keepdim=True)

        epsilon = 1e-12
        torch.mul(input1, input1, out=buffer)
        torch.sum(buffer, 1, out=ctx.w22, keepdim=True).add_(epsilon)

        ctx._outputs.resize_as_(ctx.w22).fill_(1)
        torch.div(ctx._outputs, ctx.w22, out=ctx.w22)
        ctx.w.resize_as_(ctx.w22).copy_(ctx.w22)

        torch.mul(input2, input2, out=buffer)
        torch.sum(buffer, 1, out=ctx.w32, keepdim=True).add_(epsilon)
        torch.div(ctx._outputs, ctx.w32, out=ctx.w32)
        ctx.w.mul_(ctx.w32)
        ctx.w.sqrt_()

        torch.mul(ctx.w1, ctx.w, out=ctx._outputs)
        ctx._outputs = ctx._outputs.select(1, 0)

        torch.eq(y, -1, out=_idx)
        ctx._outputs[_idx] = ctx._outputs[_idx].add_(-ctx.margin).clamp_(min=0)
        torch.eq(y, 1, out=_idx)
        ctx._outputs[_idx] = ctx._outputs[_idx].mul_(-1).add_(1)

        output = ctx._outputs.sum()

        if ctx.size_average:
            output = output / y.size(0)

        ctx.save_for_backward(input1, input2, y)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        v1, v2, y = ctx.saved_tensors

        buffer = v1.new()
        _idx = v1.new().byte()

        gw1 = grad_output.new()
        gw2 = grad_output.new()
        gw1.resize_as_(v1).copy_(v2)
        gw2.resize_as_(v1).copy_(v1)

        torch.mul(ctx.w1, ctx.w22, out=buffer)
        gw1.addcmul_(-1, buffer.expand_as(v1), v1)
        gw1.mul_(ctx.w.expand_as(v1))

        torch.mul(ctx.w1, ctx.w32, out=buffer)
        gw2.addcmul_(-1, buffer.expand_as(v1), v2)
        gw2.mul_(ctx.w.expand_as(v1))

        torch.le(ctx._outputs, 0, out=_idx)
        _idx = _idx.view(-1, 1).expand(gw1.size())
        gw1[_idx] = 0
        gw2[_idx] = 0

        torch.eq(y, 1, out=_idx)
        _idx = _idx.view(-1, 1).expand(gw2.size())
        gw1[_idx] = gw1[_idx].mul_(-1)
        gw2[_idx] = gw2[_idx].mul_(-1)

        if ctx.size_average:
            gw1.div_(y.size(0))
            gw2.div_(y.size(0))

        grad_output_val = grad_output[0]
        if grad_output_val != 1:
            gw1.mul_(grad_output_val)
            gw2.mul_(grad_output_val)

        return gw1, gw2, None, None, None


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
