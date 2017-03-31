import torch
from torch.autograd import Function


class Bilinear(Function):

    def forward(self, input1, input2, weight, bias=None):
        self.save_for_backward(input1, input2, weight, bias)

        output = input1.new()

        buff2 = input1.new()
        buff2.resize_as_(input2)

        # compute output scores:
        output.resize_(input1.size(0), weight.size(0))
        for k, w in enumerate(weight):
            torch.mm(input1, w, out=buff2)
            buff2.mul_(input2)
            torch.sum(buff2, 1, out=output.narrow(1, k, 1))

        if bias is not None:
            output.add_(bias.expand_as(output))

        return output

    def backward(self, grad_output):
        input1, input2, weight, bias = self.saved_tensors
        grad_input1 = grad_input2 = grad_weight = grad_bias = None

        buff1 = input1.new()
        buff1.resize_as_(input1)
        buff2 = input1.new()
        buff2.resize_as_(input2)

        if self.needs_input_grad[0] or self.needs_input_grad[1]:
            grad_input1 = torch.mm(input2, weight[0].t())
            grad_input1.mul_(grad_output.narrow(1, 0, 1).expand(grad_input1.size()))
            grad_input2 = torch.mm(input1, weight[0])
            grad_input2.mul_(grad_output.narrow(1, 0, 1).expand(grad_input2.size()))

            for k in range(1, weight.size(0)):
                torch.mm(input2, weight[k].t(), out=buff1)
                buff1.mul_(grad_output.narrow(1, k, 1).expand(grad_input1.size()))
                grad_input1.add_(buff1)

                torch.mm(input1, weight[k], out=buff2)
                buff2.mul_(grad_output.narrow(1, k, 1).expand(grad_input2.size()))
                grad_input2.add_(buff2)

        if self.needs_input_grad[2]:
            # accumulate parameter gradients:
            for k in range(weight.size(0)):
                torch.mul(input1, grad_output.narrow(1, k, 1).expand_as(input1), out=buff1)
            grad_weight = torch.mm(buff1.t(), input2)

        if bias is not None and self.needs_input_grad[3]:
            grad_bias = grad_output.sum(0)

        return grad_input1, grad_input2, grad_weight, grad_bias
