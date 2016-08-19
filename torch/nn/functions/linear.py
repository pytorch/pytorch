import torch
from torch.autograd import Function


class LinearFunction(Function):

    def forward(self, input, weight, bias=None):
        self.forward_args = (input, weight, bias)
        output = input.new(input.size(0), weight.size(0))
        output.addmm_(0, 1, input, weight.t())
        if bias is not None:
            # cuBLAS doesn't support 0 strides in sger, so we can't use expand
            self.add_buffer = input.new(1).resize_(input.size(0)).fill_(1)
            output.addr_(self.add_buffer, bias)
        return output

    def backward(self, grad_output):
        input, weight, bias = self.forward_args
        grad_tuple = (
            torch.mm(grad_output, weight) if \
                self.needs_input_grad[0] else None,
            torch.mm(grad_output.t(), input) if \
                self.needs_input_grad[1] else None,
            torch.mv(grad_output.t(), self.add_buffer) if \
                bias is not None and self.needs_input_grad[2] else None,
        )
        return grad_tuple

