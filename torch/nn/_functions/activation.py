from torch.autograd.function import Function


class Softsign(Function):

    def forward(self, input):
        self.buffer = input.clone().abs_().add_(1)
        self.buffer_squared = False
        output = input.clone().div_(self.buffer)
        return output

    def backward(self, grad_output):
        if not self.buffer_squared:
            self.buffer.mul_(self.buffer)
            self.buffer_squared = True
        grad_input = grad_output.clone().div_(self.buffer)
        return grad_input
