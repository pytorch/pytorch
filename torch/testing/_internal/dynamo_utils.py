import torch

# Used for testing an autograd.Function inside torch/, which is worth testing
# because of skipfiles
class SinAutogradFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.detach().sin()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.cos()
