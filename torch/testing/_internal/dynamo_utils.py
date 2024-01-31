import torch


# Used for testing an autograd.Function inside torch/, which is worth testing
# because of skipfiles
class SinAutogradFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.sin()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * torch.cos(ctx.saved_tensors[0])
