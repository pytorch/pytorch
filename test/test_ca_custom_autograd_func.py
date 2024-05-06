import torch
from torch._dynamo import compiled_autograd

class CommOpGradientScaling(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, input_tensor, scale_gradient_factor
    ) -> torch.Tensor:
        ctx.scale_gradient_factor = scale_gradient_factor
        return input_tensor

    @staticmethod
    def backward(
        ctx, grad_output
    ):
        grad_output.mul_(ctx.scale_gradient_factor)
        return grad_output, None


class Net(torch.nn.Module):
    def __init__(self, checkpoint=False):
        super().__init__()
        self.fc1 = torch.nn.Linear(4, 4)

    def forward(self, x):
        x = CommOpGradientScaling.apply(x, 0.5)
        return self.fc1(x)


model = Net()
# model = torch.compile(Net())
input = torch.randn([1, 4], requires_grad=True)

def compiler_fn(gm):
    return torch.compile(gm, backend="aot_eager", fullgraph=True)

with compiled_autograd.enable(compiler_fn):
    loss = model(input).sum()
    loss.backward()
