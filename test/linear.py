import torch
class LinearMod(torch.nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        return torch._C._nn.linear(input, self.weight, self.bias)

print(torch.jit.trace(LinearMod(20, 20), torch.rand([20, 20])).graph)
