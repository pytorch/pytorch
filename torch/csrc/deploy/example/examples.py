import torch

class Simple(torch.nn.Module):
    def __init__(self, N, M):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.rand(N, M))

    def forward(self, input):
        output = self.weight + input
        return output
