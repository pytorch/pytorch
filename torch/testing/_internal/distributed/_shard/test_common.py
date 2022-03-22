import torch.nn as nn


class SimpleMegatronLM(nn.Module):
    def __init__(self, linear_size, rank=None):
        super().__init__()
        self.fc1 = nn.Linear(*linear_size[0])
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(*linear_size[1])
        if rank:
            self.fc1.cuda(rank)
            self.fc2.cuda(rank)

    def forward(self, inp):
        return self.fc2(self.gelu(self.fc1(inp)))
