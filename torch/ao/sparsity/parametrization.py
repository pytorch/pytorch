from torch import nn

class MulBy(nn.Module):
    def __init__(self, other):
        super().__init__()
        self.other = other

    def forward(self, x):
        assert self.other.shape == x.shape
        return self.other * x
