import torch

class M(torch.nn.Module):
    def forward(self, x, y):
        assert x.shape == y.shape
        #return x + y

m = M()
gm = torch.fx.symbolic_trace(m)
print(gm)
