import torch


class MyMod(torch.nn.Module):
    def __init__(self, buf):
        self.buf = buf

    def forward(self, x):
        return torch.mul(x, self.buf)


m = MyMod(torch.ones(4))
m_compiled = torch.compile(m)
out = m_compiled(torch.ones(4, requires_grad=True))
print(out)
