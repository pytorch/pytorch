import torch

class SubModule(torch.nn.Module):
    def __init__(self):
        super(SubModule, self).__init__()

    def forward(self, x, y: int):
        return x if y <= 2 else x + 2

m = torch.jit.script(SubModule())

torch._C._jit_trace_module(m._c, (torch.rand(2, 2), 3))

print(m.graph)