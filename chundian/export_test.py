import torch
from torch import nn

class MethodProvider:
    def some_method(self, x):
        return x + x

class MyModule(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.bound_method = MethodProvider().some_method
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(4*4, 1),
            nn.ReLU()
        )

    def forward(self, a):
        b = a + 1
        c = b.view(-1)
        c.add_(1)
        logits = self.linear_relu_stack(c)
        # return (logits, logits)
        return (self.bound_method(logits), b)

# torch.export.export(MyModule(), (torch.rand(2, 3),))
input = torch.randn(4, 4)

ep = torch.export.export(MyModule(), (torch.randn(4, 4),))
print(ep)

# from torch._functorch.aot_autograd import aot_export_module
# gm, sig = aot_export_module(MyModule(), [input], trace_joint=False)
# print(gm.print_readable())
