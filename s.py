import torch

@torch.compile(dynamic=True)
def f(x):
    return x * x.size()[0]

x = torch.randn(10)
torch._dynamo.mark_dynamic(x, 0)

f(x)
f(torch.randn(20))
f(torch.randn(30))
f(torch.randn(40))
