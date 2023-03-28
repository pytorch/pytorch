import torch
import torch._dynamo


@torch._dynamo.optimize("eager")
def f(x, inf):
    return x + inf


print(f(torch.randn(2), 3))
print(f(torch.randn(2), 3))
