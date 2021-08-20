import torch

from typing import List


def foo(x, y):
    return x * y

def bar(x):
    # f = torch.nn.AdaptiveAvgPool2d((2,2))
    # return f(x)
    return torch.nn.functional.adaptive_avg_pool2d(x, (2,2))

# f = torch.jit.script(foo)
# print(f.graph)

f = torch.jit.script(bar)
print(f.graph)
print("---------")

inp = torch.rand(1, 64, 10, 9)
# out = (2, 2)
t = torch.jit.trace(bar, inp)
# print(t.shape)

print("---------njiji---")
print("torch.nn.functional.adaptive_avg_pool2d")
print(torch.nn.functional.adaptive_avg_pool2d(inp, (5,7)).shape)

m = torch.nn.AdaptiveAvgPool2d((5,7))
# input = torch.randn(1, 64, 8, 9)
output = m(inp)
print("AdaptiveAvgPool2d")
print(output.shape)

# print(torch.nn.functional.adaptive_avg_pool2d((1, 64, 10, 9), 7).shape)

print(torch.nn.functional.adaptive_avg_pool2d((1, 64, 10, 9), (None, 7)).shape)

# print(torch.nn.functional.adaptive_avg_pool2d((1, 64, 8, 9), (5,7)).shape)


print(t.graph)
