import torch
from torch._ops import HigherOrderOperator

class Wrap(HigherOrderOperator):
    def __init__(self):
        super().__init__('wrap')

    def __call__(self, func, *args):
        return func(*args)

wrap = Wrap()

# Case 1: no free variables
@torch.compile(backend='aot_eager')
def f(x):
    return wrap(lambda x: torch.sin(x), x)

x = torch.randn(3, 3)
f(x)

# Case 2: free variables not being tracked.
y = torch.randn(3, 3)

@torch.compile(backend='aot_eager')
def f(x):
    return wrap(lambda x: x + y, x)

x = torch.randn(3, 3)
f(x)

# Case 3: free variables are tracked.
x = torch.randn(3, 3)
y = torch.randn(3, 3)
@torch.compile(backend='aot_eager')
def f(x, y):
    return wrap(lambda x: x + y, x)

f(x, y)

# Case 4: free variables not being tracked. nested case.
# Doesn't work.
# y = torch.randn(3, 3)
# 
# @torch.compile(backend='aot_eager')
# def f(x):
#     return wrap(lambda x: wrap(lambda x: x + y, x), x)
# 
# x = torch.randn(3, 3)
# f(x)

# Case 5: free variables are being tracked. nested case.
# Doesn't work.
# y = torch.randn(3, 3)
# 
# @torch.compile(backend='aot_eager')
# def f(x, y):
#     return wrap(lambda x: wrap(lambda x: x + y, x), x)
# 
# x = torch.randn(3, 3)
# f(x, y)
