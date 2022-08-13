from torch.fx.experimental.proxy_tensor import make_fx
import torch
from functorch.experimental.cond import cond
"""
Test case #1: basic
# """
print("EXAMPLE 1")

def true_fn(x):
    return x.sin()

def false_fn(x):
    return x.cos()

x = torch.randn(4)
result = cond(False, true_fn, false_fn, x)
assert torch.allclose(result, torch.cos(x))

"""
Test case #2: tracing
"""
print("EXAMPLE 2")

def f(x, y):
    return cond(y, true_fn, false_fn, x)

graph = make_fx(f)(x, torch.tensor(False))
result_true = graph.forward(x, torch.tensor(True))
result_false = graph.forward(x, torch.tensor(False))
assert not torch.allclose(result_true, result_false)
assert torch.allclose(result_true, torch.sin(x))
assert torch.allclose(result_false, torch.cos(x))

"""
Test case #3: tracing complex/nested

I've hardcoded the logic into ProxyTensor.
"""
print("EXAMPLE 3")

def true_nested(y):
    return y * y

def false_nested(y):
    return y + y

def true_fn(x, pred2):
    return cond(pred2, true_nested, false_nested, x)

def false_fn(x, _):
    return x.cos()

def f(x, pred, pred2):
    return cond(pred, true_fn, false_fn, (x, pred2))

graph = make_fx(f)(x, torch.tensor(False), torch.tensor(False))

result_true_true = graph.forward(x, torch.tensor(True), torch.tensor(True)) # True + True -> x * x
result_true_false = graph.forward(x, torch.tensor(True), torch.tensor(False)) # True + True -> x + x
result_false_true = graph.forward(x, torch.tensor(False), torch.tensor(True)) #  False + either -> cos
result_false_false = graph.forward(x, torch.tensor(False), torch.tensor(False)) #  False + either -> cos


assert not torch.allclose(result_true_true, result_true_false)
assert not torch.allclose(result_false_true, result_true_true)

assert torch.allclose(result_false_true, result_false_false)

assert torch.allclose(result_true_true, x * x)
assert torch.allclose(result_true_false, x + x)

assert torch.allclose(result_false_true, torch.cos(x))

print("Done, all tests passed")

"""
More test cases (coming soon)

3. Autograd
4. functorch transforms!
"""
