import torch
import torch._dynamo

def fn(x):
    torch._dynamo.graph_break()
    return x.sin().sum()

@torch.compile(backend='aot_eager')
def wrapper_fn(x):
    grad_f = torch.func.grad(fn)
    result = grad_f(x)
    return result

x = torch.randn(3)
wrapper_fn(x)
