import torch

torch._dynamo.config.capture_scalar_outputs = True

def true_fn():
    return torch.randn(10)

def false_fn():
    return torch.randn(5)

@torch.compile(backend="eager", fullgraph=True)
def f(x):
    u0, u1 = x.tolist()

    return torch.cond(u0 == 20, true_fn, false_fn, ()) * 2

f(torch.tensor([20, 21]))
