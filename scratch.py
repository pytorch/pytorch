import torch.jit.te

@torch.jit.te.pointwise_operator
def add(a, b):
    return a + b


add(torch.randn(10), torch.randn(10))
