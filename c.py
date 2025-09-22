import torch
@torch.compile
def foo(a):
    sym_int = a.size(0)
    return sym_int * 2
a = torch.randn(5)
torch._dynamo.decorators.mark_dynamic(a, 0)
foo(a)
