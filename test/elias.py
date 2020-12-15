import torch

@torch.jit.script
def foo(x):
    return x + x + x

torch._C._jit_override_can_fuse_on_cpu(True)

foo(torch.rand([2], requires_grad=False))
foo(torch.rand([2], requires_grad=False))
foo(torch.rand([2], requires_grad=False))
print(torch.jit.last_executed_optimized_graph())
