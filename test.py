import torch

@torch.compile
def foo(a, b, c):
    torch.addmm(
        a,
        b,
        c,
        alpha=1,
        beta=1
    )

a = torch.empty((384, 256), dtype=torch.bfloat16, device='cuda')
b = torch.empty((384, 128), dtype=torch.bfloat16, device='cuda')
c = torch.empty((256,), dtype=torch.bfloat16, device='cuda')

foo(a, b, c)
