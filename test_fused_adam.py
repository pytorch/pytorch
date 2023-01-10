import torch

from torch.optim import adam

params = [
    # torch.rand(2, 3, dtype=torch.float64, device='cuda:0', requires_grad=True),
    # torch.rand(2, 3, dtype=torch.float32, device='cuda:0', requires_grad=True),
    # torch.rand(2, 3, dtype=torch.float16, device='cuda:0', requires_grad=True),
    torch.rand(2, 3, dtype=torch.bfloat16, device='cuda:0', requires_grad=True),
    torch.rand(2, 3, dtype=torch.float64, device='cuda:1', requires_grad=True),
    # torch.rand(2, 3, dtype=torch.float32, device='cuda:1', requires_grad=True),
    # torch.rand(2, 3, dtype=torch.float16, device='cuda:1', requires_grad=True),
    # torch.rand(2, 3, dtype=torch.bfloat16, device='cuda:1', requires_grad=True),
]

for p in params:
    p.grad = torch.rand(2, 3, dtype=p.dtype, device=p.device)

o = adam.Adam(params, fused=True)
o.step()