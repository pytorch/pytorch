import torch

@torch.compile()
def f(a, b, c):
    res = torch.sum((a @ b) + 1.0) + torch.sum(b @ c) + torch.sum(c @ a)
    
    return res

a = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
b = torch.randn(1024, 2048, device="cuda", dtype=torch.bfloat16)
c = torch.randn(2048, 1024, device="cuda", dtype=torch.bfloat16)

f(a, b, c)

# print(f"Result from {"cuda"} is {f(a, b, c)}")