import torch

A = torch.ones(128, 128).cuda().to(torch.int8)
B = torch.ones(128, 128).cuda().to(torch.int8).t()

alpha = torch.ones(128).cuda()

A_sparse = torch._cslt_compress(A)

print(A)
print(B)
print(alpha)

def func():
    return torch._cslt_sparse_mm(A_sparse, B, alpha=alpha)

print("EAGER RES")
print(func())
print(func())

func = torch.compile(func, mode="max-autotune")

print("RES")
print(func())
