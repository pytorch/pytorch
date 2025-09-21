import torch

a = torch.randn(10, 2, 3, dtype=torch.float16)
b = torch.randn(10, 3, 4, dtype=torch.float32)

c = torch.bmm(a, b)  # result is float32
print(c.dtype)  # torch.float32
