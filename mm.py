import torch

a = torch.randn(32, 16384, device="cuda", dtype=torch.float16)
b = torch.randn(16384, 32, device="cuda", dtype=torch.float16)
f = lambda a, b: (a @ b).relu() + 1.0

compiled_res = torch.compile(f)(a, b)
torch.testing.assert_close(compiled_res, (a @ b).relu() + 1.0, atol=1e-2, rtol=1e-2)