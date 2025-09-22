import torch

@torch.compile(options={"max_autotune": True, "max_autotune_gemm_backends": "TRITON"})
def foo(x, y):
    return x @ y

x = torch.randn(1000, 1000, device="cuda")
y = torch.randn(1000, 1000, device="cuda")

foo(x, y)
