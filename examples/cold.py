import torch


@torch.compile(backend="eager")
def fn(x, y, z):
    for _ in range(100):
        # x = torch.nn.functional.silu(x)
        x = torch.addmm(x, y, z)
    return x
    # return torch.sin(torch.cos(x))


x = torch.randn(20, 20)
y = torch.randn(20, 20)
z = torch.randn(20, 20)
fn(x, y, z)
