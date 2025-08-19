import torch

with torch._subclasses.FakeTensorMode() as fake_mode:
    for _ in range(100000):
        x = torch.randn(1024)
        torch.sin(x)
