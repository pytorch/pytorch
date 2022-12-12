import torch
import nvfuser_extension  # noqa: F401

t = torch.randn((5, 5), device='cuda')
expected = torch.sinh(t)
output = torch.ops.myop.sinh_nvfuser(t)

print("Expected:", expected)
print("Output:", output)

assert torch.allclose(output, expected)
print("They match!")
