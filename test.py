import torch


a = torch.zeros(3, 3)
b = torch.ones(3)

torch.compile(torch.ops.aten.diagonal_scatter, backend="aot_eager")(a, b)
print(f"yay, test passed I guess?")
