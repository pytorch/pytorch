import torch

x = torch.ones(2, 2, 2)

torch.save(x, "out.zip")

print(torch.load("out.zip"))