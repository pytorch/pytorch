import torch
import numpy as np

a = np.array([True, False])
b = np.array([True, True, False])
print(b > a)

z = torch.tensor([True, False], dtype=torch.bool)
y = torch.tensor([False, True], dtype=torch.bool)

c = z < y
print(c)
