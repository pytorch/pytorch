import torch
import pytorch_openreg

my_tensor = torch.empty((2, 2), device="openreg")

del my_tensor


