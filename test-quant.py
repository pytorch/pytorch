import torch

x = torch.randn(1, 3, 2, 2)
qx = torch.quantize_per_tensor(x, 0.1, 13, torch.qint8)
