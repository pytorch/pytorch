import torch

x = [torch.ones(200, 200) for i in range(30)]
torch.save(x, "big_tensor.zip", _use_new_zipfile_serialization=True)
v = torch.load("big_tensor.zip")
