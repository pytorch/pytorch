import torch

torch.utils._crash_handler.enable_minidumps()

torch.bincount(input=torch.tensor([9223372036854775807]))
