import torch

autocast = torch.cuda.amp.autocast
custom_fwd = torch.cuda.amp.custom_fwd
custom_bwd = torch.cuda.amp.custom_bwd
