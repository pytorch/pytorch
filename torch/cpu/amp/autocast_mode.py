import torch
import functools

autocast = functools.partial(torch.autocast,device_type='cpu')


