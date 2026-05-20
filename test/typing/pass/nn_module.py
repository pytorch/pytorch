# flake8: noqa
import torch


module = torch.nn.Module()
module.to(torch.device("cpu"), dtype=torch.float64)
module.to(torch.float64)
module.to(torch.empty(1))
module.to(memory_format=torch.channels_last)
