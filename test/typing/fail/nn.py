import torch


module = torch.nn.Conv2d(1, 1, 1)

module.to(memory_format=torch.channels_last)
module.to(torch.channels_last)  # E: No overload variant of "to"
