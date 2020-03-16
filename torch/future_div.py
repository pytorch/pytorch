import torch

torch.div = torch.true_divide
torch.Tensor.div = torch.Tensor.true_divide
torch.Tensor.div_ = torch.Tensor.true_divide_
torch.Tensor.__truediv__ = torch.Tensor.true_divide