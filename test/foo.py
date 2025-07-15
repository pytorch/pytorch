import torch

a=torch.tensor(1.0,requires_grad=True)
b=torch.tensor(2.0,requires_grad=True)
print(torch.stack([a,b]))
print(torch.as_tensor([a,b]))
