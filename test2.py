import torch
from torch.autograd import forward_ad as fwAD
import torchvision.models as models

inp = torch.rand(1, 3, 224, 224, requires_grad=True)
resnet18 = models.resnet18(pretrained=True)

with fwAD.dual_level():
    dinp = fwAD.make_dual(inp, torch.rand_like(inp))
    out = resnet18(dinp)
    print("SUCCESS!")
