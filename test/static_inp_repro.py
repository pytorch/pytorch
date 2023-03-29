import torch
from torch.multiprocessing.reductions import StorageWeakRef
import torchvision

m = torchvision.models.resnet50()
m = m.cuda()

inp = torch.rand([1, 3, 255, 255]).cuda().requires_grad_(True)


@torch.compile()
def comp(m, inp):
    return m(inp).sum().backward()

m = torch.nn.Sequential(m.conv1, m.bn1) #, m.maxpool)
comp(m, inp)