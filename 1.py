import torch
import time
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau, StepLR
import torch.nn as nn
import time
import torchvision
import torch.utils._benchmark as benchmark_utils


device = 'cpu'
a = torch.tensor([1, 2, 3, 4], device=device)
b = torch.tensor([1, 1, 5, 4], device=device)
c = torch.tensor([11, 2, 34, 4], device=device)
d = torch.tensor([1, 12, 5, 4], device=device)

res = torch._foreach_max([a, b], [c, d])
print(res)