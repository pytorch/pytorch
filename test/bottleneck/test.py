import torch
from torch.autograd import Variable

x = Variable(torch.ones(3, 3), requires_grad=True)
(3 * x).sum().backward()
