import math
import torch
import torch.nn.functional as F
from torch._six import inf, nan

device = 'cpu'
dtype = torch.float

x2 = torch.full([1, 1, 3], -inf, device=device, dtype=dtype, requires_grad=True)
res2 = F.adaptive_max_pool1d(x2, 1)
res2.backward(torch.randn_like(res2))
