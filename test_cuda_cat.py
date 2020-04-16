import torch
a = torch.cuda.FloatTensor([1.])
torch.cat([a, a])
