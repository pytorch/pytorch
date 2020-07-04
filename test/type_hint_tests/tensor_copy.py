import torch


t = torch.randn(2, 3)
u = torch.randn(2, 3)
t.copy_(u)
(t == u).all()
