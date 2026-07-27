import torch


t = torch.randn(2, 3)
reveal_type(t)  # E: {Tensor}
u = torch.randn(2, 3)
reveal_type(u)  # E: {Tensor}
t.copy_(u)
reveal_type(t)  # E: {Tensor}
r = (t == u).all()
reveal_type(r)  # E: {Tensor}
