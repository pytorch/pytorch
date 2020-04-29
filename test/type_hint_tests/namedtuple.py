import torch


t = torch.tensor([[3.0, 1.5], [2.0, 1.5]])

t_sort = t.sort()
t_sort[0][0, 0] == 1.5
t_sort.indices[0, 0] == 1
t_sort.values[0, 0] == 1.5

t_qr = torch.qr(t)
t_qr[0].shape == [2, 2]
t_qr.Q.shape == [2, 2]
