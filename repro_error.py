import torch

from torch._export.converter import TS2EPConverter


class M(torch.nn.Module):
    def forward(self, x):
        return x.view(x.shape[1] * 2, x.size(0), 2)

import torch

data = (torch.randn(5, 2, 4, requires_grad=False),)
ts_fn = torch.jit.script(M())
ep = TS2EPConverter(ts_fn, data).convert()
ep.module()(*data)

