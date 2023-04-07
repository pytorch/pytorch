import torch
from torch import nn
"""
TODO find a better name for this class/file
"""

class CUTLASSLinear(nn.Linear):
    def forward(self, x):
        b = x.view(-1, x.shape[-1]).T
        c = self.bias_tensor.view(torch.numel(self.bias_tensor), 1)
        if self.mask is not None:
            prod, self.meta = torch._cutlass_linear(self.weight_tensor, b, c, self.mask)
            self.mask = None
        else:
            prod, _ = torch._cutlass_linear(self.weight_tensor, b, c, self.meta)
        return prod.T.view(*x.shape[:-1], -1)

    @classmethod
    def from_dense(cls, mod):
        """
        convert from nn.Linear
        """

        cutlass_linear = cls(mod.in_features, mod.out_features)

        m, k = mod.weight.shape
        mask = mod.weight.data != 0

        cutlass_linear.weight_tensor = mod.weight.data.masked_select(mask).view(m, k // 2)
        cutlass_linear.bias_tensor = mod.bias

        cutlass_linear.mask = mask
        cutlass_linear.meta = None

        return cutlass_linear
