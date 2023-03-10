import torch
from torch import nn
"""
TODO fin a better name for this class/file

"""

class cuSPARSELtLinear(nn.Linear):

    def forward(self, x):
        return self.cslt.masked_mm(x.mT).mT

    @classmethod
    def from_dense(cls, mod):
        """
        convert from nn.Linear
        """

        print("Converting:")

        print(mod.weight.data.shape)
        print(mod.bias.data.shape)

        cusparselt = cls(mod.in_features,
                         mod.out_features)

        # must convert to half
        cusparselt.weight.data = mod.weight.data.half()
        cusparselt.bias.data = mod.bias.data.half()

        # set up cusparselt
        cslt = torch.classes.cusparselt.CusparseLtLinear()
        cslt.init(cusparselt.weight.data, cusparselt.bias.data)
        cslt.prune()
        cslt.compress()

        cusparselt.cslt = cslt
        return cusparselt
