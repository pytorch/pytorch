import torch
from torch import nn
"""
TODO find a better name for this class/file

"""

class cuSPARSELtLinear(nn.Linear):

    # def __init__(self):
    #     super().__init__(

    def forward(self, x):
        return self.cslt.masked_mm(x.mT).mT

    @classmethod
    def from_dense(cls, mod):
        """
        convert from nn.Linear
        """
        cusparselt = cls(mod.in_features, mod.out_features)
        # no need to clone data ptr since we prune and compress when initializing, so we can discard
        # but bias needs to be copied so other bias can be GC
        cusparselt.bias.data = torch.clone(mod.bias.data)
        num_bytes = mod.weight.data.nelement() * mod.weight.data.element_size()
        # print("num byte = ", num_bytes)
        compressed_size = num_bytes // 16 * 9
        # print("compresseds", compressed_size)
        cusparselt.weight.data = torch.empty((compressed_size // 2, ), 
                                        dtype=cusparselt.bias.dtype,
                                        device=cusparselt.bias.device)

        # print(temp.nelement()*temp.element_size())
        # set up cusparselt
        cusparselt.cslt = torch.classes.cusparselt.CusparseLtLinear(cusparselt.weight,
                                                                    cusparselt.bias)
        cusparselt.cslt.set_compressed(mod.weight.data)

        return cusparselt
