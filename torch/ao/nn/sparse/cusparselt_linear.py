import torch
from torch import nn

class cuSPARSELtLinear(nn.Module):
    def forward(self, x):
        return self.cslt.masked_mm(x)

    @classmethod
    def from_dense(cls, mod, dtype=None):
        """
        convert from nn.Linear
        """
        cusparselt = cls()
        # no need to clone data ptr since we prune and compress when initializing, so we can discard
        # but bias needs to be copied so other bias can be GC
        cusparselt.bias = torch.clone(mod.bias.data.to(torch.float16))
        num_bytes = mod.weight.data.nelement() * mod.weight.data.element_size()
        compressed_size = num_bytes * 9 // 16 
        cusparselt.weight = torch.empty(
            (compressed_size // mod.weight.data.element_size(),),
            dtype=torch.float16,
            device=cusparselt.bias.device,
        )

        # print(temp.nelement()*temp.element_size())
        # set up cusparselt
        cusparselt.cslt = torch.classes.cusparselt.CusparseLtLinear(
            cusparselt.weight, cusparselt.bias
        )
        cusparselt.cslt.set_compressed(mod.weight.data.T.contiguous())

        return cusparselt

class cuSPARSELtLinearInt8(nn.Module):
    def forward(self, x):
        return self.cslt.masked_mm(x)

    @classmethod
    def from_dense(cls, mod, dtype=None):
        """
        convert from nn.Linear
        """
        cusparselt = cls()
        # no need to clone data ptr since we prune and compress when initializing, so we can discard
        # but bias needs to be copied so other bias can be GC
        cusparselt.bias = torch.clone(mod.bias.data.to(torch.int8))

        int8_weight = mod.weight.data.to(torch.int8)

        num_bytes = int8_weight.nelement() * int8_weight.element_size()
        compressed_size = num_bytes // 16 * 10
        cusparselt.weight = torch.empty(
            (compressed_size // int8_weight.element_size(),),
            dtype=torch.int8,
            device=cusparselt.bias.device,
        )

        # print(temp.nelement()*temp.element_size())
        # set up cusparselt
        cusparselt.cslt = torch.classes.cusparselt.CusparseLtLinear(
            cusparselt.weight, cusparselt.bias
        )
        cusparselt.cslt.set_compressed(int8_weight.T.contiguous())

        return cusparselt
