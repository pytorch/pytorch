import torch
from typing import Any

#class M(torch.nn.Module):
#    def forward(self) -> Any:
#        return 'out' if self.training else {}

class M(torch.nn.Module):
    def forward(self) -> Any:
        if self.training:
            return 'out'
        else:
            return {}

m = torch.jit.script(M())
