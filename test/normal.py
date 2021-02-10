import torch
from torch import nn

class Noise(nn.Module):
    def forward(self, image):
        return image.new_empty(2, 1, 3, 4).normal_()

out = torch.jit.script(Noise())
torch._C._jit_pass_remove_mutation(out.graph)
out(torch.rand([2, 2]))

print(out.graph)
