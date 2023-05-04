import torch
import torch._inductor
from torch._inductor import config
config.optimize_for_inference = True

class ConvBN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, bias=False, dtype=torch.half, **kwargs)
        self.bn = torch.nn.BatchNorm2d(out_channels, eps=0.001, dtype=torch.float)

    def forward(self, x):
        return self.bn(self.conv(x))

mod_eager = ConvBN(3, 32, kernel_size=3, stride=2).cuda().eval()

x = torch.rand((3, 3, 32, 32), dtype=torch.half).cuda()

@torch.compile()
def foo(mod, inp):
    return mod(inp)

print("Mod eager id", id(mod_eager))

with torch.no_grad():
    with torch.cuda.amp.autocast():
        out = foo(mod_eager, x)

        out = foo(mod_eager, x)

breakpoint()
print(mod_eager)
