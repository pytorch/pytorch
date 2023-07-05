import shutil

import torch
import torch._dynamo
import torch._inductor
from torch._export import export


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(64, 10)

    def forward(self, x, y):
        return self.fc(torch.sin(x) + torch.cos(y))


x = torch.randn((32, 64), device="cuda")
y = torch.randn((32, 64), device="cuda")

with torch.no_grad():
    # FIXME: re-enable dynamic shape after we add dynamic shape support to the
    # AOTInductor runtime
    # ep = export(
    #     Net().cuda(),
    #     (x, y),
    #     constraints=[dynamic_dim(x, 0) == dynamic_dim(y, 0)],
    # )
    ep = export(Net().cuda(), (x, y))
    args = (*tuple(ep.state_dict.values()), x, y)
    lib_path = torch._inductor.aot_compile(ep.graph_module, args)

shutil.copy(lib_path, "libaot_inductor_output.so")
