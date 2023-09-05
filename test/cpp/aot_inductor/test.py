import shutil

import torch
import torch._export


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(64, 10)

    def forward(self, x, y):
        return self.fc(torch.sin(x) + torch.cos(y))


x = torch.randn((32, 64), device="cuda")
y = torch.randn((32, 64), device="cuda")

# FIXME: re-enable dynamic shape after we add dynamic shape support to the
# AOTInductor runtime
with torch._dynamo.config.patch(dynamic_shapes=False):
    torch._dynamo.reset()

    with torch.no_grad():
        lib_path, module = torch._export.aot_compile(Net().cuda(), (x, y))

shutil.copy(lib_path, "libaot_inductor_output.so")
