import shutil

import torch
import torch._dynamo
import torch._inductor


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
        module, _ = torch._dynamo.export(Net().cuda())(x, y)
        lib_path = torch._inductor.aot_compile(module, [x, y])

shutil.copy(lib_path, "libaot_inductor_output.so")
