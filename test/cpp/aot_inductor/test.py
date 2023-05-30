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


with torch.no_grad():
    from torch.fx.experimental.proxy_tensor import make_fx
    # Using export is blocked by https://github.com/pytorch/pytorch/issues/99000
    # module, _ = torch._dynamo.export(Net().cuda(), inp)
    module = make_fx(Net().cuda())(x, y)
    lib_path = torch._inductor.aot_compile(module, [x, y])

shutil.copy(lib_path, "libaot_inductor_output.so")
