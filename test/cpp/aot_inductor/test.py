import shutil

import torch
from torch._export import aot_compile, dynamic_dim


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(64, 10)
        weights = torch.arange(640)
        weights = torch.reshape(weights, (10, 64))

        with torch.no_grad():
            self.fc.weight.copy_(weights)
            self.fc.bias.copy_(torch.zeros(10))

    def forward(self, x, y):
        return self.fc(torch.sin(x) + torch.cos(y))


x = torch.randn((32, 64), device="cuda")
y = torch.randn((32, 64), device="cuda")

torch._dynamo.reset()

with torch.no_grad():
    constraints = [
        dynamic_dim(x, 0) >= 1,
        dynamic_dim(x, 0) <= 1024,
        dynamic_dim(x, 0) == dynamic_dim(y, 0),
    ]
    lib_path, module = aot_compile(Net().cuda(), (x, y), constraints=constraints)

shutil.copy(lib_path, "libaot_inductor_output.so")
