import shutil

import torch
import torch._dynamo
import torch._inductor


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.ones((32, 64), device="cuda")

    def forward(self, x):
        x = torch.relu(x + self.weight)
        return x


inp = torch.randn((32, 64), device="cuda")

module, _ = torch._dynamo.export(Net().cuda(), inp)
lib_path = torch._inductor.aot_compile(module, [inp])
shutil.copy(lib_path, "aot_inductor_output.so")
