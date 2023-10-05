import shutil

import torch
from torch._export import aot_compile, dynamic_dim

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(64, 10)

    def forward(self, x, y):
        return self.fc(torch.sin(x) + torch.cos(y))


x = torch.randn((32, 64), device="cuda")
y = torch.randn((32, 64), device="cuda")
with torch.no_grad():
    ref_output = (Net().cuda())(x, y)

torch._dynamo.reset()
with torch.no_grad():
    constraints = [
        dynamic_dim(x, 0) >= 1,
        dynamic_dim(x, 0) <= 1024,
        dynamic_dim(x, 0) == dynamic_dim(y, 0),
    ]
    lib_path, module = aot_compile(Net().cuda(), (x, y), constraints=constraints)

shutil.copy(lib_path, "libmodel.so")


# Use this to
class IOTensors(torch.nn.Module):
    def __init__(self, tensors):
        super().__init__()
        for key in tensors:
            setattr(self, key, tensors[key])

io_tensors = {
    "x": x,
    "y": y,
    "output": [ref_output],
}

# Save arbitrary values supported by TorchScript
# https://pytorch.org/docs/master/jit.html#supported-type
tensor_saver = torch.jit.script(IOTensors(io_tensors))
tensor_saver.save("io_tensors.pt")
