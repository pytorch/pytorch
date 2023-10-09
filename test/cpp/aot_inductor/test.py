
import torch
from torch._export import aot_compile, dynamic_dim

torch.manual_seed(1337)

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(64, 10)

    def forward(self, x, y):
        return self.fc(torch.sin(x) + torch.cos(y))

model = Net().to(device="cuda")
x = torch.randn((32, 64), device="cuda")
y = torch.randn((32, 64), device="cuda")
with torch.no_grad():
    ref_output = model(x, y)

torch._dynamo.reset()
with torch.no_grad():
    constraints = [
        dynamic_dim(x, 0) >= 1,
        dynamic_dim(x, 0) <= 1024,
        dynamic_dim(x, 0) == dynamic_dim(y, 0),
    ]
    model_so_path, _ = aot_compile(model, (x, y), constraints=constraints)

# Use this to communicate tensors to the cpp code
class Serializer(torch.nn.Module):
    def __init__(self, data):
        super().__init__()
        for key in data:
            setattr(self, key, data[key])

data = {
    "model_so_path": model_so_path,
    "inputs": [x, y],
    "outputs": [ref_output],
}

torch.jit.script(Serializer(data)).save("data.pt")
