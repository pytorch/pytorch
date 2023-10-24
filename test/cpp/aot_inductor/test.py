
import torch
from torch._export import aot_compile, dynamic_dim

torch.manual_seed(1337)

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(64, 10)

    def forward(self, x, y):
        return self.fc(torch.sin(x) + torch.cos(y))

data = {}

for device in ["cpu", "cuda"]:
    model = Net().to(device=device)
    x = torch.randn((32, 64), device=device)
    y = torch.randn((32, 64), device=device)
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

    data.update({
        f"model_so_path_{device}": model_so_path,
        f"inputs_{device}": [x, y],
        f"outputs_{device}": [ref_output],
    })

# Use this to communicate tensors to the cpp code
class Serializer(torch.nn.Module):
    def __init__(self, data):
        super().__init__()
        for key in data:
            setattr(self, key, data[key])

torch.jit.script(Serializer(data)).save("data.pt")
