
import torch
from torch._export import aot_compile, dynamic_dim

torch.manual_seed(1337)

class Net(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.fc = torch.nn.Linear(64, 10)
        self.const_1 = torch.randn((10), device=device)
        self.const_2 = torch.randn((10), device=device)

    def forward(self, x, y):
        const_3 = self.const_1 + self.const_2
        return self.fc(torch.sin(x) + torch.cos(y)) + const_3

data = {}

for device in ["cpu", "cuda"]:
    for use_runtime_constant_folding in [True, False]:
        if device == "cpu" and use_runtime_constant_folding:
            continue
        model = Net(device=device).to(device=device)
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
            model_so_path = aot_compile(
                model,
                (x, y),
                constraints=constraints,
                options={"use_runtime_constant_folding": use_runtime_constant_folding})

        params = dict(model.named_parameters())
        suffix = f"{device}"
        if use_runtime_constant_folding:
            suffix += "_use_runtime_constant_folding"
        data.update({
            f"model_so_path_{suffix}": model_so_path,
            f"inputs_{suffix}": [x, y],
            f"outputs_{suffix}": [ref_output],
            f"fc_weight_{suffix}": params["fc.weight"],
            f"fc_bias_{suffix}": params["fc.bias"],
            f"const_1_{suffix}": model.const_1,
            f"const_2_{suffix}": model.const_2,
        })

# Use this to communicate tensors to the cpp code
class Serializer(torch.nn.Module):
    def __init__(self, data):
        super().__init__()
        for key in data:
            setattr(self, key, data[key])

torch.jit.script(Serializer(data)).save("data.pt")
