import torch
from torch._inductor import config

config.fallback_random = True
torch.set_grad_enabled(False)

class Model(torch.nn.Module):
    def forward(self, x):
        return torch.combinations(x.flatten(), r=2)

model = Model()
x = torch.randn(3, 1, 1)
inputs = [x]
device = "cpu"

def run_test(model, inputs, device, backend):
    torch.manual_seed(0)
    model.to(device)
    inputs = [inp.to(device) for inp in inputs]
    if backend != "eager":
        model = torch.compile(model, backend=backend, dynamic=True)
    try:
        out = model(*inputs)
        print(f"succeed on {backend}: {out}")
    except Exception as e:
        print(f"failed on {backend}:", e)

run_test(model, inputs, device, "eager")
run_test(model, inputs, device, "aot_eager")
