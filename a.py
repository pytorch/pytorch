import torch
import torch.nn as nn
import inspect

torch.set_default_device("cuda")


class Model(nn.Module):
    def __init__(self, features: int):
        super().__init__()
        self.W = nn.Parameter(torch.zeros(features, features, dtype=torch.bfloat16))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + x.matmul(self.W)


if __name__ == "__main__":
    torch.manual_seed(0)
    features = 4
    batch = 3
    model = Model(features)

    x = torch.randn(batch, features, requires_grad=True, dtype=torch.bfloat16)

    torch._dynamo.reset()
    y1 = torch.compile(
        model,
        mode='reduce-overhead',
        pythonify="/tmp/model.py",
    )(x)
    loss = y1.mean()
    loss.backward()

    print("Compiled output:", y1)
    print("Gradient on x:", x.grad)

    with open("/tmp/model.py") as f:
        code = f.read()

    print("\n--- Generated code contains CUDA graph patterns? ---")
    cuda_patterns = ["CUDAGraph", "graph", "capture", "replay"]
    for pattern in cuda_patterns:
        if pattern in code:
            print(f"  Found: {pattern}")

    print("\n--- Executing pythonified code ---")
    frame = inspect.currentframe()
    exec(
        code,
        frame.f_globals,
        frame.f_locals,
    )
    y2 = frame.f_locals["y"]
    print("Pythonified output:", y2)

    if torch.allclose(y1, y2):
        print("\nSUCCESS: y1 and y2 match!")
    else:
        print(f"\nFAILURE: y1={y1}, y2={y2}")
