import torch

@torch.compile(backend="eager")
def hello():
     with torch.inference_mode():
        a = torch.rand(5, requires_grad=True)
hello()
