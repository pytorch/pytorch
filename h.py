import torch

GPU_TYPE = "cuda"

@torch.compile
def no_override(x):
    return x.sum(dim=0)

@torch.compile
def override(x):
    return x.sum(dim=0)

x_small = torch.randn(4096, 512, device=GPU_TYPE)
torch._dynamo.decorators.mark_dynamic(x_small, 0)
no_override(x_small)
torch._dynamo.decorators.mark_dynamic(x_small, 0, hint_override=4096 * 1000)
# torch._dynamo.decorators.mark_unbacked(x_small, 0)
override(x_small)
