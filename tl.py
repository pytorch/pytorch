import torch

@torch.compile
def fn(x, y):
    # Necessary runtime assert since we can't guard on unbacked
    torch._check(x.shape[0] < 10)
    if x.shape[0] < 10:
        return x * y

x = torch.randn(5)
y = torch.randn(5)
torch._dynamo.decorators.mark_unbacked(x, 0)
torch._dynamo.decorators.mark_unbacked(y, 0)

fn(x, y)
