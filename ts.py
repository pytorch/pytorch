import torch

@torch.compile()
def f(x, y):
    return x + y

torch.compiler.set_stance('force_eager')
for i in range(20):
    print("iter", i)
    if i == 5:
        torch.compiler.set_stance('default')
    f(torch.randn(3000, device='cuda'), torch.randn(3000, device='cuda'))
