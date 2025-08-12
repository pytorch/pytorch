import torch

x = torch.randn(2, 3)

# torch says xz is not changed 
# compile says xz is changed
def func(x):
    xz = x[:]
    # Issue is instead of graph breaking here and now, we delay...
    x.unsqueeze_(1)
    xz += 1
    return xz

ref = func(x)
gm = torch.compile(func, backend='eager')
test = gm(x)

assert torch.equal(ref, test), f"Error comparing {ref} and {test}"