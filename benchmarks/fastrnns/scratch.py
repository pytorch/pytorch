import torch


@torch.jit.script
def fn(x, scale, shift):
    return scale * x / shift


@torch.jit.script
def recurrent(x, scale, shift):
    y = x
    for i in range(100):
        y = fn(y, scale, shift)
    return y


x = torch.randn(2, 2, device="cuda")
scale = torch.randn(2, 2, device="cuda", requires_grad=True)
shift = torch.randn(2, 2, device="cuda", requires_grad=True)
inputs = [x, scale, shift]


out = recurrent(x, scale, shift)
recurrent.graph_for(x, scale, shift)


import torch


@torch.jit.script
def recurrent_scaleshift(x, scale, shift):
    y = x
    for i in range(64):
        y = scale * y + shift
    return y


x = torch.randn(2, 2, device="cuda")
scale = torch.randn(2, 2, device="cuda", requires_grad=True)
shift = torch.randn(2, 2, device="cuda", requires_grad=True)
inputs = [x, scale, shift]
out = recurrent_scaleshift(x, scale, shift)
recurrent_scaleshift.graph_for(x, scale, shift)


import torch

x = torch.tensor([])
x.requires_grad = True
x.mean().backward()  # no error triggered
x = x.cuda()
x.mean().backward()
