import torch
inputs = [torch.rand(1,1) for _ in range(10)]
bounds = torch.tensor([0.1, 0.2, 0.4, 0.6, 0.8, 1.], device='cuda')
slopes = torch.tensor([0., 0., 0., 0., 0.], device='cuda')
intercepts = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9], device='cuda')

for i in range(10):
    inp = inputs[i]
    o = torch.ops.fb.piecewise_linear(inp.to("cuda"), bounds, slopes, intercepts)
    print(f"input: {inp} | output: {o}")
