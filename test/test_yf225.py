import torch
import torch._dynamo
import torch.export

torch._dynamo.config.capture_scalar_outputs = True

@torch.compile(backend="inductor", fullgraph=True)
def forward(x):
    a, b = x.tolist()
    torch.export.constrain_as_size(a)
    torch.export.constrain_as_size(b)
    return torch.cat([torch.randn(a), torch.randn(b)])

forward(torch.tensor([4, 5]))
