import torch

torch._dynamo.config.assume_static_by_default = False
torch._dynamo.config.capture_dynamic_output_shape_ops = True
# torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.specialize_float = False

def blah(x):
    return x + torch.randn(3)

@torch.compile(backend="inductor", fullgraph=True)
def fn(x):
    return x + blah(x)

x = torch.randn(3)
print(x)
print(fn(x))
print(fn(x))
print(fn(x))
