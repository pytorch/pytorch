import torch


torch._dynamo.config.assume_static_by_default = False
torch._dynamo.config.capture_dynamic_output_shape_ops = True
# torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.specialize_float = False


def blah(x):
    return x + 2


@torch.compile(backend="inductor", fullgraph=True)
def fn(x, y):
    return x + blah(y)


x = torch.randn(3)
print(x)
print(fn(x, 3.0))
print(fn(x, 4.0))
print(fn(x, 12.0))
