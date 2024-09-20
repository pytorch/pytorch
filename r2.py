import torch


torch._dynamo.config.assume_static_by_default = False
torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch._dynamo.config.specialize_float = False


@torch.compile(fullgraph=True)
def fn(x, y):
    if y == 5.0:
        return x + 2
    else:
        return x + y


x = torch.randn(3)
print(fn(x, 3.0))
print(fn(x, 4.0))
