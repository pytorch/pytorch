import torch


torch._dynamo.config.assume_static_by_default = False

# Don't graph break on dynamic output shapes
torch._dynamo.config.capture_dynamic_output_shape_ops = True

# Don't graph break on .item() calls
# torch._dynamo.config.capture_scalar_outputs = True


torch._dynamo.config.specialize_float = False


def fn(x, y):
    return x + 1, y * 2


x = torch.arange(10)

fn_opt = torch._dynamo.optimize("inductor")(fn)
# fn_opt = torch._dynamo.optimize("aot_eager")(fn)
# fn_opt = torch._dynamo.optimize("eager")(fn)

print(fn_opt(x, 3.0))
print(fn(x, 3.0))
