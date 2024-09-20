import torch


torch._dynamo.config.assume_static_by_default = False
torch._dynamo.config.capture_dynamic_output_shape_ops = True
# torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.specialize_float = False


def fn(x):
    return x + 1.0 / x.size(0)


x = torch.arange(10)

fn_opt = torch._dynamo.optimize("inductor")(fn)
# fn_opt = torch._dynamo.optimize("aot_eager")(fn)
# fn_opt = torch._dynamo.optimize("eager")(fn)

print(fn(x).dtype)
print(fn_opt(x).dtype)
