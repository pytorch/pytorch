import torch


# torch._dynamo.config.assume_static_by_default = False
# torch._dynamo.config.capture_dynamic_output_shape_ops = True
# torch._dynamo.config.specialize_float = False


def fn(x):
    x_int = x.to(torch.int)
    return x_int + 1.0 / x.size(0)


x = torch.arange(10, dtype=torch.float32)  # Ensure x is a float tensor initially
print(torch._dynamo.optimize("inductor")(fn)(x))
