import torch


torch._dynamo.config.assume_static_by_default = False
torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.specialize_float = False


def nested(x, repeats):
    rank = torch.arange(repeats.numel(), device="cpu")

    index = rank.repeat_interleave(repeats, dim=0)
    return torch.index_select(x, index=index, dim=0)


example_inputs = (
    torch.randn((32, 64), device="cpu"),
    repeats := torch.tensor([5, 10, 15], device="cpu"),
)
torch._dynamo.mark_dynamic(repeats, 0)  # create backed symint

nested_opt = torch._dynamo.optimize("inductor")(nested)
# nested_opt = torch._dynamo.optimize("aot_eager")(nested)
# nested_opt = torch._dynamo.optimize("eager")(nested)

expect = nested(*example_inputs)
actual = nested_opt(*example_inputs)
print(expect == actual, expect, actual)
