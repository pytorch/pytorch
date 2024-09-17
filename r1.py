import torch


torch._dynamo.config.assume_static_by_default = False
torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch._dynamo.config.specialize_float = False

# def nested(x, repeats):
#     rank = torch.arange(repeats.numel(), device="cpu")

#     index = rank.repeat_interleave(repeats, dim=0)
#     return torch.index_select(x, index=index, dim=0)

# example_inputs = (
#     torch.randn((32, 64), device="cpu"),
#     repeats := torch.tensor([5, 10, 15], device="cpu"),
# )
# torch._dynamo.mark_dynamic(repeats, 0)  # create backed symint

# nested_opt = torch._dynamo.optimize("inductor")(nested)

# expect = nested(*example_inputs)
# actual = nested_opt(*example_inputs)
# print(expect == actual, expect, actual)

# class GraphModule(torch.nn.Module):
#     def forward(self, s0: "Sym(s0)", L_x_: "i64[s0][1]cpu"):
#         l_x_ = L_x_

#          # File: /data/users/bobren/pytorch/repro.py:27 in fn, code: return x + 1.0 / x.size(0)
#         truediv: "Sym(FloatTrueDiv(1.0, ToFloat(s0)))" = 1.0 / s0;  s0 = None
#         add: "f32[s0][1]cpu" = l_x_ + truediv;  l_x_ = truediv = None
#         return (add,)


def fn(x):
    return x + 1.0 / x.size(0)


x = torch.arange(10)
print(torch._dynamo.optimize("inductor")(fn)(x))
