import torch


# m = torch.nn.AdaptiveAvgPool2d((5,7))
# input = torch.randn(1, 64, 8, 9)
# output = m(input)
# print(output.shape)


print("testing AdaptiveAvgPool2D")

x = torch.randn(64, 8, 9)

a1 = torch.nn.AdaptiveAvgPool2d((5, 7))
a2 = torch.nn.AdaptiveAvgPool2d(7)
a3 = torch.nn.AdaptiveAvgPool2d((None, 7))

for a in [a1, a2, a3]:
    out = a(x)
    print(out.shape)

# y = torch.randn(1, 8, 4, 8)
# b1 = torch.nn.AdaptiveAvgPool2d((5, 7))
# b2 = torch.nn.AdaptiveAvgPool2d((5, None))
# b3 = torch.nn.AdaptiveAvgPool2d((None, None))

# for b in [b1, b2, b3]:
#     out = b(y)
#     print(out.shape)

# ----------------------

# ((1, 8, 8, 8), (5, 7)),
#         ((2, 8, 8, 8), (None, 7)),
#         ((1, 8, 4, 3), (5, None)),
#         ((1, 8, 4, 3), (None, None)),
#         ((1, 8, 4, 3), (5)),

# out = a(x)
# print(out.shape)

# out = a(x)
# print(out.shape)

# ----------------------


@torch.jit.script
def bar(x):
    return torch.nn.functional.adaptive_avg_pool2d(x, (5, 7))

inputs = list(bar.graph.inputs())
torch._C._jit_pass_inline(bar.graph)


def prop_shapes(inp):
    inputs[0].setType(inputs[0].type().with_sizes(inp))
    torch._C._jit_pass_propagate_shapes_on_graph(bar.graph)

prop_shapes([1, 64, 8, 9])
print("shape..", next(bar.graph.outputs()).type().symbolic_sizes())
print(bar.graph)


# ----------------------
