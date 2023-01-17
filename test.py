import torch

import logging

torch.set_float32_matmul_precision("high")
torch._dynamo.config.log_level = logging.DEBUG
torch._inductor.config.log_level = logging.DEBUG
torch._functorch.log_level=logging.DEBUG
torch._functorch.debug_graphs = True

# def custom_backend(gm: torch.fx.GraphModule, example_inputs):
#     print("custom backend called with FX graph:")
#     gm.graph.print_tabular()
#     return gm.forward

# def bar(a, b):
#     x = a / (torch.abs(a) + 1)
#     if b.sum() < 0:
#         b = b * -1
#     return x * b

# opt_bar = dynamo.optimize(custom_backend)(bar)
# inp1 = torch.randn(10)
# inp2 = torch.randn(10)
# opt_bar(inp1, inp2)
# opt_bar(inp1, -inp2)

def foo(x, y, z):
    a = torch.cos(x)
    b = torch.sin(y)
    c = torch.nested.nested_tensor([a, b])
    d = torch.tanh(c)
    e = torch.nested.to_padded_tensor(d, padding=0)
    f = torch.sin(z)
    return e, f
# explanation, out_guards, graphs, ops_per_graph, break_reasons, explanation_verbose = dynamo.explain(
#     foo, torch.randn(10), torch.randn(10), torch.randn(10)
# )

# print(explanation_verbose)
# for i, graph in enumerate(graphs):
#     print(f"Printing graph {i}\n", graph.graph)
#     print(f"Printing code {i}\n",graph.code)
#     print("Printing tabular graph\n")
#     graph.graph.print_tabular()

# def custom_backend(gm: torch.fx.GraphModule, example_inputs):
#     print(gm.graph)
#     print(gm.code)
#     gm.graph.print_tabular()
#     return gm.forward


a, b, c = torch.randn(10, device='cuda'), torch.randn(10, device='cuda'), torch.randn(10, device='cuda')
print(a)
print(b)
print(c)
out = foo(a, b, c)
opt_nt_tanh_relu = torch.compile(foo)
result = opt_nt_tanh_relu(a, b, c)

print(result)

assert(torch.allclose(r, o) for r, o in zip(out, result))
