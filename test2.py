import time
import torch
import torch._dynamo as dynamo
import logging
from torch._dynamo import config
import torch._inductor.config as ind_config
import logging

ind_config.debug=True
ind_config.size_asserts=False

torch.set_float32_matmul_precision("high")
# torch._dynamo.config.log_level = logging.DEBUG
torch._inductor.config.log_level = logging.DEBUG
torch._functorch.log_level=logging.DEBUG
torch._functorch.debug_graphs = True

device = 'cuda'
requires_grad = False


def foo(x, y, z):
    a = torch.cos(x)
    a1 = torch.sin(a)
    b = torch.sin(y)
    b1 = torch.cos(b)
    c = torch.nested.as_nested_tensor([a1, b1])
    c1 = torch.nested.as_nested_tensor([a, b])
    c2 = torch.add(c, c1)
    d = torch.tanh(c2)
    e = torch.relu(d)
    f = torch.nested.to_padded_tensor(e, padding=0)
    g = torch.abs(z)
    h = torch.sin(f)
    i = torch.cos(h)
    return g, i


# explanation, out_guards, graphs, ops_per_graph, break_reasons, explanation_verbose = dynamo.explain(
#     foo, torch.randn(10, requires_grad=True), torch.randn(10, requires_grad=True), torch.randn(10, requires_grad=True)
# )
# print(explanation_verbose)
# for i, graph in enumerate(graphs):
#     print(f"Printing graph {i}\n", graph.graph)
#     print(f"Printing code {i}\n",graph.code)
#     print("Printing tabular graph\n")
#     graph.graph.print_tabular()


a, b, c = torch.randn((2, 3), device=device, requires_grad=requires_grad), torch.randn((5, 3), device=device, requires_grad=requires_grad), torch.randn((3, 3), device=device, requires_grad=requires_grad)
out1, out2 = foo(a, b, c)

opt_foo = torch.compile(foo)
g, i = opt_foo(a, b, c)
assert(torch.allclose(out1, g))
assert(torch.allclose(out2, i))


if requires_grad:
    g.sum().backward()
    i.sum().backward()
