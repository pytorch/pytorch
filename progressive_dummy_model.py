import torch

def foo(x):
    x = x @ x
    torch._dynamo.graph_break()
    x = x @ x
    torch._dynamo.graph_break()
    x = x @ x
    torch._dynamo.graph_break()
    x = x @ x
    torch._dynamo.graph_break()
    x = x @ x
    torch._dynamo.graph_break()
    x = x @ x
    torch._dynamo.graph_break()
    return x

foo_c = torch.compile(foo)
foo_c(torch.randn(10, 10)).sum().backward()
foo_c(torch.randn(20, 20)).sum().backward()
