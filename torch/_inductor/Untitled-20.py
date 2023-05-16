
import torch


@torch.compile
def foo(x):
    # GRAPH 1 
    y = x * x * x
    # graph break triggered here 
    if y.sum() > 0:
        # GRAPH 2 
        z = y ** y
    else:
        # GRAPH 3
        z = (y.abs() ** y.abs())
    torch._dynamo.graph_break()

    return z * torch.rand_like(z)

# Running Graph 1, 2, and 4 
foo(torch.arange(0, 10, device="cuda"))
# Replaying Graph 1, 2, and 4
foo(torch.arange(0, 10, device="cuda"))



@torch.compile
def foo(x):
    # GRAPH 1 
    y = x * x * x
    # graph break triggered here 
    if y.sum() > 0:
        # GRAPH 2 
        z = y ** y
    else:
        # GRAPH 3
        z = (y.abs() ** y.abs())
    torch._dynamo.graph_break()

    return z * torch.rand_like(z)

...

# x * x * 3, graph executed as part of first tape, but then during execution 
# diverges from its recording, as we hit the abs() path
foo(torch.arange(-10, 0), device="cuda")