import torch
from common_utils import enable_profiling_mode

def foo(a):
    b = a + 1
    c = a * b
    return c

with enable_profiling_mode():
    ja = torch.jit.script(foo)
    # profile once to get shapes
    ja(torch.ones(1))
    g = ja._profiled_graph
    for n in g.nodes():
        for o in n.outputs():

            # tensor with a fully specified shape info
            # these are defined in python_ir.cpp
            # or can be inspected with dir()
            if o.type().kind() == "TensorType":
                print("type=", o.type().scalarType(), o.type().sizes())
    print(str(g))
