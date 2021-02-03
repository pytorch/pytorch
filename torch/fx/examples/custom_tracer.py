import torch
from torch.fx import symbolic_trace, Tracer, Graph, GraphModule, Node
from typing import Any, Callable, Dict, Optional, Tuple, Union

"""
How to Create and Use Custom Tracers

`Tracer`--the class that implements the symbolic tracing functionality
of `torch.fx.symbolic_trace`--can be subclassed to override various
behaviors of the tracing process. In this tutorial, we'll demonstrate
how to customize the symbolic tracing process using some handwritten
Tracers. Each example will show that, by simply overriding a few methods
in the `Tracer` class, you can completely alter the Graph produced by
symbolic tracing. For a complete description of the methods that
can be changed, refer to the docstrings of the methods in the Tracer
class.

If you want a real-world example of a custom tracer, check out FX's AST
Rewriter in `rewriter.py`. `RewritingTracer` inherits from Tracer but
overrides the `trace` function so that we can rewrite all calls to
`assert` to the more FX-friendly `torch.assert`.

Note that a call to `symbolic_trace(m)` is equivalent to
`Tracer().trace(m).graph`. (`Tracer` is the default implementation of
Tracer as defined in `symbolic_trace.py`.)
"""





"""
Custom Tracer #1: Trace Through All `torch.nn.ReLU` Submodules

During symbolic tracing, some submodules are traced through and their
constituent ops are recorded; other submodules appear as an
atomic "call_module" Node in the IR. A module in this latter category
is called a "leaf module". By default, all modules in the PyTorch
standard library (`torch.nn`) are leaf modules. We can change this
by creating a custom Tracer and overriding `is_leaf_module`. In this
case, we'll keep the default behavior for all `torch.nn` Modules except
for `ReLU`.
"""

class M1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(x)

"""
Tracing with the default tracer produces:

    opcode       name    target    args       kwargs
    -----------  ------  --------  ---------  --------
    placeholder  x       x         ()         {}
    call_module  relu_1  relu      (x,)       {}
    output       output  output    (relu_1,)  {}

"""
default_traced: GraphModule = symbolic_trace(M1())

class LowerReluTracer(Tracer):
    def is_leaf_module(self, m : torch.nn.Module, qualname : str):
        if isinstance(m, torch.nn.ReLU):
            return False
        return super().is_leaf_module(m, qualname)

"""
Tracing with our custom tracer produces:

    opcode         name    target                             args       kwargs
    -------------  ------  ---------------------------------  ---------  ------------------
    placeholder    x       x                                  ()         {}
    call_function  relu_1  <function relu at 0x7f66f7170b80>  (x,)       {'inplace': False}
    output         output  output                             (relu_1,)  {}
"""
lower_relu_tracer = LowerReluTracer()
custom_traced_graph: Graph = lower_relu_tracer.trace(M1())





"""
Custom Tracer #2: Add an Extra Attribute to Each Node

Here, we'll override `create_node` so that we can add a new attribute to
each Node during its creation
"""

class M2(torch.nn.Module):
    def forward(self, a, b):
        return a + b

class TaggingTracer(Tracer):
    def create_node(self, kind : str, target : Union[str, Callable],
                    args : Tuple[Any], kwargs : Dict[str, Any], name : Optional[str] = None,
                    type_expr : Optional[Any] = None) -> Node:
        n = super().create_node(kind, target, args, kwargs, name)
        n.tag = "foo"
        return n

custom_traced_graph: Graph = TaggingTracer().trace(M2())

def assert_all_nodes_have_tags(g: Graph) -> bool:
    for n in g.nodes:
        if not hasattr(n, "tag") or not n.tag == "foo":
            return False
    return True

# Prints "True"
print(assert_all_nodes_have_tags(custom_traced_graph))
