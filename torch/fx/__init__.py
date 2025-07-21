r'''
FX is a toolkit for developers to use to transform ``nn.Module``
instances. FX consists of three main components: a **symbolic tracer,**
an **intermediate representation**, and **Python code generation**. A
demonstration of these components in action:

::

    import torch


    # Simple module for demonstration
    class MyModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.param = torch.nn.Parameter(torch.rand(3, 4))
            self.linear = torch.nn.Linear(4, 5)

        def forward(self, x):
            return self.linear(x + self.param).clamp(min=0.0, max=1.0)


    module = MyModule()

    from torch.fx import symbolic_trace

    # Symbolic tracing frontend - captures the semantics of the module
    symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)

    # High-level intermediate representation (IR) - Graph representation
    print(symbolic_traced.graph)
    """
    graph():
        %x : [num_users=1] = placeholder[target=x]
        %param : [num_users=1] = get_attr[target=param]
        %add : [num_users=1] = call_function[target=operator.add](args = (%x, %param), kwargs = {})
        %linear : [num_users=1] = call_module[target=linear](args = (%add,), kwargs = {})
        %clamp : [num_users=1] = call_method[target=clamp](args = (%linear,), kwargs = {min: 0.0, max: 1.0})
        return clamp
    """

    # Code generation - valid Python code
    print(symbolic_traced.code)
    """
    def forward(self, x):
        param = self.param
        add = x + param;  x = param = None
        linear = self.linear(add);  add = None
        clamp = linear.clamp(min = 0.0, max = 1.0);  linear = None
        return clamp
    """

The **symbolic tracer** performs "symbolic execution" of the Python
code. It feeds fake values, called Proxies, through the code. Operations
on these Proxies are recorded. More information about symbolic tracing
can be found in the :func:`symbolic_trace` and :class:`Tracer`
documentation.

The **intermediate representation** is the container for the operations
that were recorded during symbolic tracing. It consists of a list of
Nodes that represent function inputs, callsites (to functions, methods,
or :class:`torch.nn.Module` instances), and return values. More information
about the IR can be found in the documentation for :class:`Graph`. The
IR is the format on which transformations are applied.

**Python code generation** is what makes FX a Python-to-Python (or
Module-to-Module) transformation toolkit. For each Graph IR, we can
create valid Python code matching the Graph's semantics. This
functionality is wrapped up in :class:`GraphModule`, which is a
:class:`torch.nn.Module` instance that holds a :class:`Graph` as well as a
``forward`` method generated from the Graph.

Taken together, this pipeline of components (symbolic tracing ->
intermediate representation -> transforms -> Python code generation)
constitutes the Python-to-Python transformation pipeline of FX. In
addition, these components can be used separately. For example,
symbolic tracing can be used in isolation to capture a form of
the code for analysis (and not transformation) purposes. Code
generation can be used for programmatically generating models, for
example from a config file. There are many uses for FX!

Several example transformations can be found at the
`examples <https://github.com/pytorch/examples/tree/master/fx>`__
repository.
'''

from torch.fx import immutable_collections
from torch.fx._symbolic_trace import (  # noqa: F401
    PH,
    ProxyableClassMeta,
    symbolic_trace,
    Tracer,
    wrap,
)
from torch.fx.graph import CodeGen, Graph  # noqa: F401
from torch.fx.graph_module import GraphModule
from torch.fx.interpreter import Interpreter, Transformer
from torch.fx.node import has_side_effect, map_arg, Node
from torch.fx.proxy import Proxy
from torch.fx.subgraph_rewriter import replace_pattern


__all__ = [
    "symbolic_trace",
    "Tracer",
    "wrap",
    "Graph",
    "GraphModule",
    "Interpreter",
    "Transformer",
    "Node",
    "Proxy",
    "replace_pattern",
    "has_side_effect",
    "map_arg",
]
