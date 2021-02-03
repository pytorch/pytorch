# type: ignore
r'''
**This feature is under a Beta release and its API may change.**

FX is a toolkit for developers to use to transform ``nn.Module``
instances. FX consists of three main components: a **symbolic tracer,**
an **intermediate representation**, and **Python code generation**. A
demonstration of these components in action:

::

    import torch
    # Simple module for demonstration
    class MyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.param = torch.nn.Parameter(torch.rand(3, 4))
            self.linear = torch.nn.Linear(4, 5)

    def forward(self, x):
        return self.linear(x + self.param).clamp(min=0.0, max=1.0)

    module = MyModule()

    from torch.fx import symbolic_trace
    # Symbolic tracing frontend - captures the semantics of the module
    symbolic_traced : torch.fx.GraphModule = symbolic_trace(module)

    # High-level intermediate representation (IR) - Graph representation
    print(symbolic_traced.graph)
    """
    graph(x):
        %param : [#users=1] = self.param
        %add_1 : [#users=1] = call_function[target=<built-in function add>](args = (%x, %param), kwargs = {})
        %linear_1 : [#users=1] = call_module[target=linear](args = (%add_1,), kwargs = {})
        %clamp_1 : [#users=1] = call_method[target=clamp](args = (%linear_1,), kwargs = {min: 0.0, max: 1.0})
        return clamp_1
    """

    # Code generation - valid Python code
    print(symbolic_traced.code)
    """
    def forward(self, x):
        param = self.param
        add_1 = x + param;  x = param = None
        linear_1 = self.linear(add_1);  add_1 = None
        clamp_1 = linear_1.clamp(min = 0.0, max = 1.0);  linear_1 = None
        return clamp_1
    """

The **symbolic tracer** performs “abstract interpretation” of the Python
code. It feeds fake values, called Proxies, through the code. Operations
on theses Proxies are recorded. More information about symbolic tracing
can be found in the
`symbolic\_trace <https://pytorch.org/docs/master/fx.html#torch.fx.symbolic_trace>`__
and `Tracer <https://pytorch.org/docs/master/fx.html#torch.fx.Tracer>`__
documentation.

The **intermediate representation** is the container for the operations
that were recorded during symbolic tracing. It consists of a list of
Nodes that represent function inputs, callsites (to functions, methods,
or ``nn.Module`` instances), and return values. More information about
the IR can be found in the documentation for
`Graph <https://pytorch.org/docs/master/fx.html#torch.fx.Graph>`__. The
IR is the format on which transformations are applied.

**Python code generation** is what makes FX a Python-to-Python (or
Module-to-Module) transformation toolkit. For each Graph IR, we can
create valid Python code matching the Graph’s semantics. This
functionality is wrapped up in
`GraphModule <https://pytorch.org/docs/master/fx.html#torch.fx.GraphModule>`__,
which is an ``nn.Module`` instance that holds a ``Graph`` as well as a
``forward`` method generated from the Graph.

Taken together, this pipeline of components (symbolic tracing →
intermediate representation → transforms → Python code generation)
constitutes the Python-to-Python transformation pipeline of FX. In
addition, these components can be used separately. For example,
symbolic tracing can be used in isolation to capture a form of
the code for analysis (and not transformation) purposes. Code
generation can be used for programmatically generating models, for
example from a config file. There are many uses for FX!
'''

from .graph_module import GraphModule
from .symbolic_trace import symbolic_trace, Tracer, wrap
from .graph import Graph
from .node import Node, map_arg
from .proxy import Proxy
from .interpreter import Interpreter as Interpreter, Transformer as Transformer
