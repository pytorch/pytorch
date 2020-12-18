# type: ignore
r'''
**This feature is experimental and its stability is not currently guaranteed. Proceed at your own risk**

FX is a toolkit for capturing and transforming functional PyTorch programs. It
consists of GraphModule and a corresponding intermediate representation (IR). When GraphModule is constructed
with an ``nn.Module`` instance as its argument, GraphModule will trace through the computation of that Module's
``forward`` method symbolically and record those operations in the FX intermediate representation.

.. code-block:: python

    import torch
    import torch.fx

    class MyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.param = torch.nn.Parameter(torch.rand(3, 4))
            self.linear = torch.nn.Linear(4, 5)

        def forward(self, x):
            return torch.topk(torch.sum(self.linear(x + self.linear.weight).relu(), dim=-1), 3)

    m = MyModule()
    gm = torch.fx.symbolic_trace(m)

The Intermediate Representation centers around a 5-opcode format::

    print(gm.graph)

.. code-block:: text

    graph(x):
        %linear_weight : [#users=1] = self.linear.weight
        %add_1 : [#users=1] = call_function[target=<built-in function add>](args = (%x, %linear_weight), kwargs = {})
        %linear_1 : [#users=1] = call_module[target=linear](args = (%add_1,), kwargs = {})
        %relu_1 : [#users=1] = call_method[target=relu](args = (%linear_1,), kwargs = {})
        %sum_1 : [#users=1] = call_function[target=<built-in method sum of type object at 0x7ff2da9dc300>](args = (%relu_1,), kwargs = {dim: -1}) # noqa: B950
        %topk_1 : [#users=1] = call_function[target=<built-in method topk of type object at 0x7ff2da9dc300>](args = (%sum_1, 3), kwargs = {}) # noqa: B950
        return topk_1

The Node semantics are as follows:

- ``placeholder`` represents a function input. The ``name`` attribute specifies the name this value will take on.
  ``target`` is similarly the name of the argument. ``args`` holds either: 1) nothing, or 2) a single argument
  denoting the default parameter of the function input. ``kwargs`` is don't-care. Placeholders correspond to
  the function parameters (e.g. ``x``) in the graph printout.
- ``get_attr`` retrieves a parameter from the module hierarchy. ``name`` is similarly the name the result of the
  fetch is assigned to. ``target`` is the fully-qualified name of the parameter's position in the module hierarchy.
  ``args`` and ``kwargs`` are don't-care
- ``call_function`` applies a free function to some values. ``name`` is similarly the name of the value to assign
  to. ``target`` is the function to be applied. ``args`` and ``kwargs`` represent the arguments to the function,
  following the Python calling convention
- ``call_module`` applies a module in the module hierarchy's ``forward()`` method to given arguments. ``name`` is
  as previous. ``target`` is the fully-qualified name of the module in the module hierarchy to call.
  ``args`` and ``kwargs`` represent the arguments to invoke the module on, *including the self argument*.
- ``call_method`` calls a method on a value. ``name`` is as similar. ``target`` is the string name of the method
  to apply to the ``self`` argument. ``args`` and ``kwargs`` represent the arguments to invoke the module on,
  *including the self argument*
- ``output`` contains the output of the traced function in its ``args[0]`` attribute. This corresponds to the "return" statement
  in the Graph printout.

GraphModule automatically generates Python code for the operations it symbolically observed::

    print(gm.code)

.. code-block:: python

    import torch
    def forward(self, x):
        linear_weight = self.linear.weight
        add_1 = x + linear_weight
        x = linear_weight = None
        linear_1 = self.linear(add_1)
        add_1 = None
        relu_1 = linear_1.relu()
        linear_1 = None
        sum_1 = torch.sum(relu_1, dim = -1)
        relu_1 = None
        topk_1 = torch.topk(sum_1, 3)
        sum_1 = None
        return topk_1
        topk_1 = None

Because this code is valid PyTorch code, the resulting ``GraphModule`` can be used in any context another
``nn.Module`` can be used, including in TorchScript tracing/compilation.
'''

from .graph_module import GraphModule
from .symbolic_trace import symbolic_trace, Tracer
from .graph import Graph
from .node import Node, map_arg
from .proxy import Proxy
