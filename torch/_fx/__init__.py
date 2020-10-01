# type: ignore
r'''
**This feature is experimental and its stability is not currently guaranteed. Proceed at your own risk**

FX (Functional Transformations) is a toolkit for capturing and transforming functional PyTorch programs. It
consists of GraphModule and a corresponding intermediate representation (IR). When GraphModule is constructed
with an `nn.Module` instance as its argument, GraphModule will trace through the computation of that Module's
`forward` method symbolically and record those operations in the FX intermediate representation.

```
import torch
from torch._fx import GraphModule

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(3, 4))
        self.linear = torch.nn.Linear(4, 5)

    def forward(self, x):
        return torch.topk(torch.sum(self.linear(x + self.linear.weight).relu(), dim=-1), 3)

m = MyModule()
gm = symbolic_trace(m)
```

The Intermediate Representation centers around a 5-opcode format:

```
from tabulate import tabulate
node_specs = [[n.op, n.name, n.target, n.args, n.kwargs] for n in gm.graph.nodes]
print(tabulate(node_specs, headers=['opcode', 'name', 'target', 'args', 'kwargs']))
```

```
opcode         name           target                                                   args                kwargs
-------------  -------------  -------------------------------------------------------  ------------------  -----------
placeholder    x              x                                                        ()                  {}
get_attr       linear_weight  linear.weight                                            ()                  {}
call_function  add_1          <built-in function add>                                  (x, linear_weight)  {}
call_module    linear_1       linear                                                   (add_1,)            {}
call_method    relu_2         relu                                                     [linear_1]          {}
call_function  sum_1          <built-in method sum of type object at 0x7f1c29dd36e0>   (relu_2,)           {'dim': -1}
call_function  topk_1         <built-in method topk of type object at 0x7f1c29dd36e0>  (sum_1, 3)          {}
```

The semantics are as follows:

- `placeholder` represents a function input. The `name` attribute specifies the name this value will take on.
  `target` is similarly the name of the argument. `args` and `kwargs` are don't-care
- `get_attr` retrieves a parameter from the module hierarchy. `name` is similarly the name the result of the
   fetch is assigned to. `target` is the fully-qualified name of the parameter's position in the module hierarchy.
   `args` and `kwargs` are don't-care
- `call_function` applies a free function to some values. `name` is similarly the name of the value to assign
  to. `target` is the function to be applied. `args` and `kwargs` represent the arguments to the function,
  following the Python calling convention
- `call_module` applies a module in the module hierarchy's `forward()` method to given arguments. `name` is
  as previous. `target` is the fully-qualified name of the module in the module hierarchy to call.
  `args` and `kwargs` represent the arguments to invoke the module on, _including the self argument_.
- `call_method` calls a method on a value. `name` is as similar. `target` is the string name of the method
  to apply to the `self` argument. `args` and `kwargs` represent the arguments to invoke the module on,
  _including the self argument_.

GraphModule automatically generates Python code for the operations it symbolically observed:

```
print(gm.code)
```

```
def forward(self, x):
    self = self.root
    linear_weight = self.linear.weight
    add_1 = x + linear_weight
    linear_1 = self.linear(add_1)
    relu_2 = linear_1.relu()
    sum_1 = torch.sum(relu_2, dim = -1)
    topk_1 = torch.topk(sum_1, 3)


    return topk_1
```

Because this code is valid PyTorch code, the resulting `GraphModule` can be used in any context another
`nn.Module` can be used, including in TorchScript tracing/compilation.
'''

from .graph_module import GraphModule
from .symbolic_trace import symbolic_trace, Tracer
from .graph import Graph, map_arg
from .node import Node
from .proxy import Proxy
