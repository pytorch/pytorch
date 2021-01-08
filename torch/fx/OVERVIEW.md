# FX Technical Overview (WIP)

We are building a toolkit for pass writers to facilitate Python-to-Python transformation of nn.Module instances. This toolkit aims to support a subset of Python language semantics—rather than the whole Python language—to facilitate ease of implementation of transforms. The toolkit is available on master in the `torch.fx` namespace. Currently, this feature is unstable, but our goal is to stabilize the toolkit by the end of this year for a prototype release through collaboration with first-party partners.

## Table of Contents

- [FX Technical Overview](#fx-technical-overview)
  - [Table of Contents](#table-of-contents)
- [Introduction](#introduction)
  - [Motivation](#motivation)
  - [Use Cases](#use-cases)
- [Symbolic Tracing](#symbolic-tracing)
  - [About](#about)
  - [Tracer](#tracer)
  - [Proxy](#proxy)
- [The FX IR](#ir)
- [Transformation and Codegen](#codegen)

# Introduction

## Motivation ##

Due to strategic objectives, TorchScript has been optimized for the framework interchange use-case over the transformation and device lowering use-cases. These two use-cases actually represent different customer/partner segments with different requirements. This has been apparent ever since the ONNX days, when most of the disagreements about code representation centered around expressivity v.s. analyzability/transformability. To acknowledge this reality, we’d like to deliver a component separate from (and composable with) TorchScript that focuses more on the transformation and lowering use cases. Ultimately, we’d like to create an ecosystem of separate components that serve different customer segments well.

TorchScript IR resides in the “low level of abstraction, many language features” category. FX is meant to flip us about the origin: let’s build a system that lives in the “high level of abstraction, few language features” quadrant. To provide a higher level of abstraction, we would like to represent code as higher-level blocks—namely the standard `torch.nn` Module blocks—as opposed to TorchScript IR, which represented everything as low-level ATen/primitive operators. To provide a restricted set of language features, we would like to be judicious in which features we support capturing and representing, namely things like control flow, aliasing, complex data structures, hierarchical representation, and the like.

We believe building a component that occupies this point in the space will serve the transformation and lowering customer segments well.

## Use Cases ##

FX should be used by pass writers to provide functionality for capturing and constructing nn.Module code in a structured way. We do not expect end users to utilize FX directly. A useful property of framing FX in this way is that passes can be seen as functions of the form `pass(in_mod : nn.Module) -> nn.Module`. This means we can create composable pipelines of transformations.

![An image of a sample nn.Module transformation pipeline that starts with a Quantize transformation, which is then composed with a Split transformation, then a Lower to Accelerator transformation](https://scontent.xx.fbcdn.net/v/t1.0-9/p720x720/118780346_998065427281509_1321963618820011983_o.png?_nc_cat=104&ccb=2&_nc_sid=32a93c&_nc_ohc=8YbXsAHKRjoAX-Hwj4g&_nc_ht=scontent.xx&_nc_tp=30&uss=d63649d5ad92822b&odm=ZmIud29ya3BsYWNlLmNvbQ&_nc_rmd=260&_nc_log=1&oe2=601C8940&oh=225eeda6b37d2198ffbe409a2276b43d&oe=5FF78064 "nn.Module transformation pipeline")

In this example pipeline, we have a Quantize transformation, which is then composed with a Split transformation, then a Lower to Accelerator transformation. Finally, the transformed Modules are compiled with TorchScript for deployment. This last point emphasizes that not only should FX transforms be composable with each other, but their products are composable with other systems like TorchScript compilation or tracing.

# Symbolic Tracing

## About ##

FX’s front-end makes use of the dynamic nature of Python to intercept call-sites for various entities (PyTorch operators, Module invocations, and Tensor method invocations). This functionality is exposed through an API called `torch.fx.symbolic_trace`.  We can see how this works by way of an example:

```python
import torch

class MyModule(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.param = torch.nn.Parameter(
        torch.rand(3, 4))
    self.linear = torch.nn.Linear(4, 5)

  def forward(self, x):
    return self.linear(x + self.param).clamp(min=0.0, max=1.0)

from torch.fx import symbolic_trace
module = MyModule()
symbolic_traced : torch.fx.GraphModule = symbolic_trace(module)

input = torch.rand(3, 4)
torch.testing.assert_allclose(symbolic_traced(input), module(input))
```

Here, we set up a simple Module that exercises different language features: fetching a parameter, applying an arithmetic operator, applying a submodule (linear), and applying a Tensor method. `symbolic_trace` returns an instance of GraphModule, which is in itself a subclass of `nn.Module`. We can see that the `symbolic_traced` instance runs and returns the same result as the original module instance module.

## Tracer ##

`Tracer` is the class that implements the symbolic tracing functionality of `torch.fx.symbolic_trace`. A call to `symbolic_trace(m)` is equivalent to `Tracer().trace(m)`. Tracer can be subclassed to override various behaviors of the tracing process. The different behaviors that can be overridden are described in the docstrings of the methods on the class.

In the default implementation of `Tracer().trace`, the tracer first creates Proxy objects for all arguments in the `forward` function. (This happens in the call to `create_args_for_root`.) Next, the `forward` function is called with the new Proxy arguments. As the Proxies flow through the program, they record all the operations (`torch` function calls, method calls, and operators) that they touch into the growing FX Graph as Nodes.

## Proxy ##

Proxy objects are Node wrappers used by the Tracer to record operations seen during symbolic tracing. If you're doing graph transforms, you can wrap your own Proxy method around a raw Node so that you can use the overloaded operators to add additional things to a Graph.

The mechanism through which Proxy objects record computation is [`__torch_function__`](https://pytorch.org/docs/stable/notes/extending.html). If any custom Python type defines a method named `__torch_function__`, PyTorch will invoke that `__torch_function__` implementation when an instance of that custom type is passed to a function in the `torch` namespace. In the Proxy implementation of `__torch_function__`, we simply create a new Proxy object with the passed-in operator. Creating a new Proxy means that a new Node--the Node wrapped by the Proxy--will be added to the Graph as well. When this new Proxy object is used later in the code, it “transforms” its user into another Proxy object and the process is repeated.

Consider the following example:

```python
  class M(torch.nn.Module):
      def forward(self, x):
          return torch.relu(x)

  m = M()
  traced = symbolic_trace(m)
```

First, the parameter `x` is transformed into a Proxy object and the corresponding Node (a Node with op = “placeholder” and target = “x”) is added to the Graph. The next operation is `torch.relu`, which takes `x` as a parameter. Because `x` is a Proxy, `__torch_function__` is called on `torch.relu`. This, in turn, transforms `torch.relu` into a Proxy object and adds a `torch.relu` Node to the Graph.

# The FX IR

Symbolic tracing captures an intermediate representation (IR), which is represented as a doubly-linked list of Nodes.

Node is the data structure that represents individual operations within a Graph. For the most part, Nodes represent callsites to various entities, such as operators, methods, and Modules (some exceptions include Nodes that specify function inputs and outputs). Each Node has a function specified by its `op` property. The Node semantics for each value of `op` are as follows:

- `placeholder` represents a function input. The `name` attribute specifies the name this value will take on. `target` is similarly the name of the argument. `args` holds either: 1) nothing, or 2) a single argument denoting the default parameter of the function input. `kwargs` is don't-care. Placeholders correspond to the function parameters (e.g. `x`) in the graph printout.
- `get_attr` retrieves a parameter from the module hierarchy. `name` is similarly the name the result of the fetch is assigned to. `target` is the fully-qualified name of the parameter's position in the module hierarchy. `args` and `kwargs` are don't-care
- `call_function` applies a free function to some values. `name` is similarly the name of the value to assign to. `target` is the function to be applied. `args` and `kwargs` represent the arguments to the function, following the Python calling convention
- `call_module` applies a module in the module hierarchy's `forward()` method to given arguments. `name` is as previous. `target` is the fully-qualified name of the module in the module hierarchy to call. `args` and `kwargs` represent the arguments to invoke the module on, *including the self argument*.
- `call_method` calls a method on a value. `name` is as similar. `target` is the string name of the method to apply to the `self` argument. `args` and `kwargs` represent the arguments to invoke the module on, *including the self argument*
- `output` contains the output of the traced function in its `args[0]` attribute. This corresponds to the "return" statement in the Graph printout.

Each Node also has a reference to the Nodes that it takes as input (`input_nodes`) and the Nodes that use it (`users`). Although Nodes are represented as a doubly-linked list, the use-def relationships form an acyclic graph and can be traversed as such.

# Transformation and Codegen

An invocation of `symbolic_traced` above requires a valid `forward()` method to be defined on the Module instance. How does this work? GraphModule actually generates valid Python source code based on the IR it is instantiated with. This can be seen by accessing the code attribute on the GraphModule: `print(symbolic_traced.code)`.

This outputs:

```python
def forward(self, x):
    param = self.param
    add_1 = x + param
    linear_1 = self.linear(add_1)
    clamp_1 = linear_1.clamp(min = 0.0, max = 1.0)
    return clamp_1
```

This is the core of why FX is a Python-to-Python translation toolkit. Outside users can treat the results of FX transformations as they would any other `nn.Module` instance.
