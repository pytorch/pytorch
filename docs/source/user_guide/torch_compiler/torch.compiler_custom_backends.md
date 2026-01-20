# Custom Backends

## Overview

`torch.compile` provides a straightforward method to enable users
to define custom backends.

A backend function has the contract
`(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]) -> Callable`.

Backend functions can be called by TorchDynamo, the graph tracing component of `torch.compile`,
after tracing an FX graph and are
expected to return a compiled function that is equivalent to the traced FX graph.
The returned callable should have the same contract as the `forward` function of the original `torch.fx.GraphModule`
passed into the backend:
`(*args: torch.Tensor) -> List[torch.Tensor]`.

In order for TorchDynamo to call your backend, pass your backend function as the `backend` kwarg in
`torch.compile`. For example,

```python
import torch

def my_custom_backend(gm, example_inputs):
    return gm.forward

def f(...):
    ...

f_opt = torch.compile(f, backend=my_custom_backend)

@torch.compile(backend=my_custom_backend)
def g(...):
    ...
```

See below for more examples.

## Registering Custom Backends

You can register your backend using the `register_backend` decorator, for example,

```python
from torch._dynamo import register_backend

@register_backend
def my_compiler(gm, example_inputs):
    ...
```

Besides the `register_backend` decorator, if your backend is in another python package, you could also register your
backend through entry points of python package, which provides a way for a package to register a plugin for another one.

:::{hint}
You can learn more about `entry_points` in the
[python packaging documentation](https://setuptools.pypa.io/en/latest/userguide/entry_point.html).
:::

To register your backend through `entry_points`, you could add your backend function to the `torch_dynamo_backends` entry point group in the
`setup.py` file of your package like:

```python
...
setup(
    ...
    'torch_dynamo_backends': [
        'my_compiler = your_module.submodule:my_compiler',
    ]
    ...
)
```

Please replace the `my_compiler` before `=` to the name of your backend's name and replace the part after `=` to
the module and function name of your backend function.
The entry point will be added to your python environment after the installation of the package.
When you call `torch.compile(model, backend="my_compiler")`, PyTorch would first search the backend named `my_compiler`
that has been registered with `register_backend`. If not found, it will continue to search in all backends registered
via `entry_points`.

Registration serves two purposes:

- You can pass a string containing your backend function's name to `torch.compile` instead of the function itself,
  for example, `torch.compile(model, backend="my_compiler")`.
- It is required for use with the [minifier](https://pytorch.org/docs/main/torch.compiler_troubleshooting_old.html#minifier). Any generated
  code from the minifier must call your code that registers your backend function, typically through an `import` statement.

## Custom Backends after AOTAutograd

It is possible to define custom backends that are called by AOTAutograd rather than TorchDynamo.
This is useful for 2 main reasons:

- Users can define backends that support model training, as AOTAutograd can generate the backward graph for compilation.
- AOTAutograd produces FX graphs consisting of [core Aten ops](https://pytorch.org/docs/main/torch.compiler_ir.html#core-aten-ir). As a result,
  custom backends only need to support the core Aten opset, which is a significantly smaller opset than the entire torch/Aten opset.

Wrap your backend with
`torch._dynamo.backends.common.aot_autograd` and use `torch.compile` with the `backend` kwarg as before.
Backend functions wrapped by `aot_autograd` should have the same contract as before.

Backend functions are passed to `aot_autograd` through the `fw_compiler` (forward compiler)
or `bw_compiler` (backward compiler) kwargs. If `bw_compiler` is not specified, the backward compile function
defaults to the forward compile function.

One caveat is that AOTAutograd requires compiled functions returned by backends to be "boxed". This can be done by wrapping
the compiled function with `functorch.compile.make_boxed_func`.

For example,

```python
from torch._dynamo.backends.common import aot_autograd
from functorch.compile import make_boxed_func

def my_compiler(gm, example_inputs):
    return make_boxed_func(gm.forward)

my_backend = aot_autograd(fw_compiler=my_compiler)  # bw_compiler=my_compiler

model_opt = torch.compile(model, backend=my_backend)
```

## Examples

### Debugging Backend

If you want to better understand what is going on during a
compilation, you can create a custom compiler, which is referred to as
backend in this section, that will print pretty print the fx
`GraphModule` extracted from Dynamo’s bytecode analysis
and return a `forward()` callable.

For example:

```python
from typing import List
import torch
def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    print("my_compiler() called with FX graph:")
    gm.graph.print_tabular()
    return gm.forward  # return a python callable
@torch.compile(backend=my_compiler)
def fn(x, y):
    a = torch.cos(x)
    b = torch.sin(y)
    return a + b
fn(torch.randn(10), torch.randn(10))
```

Running the above example produces the following output:

```
my_compiler() called with FX graph:
opcode         name    target                                                  args        kwargs
-------------  ------  ------------------------------------------------------  ----------  --------
placeholder    x       x                                                       ()          {}
placeholder    y       y                                                       ()          {}
call_function  cos     <built-in method cos of type object at 0x7f1a894649a8>  (x,)        {}
call_function  sin     <built-in method sin of type object at 0x7f1a894649a8>  (y,)        {}
call_function  add     <built-in function add>                                 (cos, sin)  {}
output         output  output                                                  ((add,),)   {}
```

This works for `torch.nn.Module` as well as shown below:

```python
from typing import List
import torch
def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    print("my_compiler() called with FX graph:")
    gm.graph.print_tabular()
    return gm.forward  # return a python callable
class MockModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        return self.relu(torch.cos(x))
mod = MockModule()
optimized_mod = torch.compile(mod, backend=my_compiler)
optimized_mod(torch.randn(10))
```

Let’s take a look at one more example with control flow:

```python
from typing import List
import torch
def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    print("my_compiler() called with FX graph:")
    gm.graph.print_tabular()
    return gm.forward  # return a python callable
@torch.compile(backend=my_compiler)
def toy_example(a, b):
    x = a / (torch.abs(a) + 1)
    if b.sum() < 0:
        b = b * -1
    return x * b
for _ in range(100):
    toy_example(torch.randn(10), torch.randn(10))
```

Running this example produces the following output:

```
my_compiler() called with FX graph:
opcode         name     target                                                  args              kwargs
-------------  -------  ------------------------------------------------------  ----------------  --------
placeholder    a        a                                                       ()                {}
placeholder    b        b                                                       ()                {}
call_function  abs_1    <built-in method abs of type object at 0x7f8d259298a0>  (a,)              {}
call_function  add      <built-in function add>                                 (abs_1, 1)        {}
call_function  truediv  <built-in function truediv>                             (a, add)          {}
call_method    sum_1    sum                                                     (b,)              {}
call_function  lt       <built-in function lt>                                  (sum_1, 0)        {}
output         output   output                                                  ((truediv, lt),)  {}

my_compiler() called with FX graph:
opcode         name    target                   args         kwargs
-------------  ------  -----------------------  -----------  --------
placeholder    b       b                        ()           {}
placeholder    x       x                        ()           {}
call_function  mul     <built-in function mul>  (b, -1)      {}
call_function  mul_1   <built-in function mul>  (x, mul)     {}
output         output  output                   ((mul_1,),)  {}

my_compiler() called with FX graph:
opcode         name    target                   args       kwargs
-------------  ------  -----------------------  ---------  --------
placeholder    b       b                        ()         {}
placeholder    x       x                        ()         {}
call_function  mul     <built-in function mul>  (x, b)     {}
output         output  output                   ((mul,),)  {}

The order of the last two graphs is nondeterministic depending
on which one is encountered first by the just-in-time compiler.
```

### Speedy Backend

Integrating a custom backend that offers superior performance is also
easy and we’ll integrate a real one
with [optimize_for_inference](https://pytorch.org/docs/stable/generated/torch.jit.optimize_for_inference.html):

```python
def optimize_for_inference_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    scripted = torch.jit.script(gm)
    return torch.jit.optimize_for_inference(scripted)
```

And then you should be able to optimize any existing code with:

```python
@torch.compile(backend=optimize_for_inference_compiler)
def code_to_accelerate():
    ...
```

### Composable Backends

TorchDynamo includes many backends, which can be listed with
`torch._dynamo.list_backends()`. You can combine these backends
together with the following code:

```python
from torch._dynamo import lookup_backend
def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    try:
        trt_compiled = lookup_backend("tensorrt")(gm, example_inputs)
        if trt_compiled is not None:
            return trt_compiled
    except Exception:
        pass
    # first backend failed, try something else...
    try:
        inductor_compiled = lookup_backend("inductor")(gm, example_inputs)
        if inductor_compiled is not None:
            return inductor_compiled
    except Exception:
        pass
    return gm.forward
```
