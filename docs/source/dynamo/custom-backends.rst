Custom Backends
===============

Debugging Backend
-----------------

If you want to better understand what is going on during a
compilation, you can create a custom compiler, which is referred to as
backend in this section, that will print pretty print the fx
``GraphModule`` extracted from Dynamo’s bytecode analysis
and return a ``forward()`` callable.

For example:

.. code-block:: python

   from typing import List
   import torch
   import torch._dynamo as dynamo
   def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
       print("my_compiler() called with FX graph:")
       gm.graph.print_tabular()
       return gm.forward  # return a python callable
   @dynamo.optimize(my_compiler)
   def fn(x, y):
       a = torch.cos(x)
       b = torch.sin(y)
       return a + b
   fn(torch.randn(10), torch.randn(10))

Running the above example produces the following output:

::

   my_compiler() called with FX graph:
   opcode         name    target                                                  args        kwargs
   -------------  ------  ------------------------------------------------------  ----------  --------
   placeholder    x       x                                                       ()          {}
   placeholder    y       y                                                       ()          {}
   call_function  cos     <built-in method cos of type object at 0x7f1a894649a8>  (x,)        {}
   call_function  sin     <built-in method sin of type object at 0x7f1a894649a8>  (y,)        {}
   call_function  add     <built-in function add>                                 (cos, sin)  {}
   output         output  output                                                  ((add,),)   {}

This works for ``torch.nn.Module`` as well as shown below:

.. code-block:: python

   import torch
   import torch._dynamo as dynamo
   class MockModule(torch.nn.Module):
       def __init__(self):
           super().__init__()
           self.relu = torch.nn.ReLU()
       def forward(self, x):
           return self.relu(torch.cos(x))
   mod = MockModule()
   optimized_mod = dynamo.optimize(my_compiler)(mod)
   optimized_mod(torch.randn(10))

Let’s take a look at one more example with control flow:

.. code-block:: python

   from typing import List
   import torch
   import torch._dynamo as dynamo
   def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
       print("my_compiler() called with FX graph:")
       gm.graph.print_tabular()
       return gm.forward  # return a python callable
   @dynamo.optimize(my_compiler)
   def toy_example(a, b):
       x = a / (torch.abs(a) + 1)
       if b.sum() < 0:
           b = b * -1
       return x * b
   for _ in range(100):
       toy_example(torch.randn(10), torch.randn(10))

Running this example produces the following output:

::

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

Speedy Backend
--------------

Integrating a custom backend that offers superior performance is also
easy and we’ll integrate a real one
with `optimize_for_inference <https://pytorch.org/docs/stable/generated/torch.jit.optimize_for_inference.html>`__:

.. code-block:: python

   def optimize_for_inference_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
       scripted = torch.jit.trace(gm, example_inputs)
       return torch.jit.optimize_for_inference(scripted)

And then you should be able to optimize any existing code with:

.. code-block:: python

   @dynamo.optimize(optimize_for_inference_compiler)
   def code_to_accelerate():
       ...

Composable Backends
-------------------

TorchDynamo includes many backends, which can be found in
`backends.py <https://github.com/pytorch/pytorch/blob/master/torch/_dynamo/optimizations/backends.py>`__
or ``torchdynamo.list_backends()``. You can combine these backends
together with the following code:

.. code-block:: python

   from torch._dynamo.optimizations import BACKENDS
   def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
       trt_compiled = BACKENDS["tensorrt"](gm, example_inputs)
       if trt_compiled is not None:
           return trt_compiled
       # first backend failed, try something else...
       cudagraphs_compiled = BACKENDS["cudagraphs"](gm, example_inputs)
       if cudagraphs_compiled is not None:
           return cudagraphs_compiled
       return gm.forward
