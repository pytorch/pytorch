# Dynamo Overview

Before you read this section, read {ref}`torch.compiler_overview`.

TorchDynamo (or simply Dynamo) is a Python-level Just-In-Time (JIT) compiler designed to make
unmodified PyTorch programs faster. Dynamo hooks into the frame evaluation
API in CPython ([PEP 523](https://peps.python.org/pep-0523/)) to
dynamically modify Python bytecode right before it is executed. It
rewrites Python bytecode to extract sequences of PyTorch
operations into an [FX Graph](https://pytorch.org/docs/stable/fx.html)
which is then compiled with a customizable backend.
It creates this FX Graph through bytecode analysis and is designed to
mix Python execution with compiled backends to get the best of both
worlds — usability and performance.

Dynamo makes it easy to experiment with different compiler
backends to make PyTorch code faster with a single line decorator
`torch._dynamo.optimize()` which is wrapped for convenience by `torch.compile()`

The following diagram demonstrates how PyTorch works with `torch.compile`
and without it:

```{image} _static/img/dynamo/TorchDynamo.png
```

`TorchInductor` is one of the backends
supported by [Dynamo Graph](https://pytorch.org/docs/stable/fx.html)
into [Triton](https://github.com/openai/triton) for GPUs or
[C++/OpenMP](https://www.openmp.org/) for CPUs. We have a
[training performance dashboard](https://github.com/pytorch/torchdynamo/issues/681#issuecomment-1233828468)
that provides performance comparison for different training backends. You can read
more in the [TorchInductor post on PyTorch
dev-discuss](https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747).

For an in-depth overview, read the sections below, watch the deep-dive video,
and check out the dev-discuss topics.

- [Dynamo deep-dive video](https://www.youtube.com/watch?v=egZB5Uxki0I)
- [dev-discuss topics](https://dev-discuss.pytorch.org/search?q=TorchDynamo%20order%3Alatest)
## Dynamo Internals

**Author**: [Jason Ansel](https://github.com/jansel) and [Kaichao You](https://github.com/youkaichao)

This section will go over some of the Dynamo internals and will
demonstrate how Dynamo works under the hood.

### What is a guard?

Dynamo operates just-in-time and specializes graphs based on
dynamic properties. Below is a basic example of how to use Dynamo.
One can decorate a function or a method using `torchdynamo.optimize` to enable
Dynamo optimization:

```python
from typing import List
import torch
from torch import _dynamo as torchdynamo
def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    print("my_compiler() called with FX graph:")
    gm.graph.print_tabular()
    return gm.forward  # return a python callable

@torchdynamo.optimize(my_compiler)
def toy_example(a, b):
    x = a / (torch.abs(a) + 1)
    if b.sum() < 0:
        b = b * -1
    return x * b
for _ in range(100):
    toy_example(torch.randn(10), torch.randn(10))
```

For example, the first graph above has the following
guards:

```
GUARDS:
hasattr(L['a'], '_dynamo_dynamic_indices') == False
hasattr(L['b'], '_dynamo_dynamic_indices') == False
utils_device.CURRENT_DEVICE == None
___skip_backend_check() or ___current_backend() == ___lookup_backend(140355900538256)
check_tensor(L['a'], Tensor, DispatchKeySet(CPU, BackendSelect, ADInplaceOrView, AutogradCPU), torch.float32, device=None, requires_grad=False, size=[10], stride=[1])
check_tensor(L['b'], Tensor, DispatchKeySet(CPU, BackendSelect, ADInplaceOrView, AutogradCPU), torch.float32, device=None, requires_grad=False, size=[10], stride=[1])
```

If any of those guards fail, the graph will be recaptured and
recompiled. The interesting guard there is `check_tensor`, which
checks the following `torch.Tensor` properties:

- Python class of the tensor (tensor subclassing, etc)
- dtype
- device
- requires_grad
- dispatch_key (with thread-local includes/excludes applied)
- ndim
- sizes\*
- strides\*

The full specialization mode allows the backend compiler to assume an
entirely static graph. Unfortunately, most backends require this.
Operators which return dynamic shapes will trigger a graph break when
not in dynamic shape mode.

### What is Dynamo doing?

If you want to understand better what Dynamo is doing, you can run your code with:

```
TORCH_LOGS="+dynamo,guards,bytecode"
```

If you are not familiar with Python bytecode, you can add a decompiler hook
to decompile the bytecode into human-readable source code. One available
tool is [depyf](https://github.com/youkaichao/depyf). If you don't have
`depyf` already installed, run `pip install depyf`. Then, add the
following code to install decompilation hooks before you run any code.

```python
import depyf
depyf.install()
```

This code triggers useful (but spammy) printouts.

For example, the printouts for the first graph in the `toy_example`
are:

```
__compiled_fn_0 <eval_with_key>.1
opcode         name     target                                                  args              kwargs
-------------  -------  ------------------------------------------------------  ----------------  --------
placeholder    a        a                                                       ()                {}
placeholder    b        b                                                       ()                {}
call_function  abs_1    <built-in method abs of type object at 0x7f9ca082f8a0>  (a,)              {}
call_function  add      <built-in function add>                                 (abs_1, 1)        {}
call_function  truediv  <built-in function truediv>                             (a, add)          {}
call_method    sum_1    sum                                                     (b,)              {}
call_function  lt       <built-in function lt>                                  (sum_1, 0)        {}
output         output   output                                                  ((truediv, lt),)  {}
ORIGINAL BYTECODE toy_example example.py line 12
 14           0 LOAD_FAST                0 (a)
              2 LOAD_GLOBAL              0 (torch)
              4 LOAD_METHOD              1 (abs)
              6 LOAD_FAST                0 (a)
              8 CALL_METHOD              1
             10 LOAD_CONST               1 (1)
             12 BINARY_ADD
             14 BINARY_TRUE_DIVIDE
             16 STORE_FAST               2 (x)
 15          18 LOAD_FAST                1 (b)
             20 LOAD_METHOD              2 (sum)
             22 CALL_METHOD              0
             24 LOAD_CONST               2 (0)
             26 COMPARE_OP               0 (<)
             28 POP_JUMP_IF_FALSE       19 (to 38)
 16          30 LOAD_FAST                1 (b)
             32 LOAD_CONST               3 (-1)
             34 BINARY_MULTIPLY
             36 STORE_FAST               1 (b)
 17     >>   38 LOAD_FAST                2 (x)
             40 LOAD_FAST                1 (b)
             42 BINARY_MULTIPLY
             44 RETURN_VALUE
MODIFIED BYTECODE toy_example example.py line 12
 12           0 LOAD_GLOBAL              3 (__compiled_fn_0)
              2 LOAD_FAST                0 (a)
              4 LOAD_FAST                1 (b)
              6 CALL_FUNCTION            2
              8 UNPACK_SEQUENCE          2
             10 STORE_FAST               2 (x)
             12 POP_JUMP_IF_FALSE       12 (to 24)
             14 LOAD_GLOBAL              4 (__resume_at_30_1)
             16 LOAD_FAST                1 (b)
             18 LOAD_FAST                2 (x)
             20 CALL_FUNCTION            2
             22 RETURN_VALUE
        >>   24 LOAD_GLOBAL              5 (__resume_at_38_2)
             26 LOAD_FAST                1 (b)
             28 LOAD_FAST                2 (x)
             30 CALL_FUNCTION            2
             32 RETURN_VALUE
possible source code:
def toy_example(a, b):
    __temp_1 = __compiled_fn_0(a, b)
    x = __temp_1[0]
    if __temp_1[1]:
        return __resume_at_30_1(b, x)
    return __resume_at_38_2(b, x)
If you find the decompiled code is wrong,please submit an issue at https://github.com/youkaichao/depyf/issues.
```

At the top you can see the FX graph.
Next, you see the original bytecode of the function, followed by the
modified bytecode generated by Dynamo, and the decompiled source
code for reference. Finally, you see the guards which we covered above.

In the modified bytecode, `__compiled_fn_0` is the return value of
`my_compiler()` (the compiled graph). `__resume_at_30_1` and
`__resume_at_38_2` are both generated continuation functions that pick
up execution after a graph break (at bytecode offsets 30 and 38). Each
of these functions take the form:

```
__resume_at_<offset>:
    ... restore stack state if needed ...
    JUMP_ABSOLUTE <offset> into toy_example
    ... original bytecode of toy_example ...
```

By generating this `resume_at` function, we force the remainder of the
function to be executed in a new Python frame which recursively
triggers Dynamo to restart its capture once execution reaches that
point for the first time.

### How to inspect artifacts generated by Dynamo?

To inspect the artifacts generated by Dynamo, there is an API `torch._dynamo.eval_frame._debug_get_cache_entry_list` that retrieves compiled code and guards out of a function's `__code__` object. A compiled function can have several cache entries, and each cache entry consists a generated function to check guards, and a `types.CodeType` object to keep the code to be executed if the guarding conditions are satisfied.

```python
from torch._dynamo.eval_frame import _debug_get_cache_entry_list, innermost_fn
cache_entries = _debug_get_cache_entry_list(innermost_fn(toy_example))
cache_entry = cache_entries[0]
guard, code = cache_entry.check_fn, cache_entry.code
# the guard takes the local variables of an input frame, and tells whether a re-compilation should be triggered.
import dis
dis.dis(guard)
dis.dis(code)
```

If you know Python bytecode, you can understand the above output.

For the guard function, there is no need to inspect the bytecode. We can directly access its guarding conditions:

```python
for code_part in guard.code_parts:
    print(code_part)
```

The output is:

```
___guarded_code.valid
___check_global_state()
hasattr(L['a'], '_dynamo_dynamic_indices') == False
hasattr(L['b'], '_dynamo_dynamic_indices') == False
utils_device.CURRENT_DEVICE == None
___skip_backend_check() or ___current_backend() == ___lookup_backend(140215810860528)
___check_tensors(L['a'], L['b'], tensor_check_names=tensor_check_names)
```

Only when all the conditions are satisfied, the guard function returns true, and the compiled code is executed.

For the compiled code, we cannot directly access its source but have to decompile it.

```python
from depyf import decompile
print(decompile(code))
```

The output is:

```
def toy_example(a, b):
    __temp_1 = __compiled_fn_0(a, b)
    x = __temp_1[0]
    if __temp_1[1]:
        return __resume_at_30_1(b, x)
    return __resume_at_38_2(b, x)
```

Some names referenced in the code are:

- Compiled functions, stored in the global namespace of the module containing the original function `toy_example`. These include names like `__compiled_fn_0` / `__resume_at_30_1` / `__resume_at_38_2`.
- Closure variables used for checking guards. The names can be accessed from `guard.__code__.co_freevars`, and the values are stored in `guard.__closure__`. These include names like `___guarded_code` / `___is_grad_enabled` / `___are_deterministic_algorithms_enabled` / `___is_torch_function_enabled` / `utils_device` / `___check_tensors` / `tensor_check_names`.
- Argument `L` of the `guard` function. This is a dict mapping the name of arguments of `toy_example` to its values. This is only available when the function is called, where the frame evaluation API comes into play. In short, `L` is a `dict` with structure of `{'a': value_a, 'b': value_b}`. Therefore, you can see the code uses `L['a']` to refer to the input variable `a`.

The graph break is shown in the code of compiled `toy_example`, where we have to use Python interpreter to select the following graph to execute.

Note that we pass a simple `my_compiler` function as the backend compiler, therefore the subgraph code `__resume_at_38_2`, `__resume_at_30_1`, and `__compiled_fn_0` remain Python code. This can also be inspected (please ignore the function name, and only use the function signature and function body code):

```python
print("source code of __compiled_fn_0:")
print(innermost_fn(__compiled_fn_0).__self__.code)
print("=" * 60)
print("source code of __resume_at_30_1:")
print(decompile(__resume_at_30_1))
print("=" * 60)
print("source code of __resume_at_38_2:")
print(decompile(__resume_at_38_2))
```

```
source code of __compiled_fn_0:
def forward(self, L_a_ : torch.Tensor, L_b_ : torch.Tensor):
    l_a_ = L_a_
    l_b_ = L_b_
    abs_1 = torch.abs(l_a_)
    add = abs_1 + 1;  abs_1 = None
    truediv = l_a_ / add;  l_a_ = add = None
    sum_1 = l_b_.sum();  l_b_ = None
    lt = sum_1 < 0;  sum_1 = None
    return (truediv, lt)
# To see more debug info, please use ``graph_module.print_readable()``
============================================================
source code of __resume_at_30_1:
def <resume in toy_example>(b, x):
    b = b * -1
    return x * b
============================================================
source code of __resume_at_38_2:
def <resume in toy_example>(b, x):
    return x * b
```

However, if we use other backends like the built-in `inductor`, the subgraph code will be compiled CUDA kernels for GPU or C++ code for CPU.

To summarize, the compiled code is conceptually equivalent to the code below:

```python
def compiled_example(a, b):
    L = {'a': a, 'b': b}
    for guard, code in get_cache_entries():
        if guard(L):
            return code(a, b)
    recompile_and_add_another_cache_entry()
```

The following diagram demonstrates how `torch.compile` transforms and optimizes user-written code: it first extracts computation graphs from the user-written function, and compiles these graphs into optimized functions, then assembles them into a new function, which is functionally equivalent to the user-written code but optimized to have a good computation speed.

```{image} _static/img/dynamo/flowchart.jpg
```

To learn more about how all this is implemented internally, see {ref}`torch.compiler_dynamo_deepdive`.