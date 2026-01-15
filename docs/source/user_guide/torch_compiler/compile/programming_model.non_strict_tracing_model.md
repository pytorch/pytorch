---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  execution_timeout: 30
  execution_show_tb: True
  merge_streams: True
---

```{code-cell}
:tags: [remove-cell]
import torch

import header_code
```

# Non-strict Tracing Programming Model

**Summary:**
- **Non-strict tracing** is a way to trace Python code that is less strict than Dynamo, but may result in silent incorrectness.
- Non-strict tracing runs a Python function and uses Python and PyTorch’s operator overloading capabilities to record what Tensor operations occurred during execution into a trace.
- A function is **non-strict traceable** if it complies with some constraints, namely, that the function is **pure** and does not directly manipulate Tensor.data_ptr().
- Non-strict tracing may **specialize** on certain variables and treat them as **constants**, baking the values of the variables into the trace.

`torch.compile` internals (`make_fx`, AOTDispatcher) use **non-strict tracing**. [`torch._dynamo.nonstrict_trace`](programming_model.dynamo_nonstrict_trace) can also be used in `torch.compile`d code to mark sections of code to be traced with non-strict tracing.
Non-strict tracing runs a Python function and uses Python and PyTorch’s operator overloading capabilities to record what Tensor operations occurred during execution into a trace.

**`make_fx`** is the main entrypoint for non-strict tracing. For the following function, only the top branch is taken during execution of the inputs, so it captures a graph with only that branch.

```{code-cell}
from torch.fx.experimental.proxy_tensor import make_fx
def f(x):
    if x.shape[0] > 2:
        return x ** 2 / 6
    else:
        return x * 3
x = torch.randn(3)
gm = make_fx(f, tracing_mode="fake")(x)
gm.print_readable()
```

Non-strict tracing differs from Dynamo (strict) tracing in that **it is unsafe**, that is, given a function, it captures a graph of Tensor operations that may have different semantics than the original function.
Given a Python function, Dynamo Tracing captures a graph of Tensor operations and residual bytecode that when combined give the same semantics as the Python function.

(programming_model.non_strict_tracing_model.pure_functions)=

## Pure Functions

Non-strict tracing is sound only on **pure functions**, and thus only pure functions should be non-strict traced.

A pure function is a function with the following properties:

- **Determinism.** Given the same inputs, the pure function will always return the same output.
- **No side effects.** A pure function does not have any side effects such as modifying external state or performing I/O operations.
- **Explicit input/output.** All the input data must be passed through the function parameters and all of the outputs are returned from the function.

Here are some examples of impure functions for which the captured graph behaves differently from the original function.

### Example 1: No explicit input (e.g. accesses global tensor)
```{code-cell}
var = torch.tensor(1)
def function_with_global_access(y):
    return y + var
x = torch.tensor([0, 1, 2])
# _allow_non_fake_inputs=True is needed to capture the global variable
# for demonstration purposes.
gm = make_fx(
    function_with_global_access, tracing_mode="fake", _allow_non_fake_inputs=True
)(x)
# Non-strict Tracing captures the value of the global (1.)
print("1. call function", function_with_global_access(x))
print("1. call graph", gm(x))
# However, after changing the global, the captured graph
# produces a different result from the original function
var = torch.tensor(2)
print("2. call function", function_with_global_access(x))
print("2. call graph", gm(x))
# To capture a graph that can have a varying `var` tensor,
# it must be an explicit input:
def function_fixed(y, var):
    return y + var
var = torch.tensor(3)
gm = make_fx(function_fixed, tracing_mode="fake")(x, var)
print("3. call function", function_fixed(x, var))
print("3. call graph", gm(x, var))
var = torch.tensor(4)
print("4. call function", function_fixed(x, var))
print("4. call graph", gm(x, var))
```

See [Specialization and Constants](specialization-and-constants) for an explanation of why.

### Example 2: Side effect (printing)

```{code-cell}
def function_with_side_effect(y):
    print(y)
x = torch.tensor([0, 1, 2])
_ = function_with_side_effect(x)
```

Running `f` in Python prints a Tensor as a side effect.

```{code-cell}
gm = make_fx(function_with_side_effect, tracing_mode="fake")(x)
```

During non-strict tracing, this print occurs during the graph capture.

```{code-cell}
_ = gm(x)
```

The graph does not store a call to the `print` statement, so executing the graph doesn’t print anything.

### Example 3: Side effect (input list mutation)

```{code-cell}
lst = []
def function_with_input_list_mutation(lst):
    val = lst.pop()
    return val
x = torch.tensor([0, 1, 2])
y = torch.tensor([0, 1, 2])
# Each time the function is executed, the list shrinks in size
lst = [x, y]
function_with_input_list_mutation(lst)
print("len(lst) after one call", len(lst))
function_with_input_list_mutation(lst)
print("len(lst) after two calls", len(lst))
# With Non-strict Tracing, the length of the list shrinks during
# the graph capture but not in invocations of the graph.
lst = [x, y]
gm = make_fx(function_with_input_list_mutation, tracing_mode="fake")(lst)
print("len(lst) after graph capture", len(lst))
gm(lst)
print("len(lst) after one call to graph", len(lst))
gm(lst)
print("len(lst) after two calls to graph", len(lst))
```

### No direct data_ptr manipulation
Directly manipulating `Tensor.data_ptr` is not non-strict traceable. The intuition behind this is that PyTorch is unable to tell *how* you manipulated the `data_ptr`.

```{code-cell}
import ctypes
# Create a tensor with a single element
tensor = torch.tensor([42], dtype=torch.int32)  # Using int32 for simplicity
def function_with_data_ptr(tensor):
    # Get the data pointer
    ptr = tensor.data_ptr()
    # Cast the pointer to a ctypes pointer
    ctypes_ptr = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_int32))
    # Increment the value at the pointer
    ctypes_ptr.contents.value += 1
    return tensor
try:
    make_fx(function_with_data_ptr, tracing_mode="fake")(tensor)
except Exception as e:
    print(e)
```

(specialization-and-constants)=
## Specialization and Constants

Non-strict tracing captures a graph that may be specialized on some values. What this means is the captured graph is only valid for these values. We say the graph treats those values as **constant**.

All non-Tensor variables are treated as constant during Non-strict Tracing:

```{code-cell}
def f(x, y):
    return x + y
x = torch.tensor([0, 1, 2])
y = 3.14
gm = make_fx(f, tracing_mode="fake")(x, y)
gm.print_readable()
```

3.14 is a constant in the graph.

Non-strict tracing will also specialize on properties of the input Tensors.

```{code-cell}
def f(x):
    if x.shape[0] > 2:
        return x ** 2 / 6
    else:
        return x * 3
x = torch.randn(3)
gm = make_fx(f, tracing_mode="fake")(x)
gm.print_readable()
```

And it will also specialize on any variables not directly passed into the function:

```{code-cell}
var = torch.tensor(1)
def f(x):
    return x + y
x = torch.randn(3)
gm = make_fx(f, tracing_mode="fake")(x)
gm.print_readable()
```
