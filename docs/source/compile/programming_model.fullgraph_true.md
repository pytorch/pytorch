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

# Use `fullgraph=True` to Identify and Eliminate Graph Breaks

Using `torch.compile(fullgraph=False)` (the default) is a good way to get started with `torch.compile`: it supports all Python programs out-of-the-box via the ability to graph break and gives good performance on common cases.

However, if you're trying to get more performance out of your model, you should explicitly think about what regions of code should be compiled:
- We recommend using `torch.compile(fullgraph=True)` to find and eliminate graph breaks in your code.
- If you're a library developer (or testing if your code "works" with `torch.compile`), we recommend testing using `torch.compile(fullgraph=True)`.

`torch.compile(fullgraph=True)` offers stronger guarantees over `fullgraph=False`:
we will always capture a single FX graph to be compiled (or error if we cannot due to a graph break).
**In particular, you are forced to resolve every graph break that is encountered.**

There are a number of strategies for resolving a graph break.

## Strategy 1:  Rewrite the unsupported code to use features supported by Dynamo

Many graph break error messages will give some suggestions on how to rewrite code to avoid the graph break.
If the graph break is still difficult to resolve, then please move on to the next strategy
or submit an issue to the [PyTorch GitHub repo](https://github.com/pytorch/pytorch/issues).

More graph break examples and how to resolve them can be found in [Common Graph Breaks](programming_model.common_graph_breaks).

Example: Dynamo does not support calling `next` on a `list_iterator` object that was an input to the function being compiled.

```{code-cell}
@torch.compile(fullgraph=True)
def f(xs):
    a = next(xs)
    b = next(xs)
    return a + b

xs = [torch.tensor(1.), torch.tensor(2.)]
try:
    out = f(iter(xs))
except Exception as e:
    print(e)
```

Instead, rewrite the compiled function to accept a list.

```{code-cell}
@torch.compile(fullgraph=True)
def f_rewritten(xs):
    it = iter(xs)
    a = next(it)
    b = next(it)
    return a + b

f_rewritten(xs)
```

## Strategy 2: Pure functions can always be compiled via an escape hatch.

**Summary**: The space of all Python functions is vast and thus it is impractical for Dynamo to be able to trace
through every Python function without graph breaks. For Python functions considered to be "pure"
that Dynamo cannot trace through without graph breaks, we provide some escape hatches to attempt
to trace through these functions anyway:

1. Use `custom_op` or `triton_op` on pure triton kernels.
2. Use `nonstrict_trace` for pure functions that only use PyTorch Tensor ops.
3. Use `custom_op` for all other pure functions.

A "pure function" is a function with the following properties:

- Determinism. Given the same inputs, the pure function will always return the same output
- No external side effects. A pure function does not have any externally-visible side effects,
  such as modifying external state or performing I/O operations.
  Side effects that remain internal to the function are allowed (e.g. mutating intermediate tensors).
  One notable exception is that mutating `torch.*` ops on function input Tensors are generally allowed.
- Explicit input/output. All the input data must be passed through the function parameters and all of the outputs are returned from the function.

See Pure Functions for examples. <!-- TODO: link -->

Dynamo is theoretically able to handle a wide variety of impure functions, but may be lacking coverage for specific
Python language features. However, pure functions can always be compiled via an escape hatch.

If you have a graph break it may be possible to refactor the code around it into a pure function and use an escape hatch that bypasses Dynamo tracing:

1. Use `torch._dynamo.nonstrict_trace` if you want the Tensor operations in the function to show up in the Dynamo output graph (and therefore be optimizable). `nonstrict_trace` tells Dynamo to use **non-strict tracing**.
2. Use custom operators if you want the function to be opaque w.r.t. to `torch.compile` (both the frontend Dynamo and the backend).

Note that there is nothing preventing these escape hatches from being applied to impure functions,
but **we do not provide any soundness guarantees**.

Example: If Dynamo doesn't support some Python feature or API that is non-strict traceable (e.g. it uses PyTorch operations), use `nonstrict_trace` to capture it instead. <!-- TODO: link -->

```{code-cell}
# this is a function that Dynamo doesn't support (due to the graph_break() call).
def g(x):
    y = x.sin()
    torch._dynamo.graph_break()
    z = y.sin()
    return z

@torch.compile(fullgraph=True)
def f(x):
    w = x.sin()
    return g(w)

x = torch.randn(3)
try:
    f(x)  # Graph Break: there was a call to torch._dynamo.graph_break()
except Exception as e:
    print(e)

@torch.compile(fullgraph=True)
def f_rewritten(x):
    w = x.sin()
    return torch._dynamo.nonstrict_trace(g)(w)
f_rewritten(x)  # works
```

Example: use custom operators to create opaque functions w.r.t. to `torch.compile` <!-- TODO: link -->

```{code-cell}
from torch.utils.cpp_extension import load_inline

# C++ source code for the square operation
cpp_source = """
torch::Tensor square_cpu(torch::Tensor input) {
    // Check that input is a CPU tensor
    TORCH_CHECK(input.device().is_cpu(), "Input must be a CPU tensor");

    // Create output tensor with same shape and dtype as input
    torch::Tensor output = torch::empty_like(input);

    // Get data pointers
    float* input_data = input.data_ptr<float>();
    float* output_data = output.data_ptr<float>();

    // Get total number of elements
    int64_t numel = input.numel();

    // For loop to compute square of each element
    for (int64_t i = 0; i < numel; i++) {
        output_data[i] = input_data[i] * input_data[i];
    }

    return output;
}
"""

# Load the extension inline
square_module = load_inline(
    name="square_cpu_kernel",
    cpp_sources=cpp_source,
    functions=["square_cpu"],
    verbose=True
)

def square(x):
    return square_module.square_cpu(x)

@torch.compile(fullgraph=True)
def f(x):
    return square(x)

try:
    f(torch.randn(3, 3))  # graph break
except Exception as e:
    print(e)
```

```{code-cell}
# Use torch.library.custom_op to define a new custom operator.
# Custom operators are opaque with respect to torch.compile:
# that is, torch.compile does not peek into them.

@torch.library.custom_op("mylib::square", mutates_args=())
def square(x: torch.Tensor) -> torch.Tensor:
    return square_module.square_cpu(x)

# Use register_fake to add a ``FakeTensor`` kernel for the operator
@square.register_fake
def _(x):
    return x.new_empty(x.size())

print(f(torch.randn(3, 3)))  # no graph break
```

For more information on `triton_op` for custom triton kernels, see the
[user-defined triton kernel tutorial](https://docs.pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html).


## Strategy 3: Don't compile the code

Not all code is amenable to being compiled. `torch.compile` is a compiler for Tensor computation;
it will not be able to optimize things like disk IO. Try to refactor the code such that the unsupported
code is not called in the compiled region.

```{code-cell}
@torch.compile(fullgraph=True)
def f(x):
   y = x ** 2  / 2
   torch.save(y, "foo.pt")
   z = y ** 3 / 6
   return z

x = torch.randn(3)
try:
    f(x)  # Graph Break: torch.save not supported
except Exception as e:
    print(e)
```

```{code-cell}
def f_rewritten(x):
   y = g(x)
   torch.save(y, "foo.pt")
   z = h(y)
   return z

@torch.compile(fullgraph=True)
def g(x):
   y = x ** 2  / 2
   return y

@torch.compile(fullgraph=True)
def h(y):
   z = y ** 3 / 6
   return z

f_rewritten(x)
```

```{code-cell}
:tags: [remove-cell]
import os
os.remove("foo.pt")
```
