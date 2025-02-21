---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  execution_timeout: 600
---
# Introduction to `torch.compile`

**Author:** William Wen

`torch.compile` is the latest method to speed up your PyTorch code! `torch.compile` makes PyTorch code run faster by JIT-compiling PyTorch code into optimized kernels, all while requiring minimal code changes.

In this tutorial, we cover basic `torch.compile` usage, and demonstrate the advantages of `torch.compile` over previous PyTorch compiler solutions, such as TorchScript and FX Tracing.

**Required pip dependencies for this tutorial**

* `torch >= 2.0`
* `torchvision`
* `numpy`
* `scipy`
* `tabulate`

**NOTE:** a modern NVIDIA GPU (H100, A100, or V100) is recommended for this tutorial in order to reproduce the speedup numbers shown below and documented elsewhere.

```{code-cell}
:tags: [remove-cell]
import torch
import warnings

gpu_ok = False
if torch.cuda.is_available():
    device_cap = torch.cuda.get_device_capability()
    if device_cap in ((7, 0), (8, 0), (9, 0)):
        gpu_ok = True

if not gpu_ok:
    warnings.warn(
        "GPU is not NVIDIA V100, A100, or H100. Speedup numbers may be lower "
        "than expected."
    )
```

## Basic Usage

`torch.compile` is included in the latest PyTorch. Running TorchInductor on GPU requires Triton, which is included with the PyTorch 2.0 nightly binary. If Triton is still missing, try installing `torchtriton` via `pip` (`pip install torchtriton --extra-index-url "https://download.pytorch.org/whl/nightly/cu117"` for CUDA 11.7).

Arbitrary Python functions can be optimized by passing the callable to torch.compile. We can then call the returned optimized function in place of the original function.

```{code-cell}
def foo(x, y):
    a = torch.sin(x)
    b = torch.cos(y)
    return a + b
opt_foo1 = torch.compile(foo)
print(opt_foo1(torch.randn(10, 10), torch.randn(10, 10)))
```

Alternatively, we can decorate the function.

```{code-cell}
t1 = torch.randn(10, 10)
t2 = torch.randn(10, 10)

@torch.compile
def opt_foo2(x, y):
    a = torch.sin(x)
    b = torch.cos(y)
    return a + b
print(opt_foo2(t1, t2))
```

We can also optimize `torch.nn.Module` instances.

```{code-cell}
t = torch.randn(10, 100)

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(100, 10)

    def forward(self, x):
        return torch.nn.functional.relu(self.lin(x))

mod = MyModule()
opt_mod = torch.compile(mod)
print(opt_mod(t))
```

## `torch.compile` and Nested Calls

Nested function calls within the decorated function will also be compiled.

```{code-cell}
def nested_function(x):
    return torch.sin(x)

@torch.compile
def outer_function(x, y):
    a = nested_function(x)
    b = torch.cos(y)
    return a + b

print(outer_function(t1, t2))
```

In the same fashion, when compiling a module all sub-modules and methods within it, that are not in a skip list, are also compiled.

```{code-cell}
class OuterModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inner_module = MyModule()
        self.outer_lin = torch.nn.Linear(10, 2)

    def forward(self, x):
        x = self.inner_module(x)
        return torch.nn.functional.relu(self.outer_lin(x))

outer_mod = OuterModule()
opt_outer_mod = torch.compile(outer_mod)
print(opt_outer_mod(t))
```

We can also disable some functions from being compiled by using `torch.compiler.disable`. Suppose you want to disable the tracing on just the `complex_function` function, but want to continue the tracing back in `complex_conjugate`. In this case, you can use `torch.compiler.disable(recursive=False)` option. Otherwise, the default is `recursive=True`.

```{code-cell}
def complex_conjugate(z):
    return torch.conj(z)

@torch.compiler.disable(recursive=False)
def complex_function(real, imag):
    # Assuming this function cause problems in the compilation
    z = torch.complex(real, imag)
    return complex_conjugate(z)

def outer_function():
    real = torch.tensor([2, 3], dtype=torch.float32)
    imag = torch.tensor([4, 5], dtype=torch.float32)
    z = complex_function(real, imag)
    return torch.abs(z)

# Try to compile the outer_function
try:
    opt_outer_function = torch.compile(outer_function)
    print(opt_outer_function())
except Exception as e:
    print("Compilation of outer_function failed:", e)
```

## Best Practices and Recommendations

When you use `torch.compile`, the compiler will try to recursively compile every function call inside the target function or module inside the target function or module that is not in a skip list (such as built-ins, some functions in the `torch.*` namespace).

Best Practices:

1. **Top-Level Compilation:** One approach is to compile at the highest level possible (i.e., when the top-level module is initialized/called) and selectively disable compilation when encountering excessive graph breaks or errors. If there are still many compile issues, compile individual subcomponents instead.

2. **Modular Testing:** Test individual functions and modules with `torch.compile` before integrating them into larger models to isolate potential issues.

3. **Disable Compilation Selectively:** If certain functions or sub-modules cannot be handled by `torch.compile`, use the `torch.compiler.disable` context managers to recursively exclude them from compilation.

4. **Compile Leaf Functions First:** In complex models with multiple nested functions and modules, start by compiling the leaf functions or modules first. For more information see [TorchDynamo APIs for fine-grained tracing](https://pytorch.org/docs/stable/torch.compiler_fine_grain_apis.html).

## Demonstrating Speedups

Let’s now demonstrate that using `torch.compile` can speed up real models. We will compare standard eager mode and `torch.compile` by evaluating and training a `torchvision` model on random data.

Before we start, we need to define some utility functions.

```{code-cell}
# Returns the result of running `fn()` and the time it took for `fn()` to run,
# in seconds. We use CUDA events and synchronization for the most accurate
# measurements.
def timed(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end) / 1000

# Generates random input and targets data for the model, where `b` is
# batch size.
def generate_data(b):
    return (
        torch.randn(b, 3, 128, 128).to(torch.float32).cuda(),
        torch.randint(1000, (b,)).cuda(),
    )

N_ITERS = 10

from torchvision.models import densenet121
def init_model():
    return densenet121().to(torch.float32).cuda()
```

First, let’s compare inference.

Note that in the call to `torch.compile`, we have the additional `mode` argument, which we will discuss below.

```{code-cell}
model = init_model()

# Reset since we are using a different mode.
import torch._dynamo
torch._dynamo.reset()

model_opt = torch.compile(model, mode="reduce-overhead")

inp = generate_data(16)[0]
with torch.no_grad():
    print("eager:", timed(lambda: model(inp))[1])
    print("compile:", timed(lambda: model_opt(inp))[1])
```

Notice that `torch.compile` takes a lot longer to complete compared to eager. This is because `torch.compile` compiles the model into optimized kernels as it executes. In our example, the structure of the model doesn’t change, and so recompilation is not needed. So if we run our optimized model several more times, we should see a significant improvement compared to eager.

```{code-cell}
eager_times = []
for i in range(N_ITERS):
    inp = generate_data(16)[0]
    with torch.no_grad():
        _, eager_time = timed(lambda: model(inp))
    eager_times.append(eager_time)
    print(f"eager eval time {i}: {eager_time}")

print("~" * 10)

compile_times = []
for i in range(N_ITERS):
    inp = generate_data(16)[0]
    with torch.no_grad():
        _, compile_time = timed(lambda: model_opt(inp))
    compile_times.append(compile_time)
    print(f"compile eval time {i}: {compile_time}")
print("~" * 10)

import numpy as np
eager_med = np.median(eager_times)
compile_med = np.median(compile_times)
speedup = eager_med / compile_med
assert(speedup > 1)
print(f"(eval) eager median: {eager_med}, compile median: {compile_med}, speedup: {speedup}x")
print("~" * 10)
```

And indeed, we can see that running our model with `torch.compile` results in a significant speedup. Speedup mainly comes from reducing Python overhead and GPU read/writes, and so the observed speedup may vary on factors such as model architecture and batch size. For example, if a model’s architecture is simple and the amount of data is large, then the bottleneck would be GPU compute and the observed speedup may be less significant.

You may also see different speedup results depending on the chosen `mode` argument. The `"reduce-overhead"` mode uses CUDA graphs to further reduce the overhead of Python. For your own models, you may need to experiment with different modes to maximize speedup. You can read more about modes [here](https://pytorch.org/get-started/pytorch-2.0/#user-experience).

You may might also notice that the second time we run our model with `torch.compile` is significantly slower than the other runs, although it is much faster than the first run. This is because the `"reduce-overhead"` mode runs a few warm-up iterations for CUDA graphs.

For general PyTorch benchmarking, you can try using `torch.utils.benchmark` instead of the `timed` function we defined above. We wrote our own timing function in this tutorial to show `torch.compile`’s compilation latency.

Now, let’s consider comparing training.

```{code-cell}
model = init_model()
opt = torch.optim.Adam(model.parameters())

def train(mod, data):
    opt.zero_grad(True)
    pred = mod(data[0])
    loss = torch.nn.CrossEntropyLoss()(pred, data[1])
    loss.backward()
    opt.step()

eager_times = []
for i in range(N_ITERS):
    inp = generate_data(16)
    _, eager_time = timed(lambda: train(model, inp))
    eager_times.append(eager_time)
    print(f"eager train time {i}: {eager_time}")
print("~" * 10)

model = init_model()
opt = torch.optim.Adam(model.parameters())
train_opt = torch.compile(train, mode="reduce-overhead")

compile_times = []
for i in range(N_ITERS):
    inp = generate_data(16)
    _, compile_time = timed(lambda: train_opt(model, inp))
    compile_times.append(compile_time)
    print(f"compile train time {i}: {compile_time}")
print("~" * 10)

eager_med = np.median(eager_times)
compile_med = np.median(compile_times)
speedup = eager_med / compile_med
assert(speedup > 1)
print(f"(train) eager median: {eager_med}, compile median: {compile_med}, speedup: {speedup}x")
print("~" * 10)
```

Again, we can see that `torch.compile` takes longer in the first iteration, as it must compile the model, but in subsequent iterations, we see significant speedups compared to eager.

We remark that the speedup numbers presented in this tutorial are for demonstration purposes only. Official speedup values can be seen at the [TorchInductor performance dashboard](https://hud.pytorch.org/benchmark/compilers).

## Comparison to TorchScript and FX Tracing

We have seen that `torch.compile` can speed up PyTorch code. Why else should we use `torch.compile` over existing PyTorch compiler solutions, such as TorchScript or FX Tracing? Primarily, the advantage of `torch.compile` lies in its ability to handle arbitrary Python code with minimal changes to existing code.

One case that `torch.compile` can handle that other compiler solutions struggle with is data-dependent control flow (the `if x.sum() < 0:` line below).

```{code-cell}
def f1(x, y):
    if x.sum() < 0:
        return -y
    return y

# Test that `fn1` and `fn2` return the same result, given
# the same arguments `args`. Typically, `fn1` will be an eager function
# while `fn2` will be a compiled function (torch.compile, TorchScript, or FX graph).
def test_fns(fn1, fn2, args):
    out1 = fn1(*args)
    out2 = fn2(*args)
    return torch.allclose(out1, out2)

inp1 = torch.randn(5, 5)
inp2 = torch.randn(5, 5)
```

TorchScript tracing `f1` results in silently incorrect results, since only the actual control flow path is traced.

```{code-cell}
traced_f1 = torch.jit.trace(f1, (inp1, inp2))
print("traced 1, 1:", test_fns(f1, traced_f1, (inp1, inp2)))
print("traced 1, 2:", test_fns(f1, traced_f1, (-inp1, inp2)))
```

FX tracing `f1` results in an error due to the presence of data-dependent control flow.

```{code-cell}
import traceback as tb
try:
    torch.fx.symbolic_trace(f1)
except:
    tb.print_exc()
```

If we provide a value for `x` as we try to FX trace `f1`, then we run into the same problem as TorchScript tracing, as the data-dependent control flow is removed in the traced function.

```{code-cell}
fx_f1 = torch.fx.symbolic_trace(f1, concrete_args={"x": inp1})
print("fx 1, 1:", test_fns(f1, fx_f1, (inp1, inp2)))
print("fx 1, 2:", test_fns(f1, fx_f1, (-inp1, inp2)))
```

Now we can see that `torch.compile` correctly handles data-dependent control flow.

```{code-cell}
# Reset since we are using a different mode.
torch._dynamo.reset()

compile_f1 = torch.compile(f1)
print("compile 1, 1:", test_fns(f1, compile_f1, (inp1, inp2)))
print("compile 1, 2:", test_fns(f1, compile_f1, (-inp1, inp2)))
print("~" * 10)
```

TorchScript scripting can handle data-dependent control flow, but this solution comes with its own set of problems. Namely, TorchScript scripting can require major code changes and will raise errors when unsupported Python is used.

In the example below, we forget TorchScript type annotations and we receive a TorchScript error because the input type for argument `y`, an `int`, does not match with the default argument type, `torch.Tensor`.

```{code-cell}
def f2(x, y):
    return x + y

inp1 = torch.randn(5, 5)
inp2 = 3

script_f2 = torch.jit.script(f2)
try:
    script_f2(inp1, inp2)
except:
    tb.print_exc()
```

However, `torch.compile` is easily able to handle `f2`.

```{code-cell}
compile_f2 = torch.compile(f2)
print("compile 2:", test_fns(f2, compile_f2, (inp1, inp2)))
print("~" * 10)
```

Another case that `torch.compile` handles well compared to previous compilers solutions is the usage of non-PyTorch functions.

```{code-cell}
import scipy
def f3(x):
    x = x * 2
    x = scipy.fft.dct(x.numpy())
    x = torch.from_numpy(x)
    x = x * 2
    return x
```

TorchScript tracing treats results from non-PyTorch function calls as constants, and so our results can be silently wrong.

```{code-cell}
inp1 = torch.randn(5, 5)
inp2 = torch.randn(5, 5)
traced_f3 = torch.jit.trace(f3, (inp1,))
print("traced 3:", test_fns(f3, traced_f3, (inp2,)))
```

TorchScript scripting and FX tracing disallow non-PyTorch function calls.

```{code-cell}
try:
    torch.jit.script(f3)
except:
    tb.print_exc()

try:
    torch.fx.symbolic_trace(f3)
except:
    tb.print_exc()
```

In comparison, `torch.compile` is easily able to handle the non-PyTorch function call.

```{code-cell}
compile_f3 = torch.compile(f3)
print("compile 3:", test_fns(f3, compile_f3, (inp2,)))
```

## TorchDynamo and FX Graphs

One important component of `torch.compile` is TorchDynamo. TorchDynamo is responsible for JIT compiling arbitrary Python code into [FX graphs](https://pytorch.org/docs/stable/fx.html#torch.fx.Graph), which can then be further optimized. TorchDynamo extracts FX graphs by analyzing Python bytecode during runtime and detecting calls to PyTorch operations.

Normally, TorchInductor, another component of `torch.compile`, further compiles the FX graphs into optimized kernels, but TorchDynamo allows for different backends to be used. In order to inspect the FX graphs that TorchDynamo outputs, let us create a custom backend that outputs the FX graph and simply returns the graph’s unoptimized forward method.

```{code-cell}
from typing import List
def custom_backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    print("custom backend called with FX graph:")
    gm.graph.print_tabular()
    return gm.forward

# Reset since we are using a different backend.
torch._dynamo.reset()

opt_model = torch.compile(init_model(), backend=custom_backend)
opt_model(generate_data(16)[0])
```

Using our custom backend, we can now see how TorchDynamo is able to handle data-dependent control flow. Consider the function below, where the line `if b.sum() < 0` is the source of data-dependent control flow.

```{code-cell}
def bar(a, b):
    x = a / (torch.abs(a) + 1)
    if b.sum() < 0:
        b = b * -1
    return x * b

opt_bar = torch.compile(bar, backend=custom_backend)
inp1 = torch.randn(10)
inp2 = torch.randn(10)
opt_bar(inp1, inp2)
opt_bar(inp1, -inp2)
```

The output reveals that TorchDynamo extracted 3 different FX graphs corresponding the following code (order may differ from the output above):

1. `x = a / (torch.abs(a) + 1)`

2. `b = b * -1; return x * b`

3. `return x * b`

When TorchDynamo encounters unsupported Python features, such as data-dependent control flow, it breaks the computation graph, lets the default Python interpreter handle the unsupported code, then resumes capturing the graph.

Let’s investigate by example how TorchDynamo would step through `bar`. If `b.sum() < 0`, then TorchDynamo would run graph 1, let Python determine the result of the conditional, then run graph 2. On the other hand, if `not b.sum() < 0`, then TorchDynamo would run graph 1, let Python determine the result of the conditional, then run graph 3.

This highlights a major difference between TorchDynamo and previous PyTorch compiler solutions. When encountering unsupported Python features, previous solutions either raise an error or silently fail. TorchDynamo, on the other hand, will break the computation graph.

We can see where TorchDynamo breaks the graph by using `torch._dynamo.explain`:

```{code-cell}
# Reset since we are using a different backend.
torch._dynamo.reset()
explain_output = torch._dynamo.explain(bar)(torch.randn(10), torch.randn(10))
print(explain_output)
```

In order to maximize speedup, graph breaks should be limited. We can force TorchDynamo to raise an error upon the first graph break encountered by `using fullgraph=True`:

```{code-cell}
opt_bar = torch.compile(bar, fullgraph=True)
try:
    opt_bar(torch.randn(10), torch.randn(10))
except:
    tb.print_exc()
```

And below, we demonstrate that TorchDynamo does not break the graph on the model we used above for demonstrating speedups.

```{code-cell}
opt_model = torch.compile(init_model(), fullgraph=True)
print(opt_model(generate_data(16)[0]))
```

We can use `torch.export` (from PyTorch 2.1+) to extract a single, exportable FX graph from the input PyTorch program. The exported graph is intended to be run on different (i.e. Python-less) environments. One important restriction is that the `torch.export` does not support graph breaks. Please check [this tutorial](https://pytorch.org/tutorials/intermediate/torch_export_tutorial.html) for more details on `torch.export`.

# Conclusion

In this tutorial, we introduced `torch.compile` by covering basic usage, demonstrating speedups over eager mode, comparing to previous PyTorch compiler solutions, and briefly investigating TorchDynamo and its interactions with FX graphs. We hope that you will give `torch.compile` a try!
