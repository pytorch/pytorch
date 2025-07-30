# Debugging with `tlparse` and `TORCH_LOGS=dynamic`

`tlparse` is a tool used for analyzing and understanding the compilation
process in PyTorch, particularly when dealing with dynamic shapes. It helps
identify where guards and specializations occur in your code.

`TORCH_LOGS=dynamic` is an environment variable setting that enables detailed
logging of dynamic shape operations, providing insights into how symbolic
shapes are handled during execution.

This section will guide you through using `tlparse` and `TORCH_LOGS=dynamic` to
troubleshoot dynamic shape issues in your code, including debugging
specialization, guards, and more.

# Debugging Specialization

In the following example, `x.shape[0]` is dynamic but becomes specialized due to multiplication:

```python
import torch

@torch.compile
def fn(x, y):
    return x * y

x = torch.randn(5)
y = torch.randn(5)
torch._dynamo.decorators.mark_dynamic(x, 0)

fn(x, y)
```

By using `TORCH_LOGS=dynamic`, you can observe this specialization in the logs:

```xml
TORCH_LOGS=dynamic python tl.py
I0721 11:10:00.950000 845259 torch/fx/experimental/symbolic_shapes.py:3776] [0/0] create_env
I0721 11:10:01.030000 845259 torch/fx/experimental/symbolic_shapes.py:5117] [0/0] create_symbol s77 = 5 for L['x'].size()[0] [2, int_oo] return x * y  # tl.py:5 in fn (_dynamo/variables/builder.py:3466 in <lambda>), for more info run with TORCHDYNAMO_EXTENDED_DEBUG_CREATE_SYMBOL="s77" or to suppress this message run with TORCHDYNAMO_EXTENDED_ADVICE="0"
I0721 11:10:01.038000 845259 torch/fx/experimental/symbolic_shapes.py:7211] [0/0] eval Eq(s77, 5) [guard added] return x * y  # tl.py:5 in fn (_subclasses/fake_impls.py:922 in infer_size), for more info run with TORCHDYNAMO_EXTENDED_DEBUG_GUARD_ADDED="Eq(s77, 5)"
```

The line `eval Eq(s77, 5) [guard added] return x * y # tl.py:5` indicates the specialization.

## Debugging Guards

Consider the following code, which may cause recompilations due to dynamic
shapes:

```python
import torch

@torch.compile
def fn(x, y):
    if x.shape[0] < 10:
        return x * y

x = torch.randn(5)
y = torch.randn(5)
torch._dynamo.decorators.mark_dynamic(x, 0)
torch._dynamo.decorators.mark_dynamic(y, 0)

fn(x, y)
```

To identify where dynamic shape guards originate, use `tlparse`. Here is an example tlparse output:

```{image} _static/img/dynamic_shapes/tlparse9_debugging_guards.png
```

By clicking on the `dynamo_cpp_guards` link, you can view all guards from the compilation, including the symbolic shape guard `L['x'].size()[0] <= 9`.

Astute readers will notice the 0/1 specialization where we guard on `L['x'].size()[0] >= 2`. By modifying the code to use unbacked symbols, this guard is removed:

```python
import torch

@torch.compile
def fn(x, y):
    # Necessary runtime assert since we can't guard on unbacked
    torch._check(x.shape[0] < 10)
    if x.shape[0] < 10:
        return x * y

x = torch.randn(5)
y = torch.randn(5)
torch._dynamo.decorators.mark_unbacked(x, 0)
torch._dynamo.decorators.mark_unbacked(y, 0)

fn(x, y)
```

Now, this compiled region can be used for inputs of size 0 and 1:

```{image} _static/img/dynamic_shapes/tlparse10_debugging_guards_unbacked.png
```

```{seealso}
* {ref}`practical_guide_reduce_recompilation`
* {ref}`troubleshooting_guardondatadependentsymnode_errors`
```
