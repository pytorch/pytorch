(dynamic_shapes_overview)=
# Dynamic Shapes

This section explains how to work with dynamic shapes in PyTorch, including how to debug and fix common errors, implement support for dynamic shapes in operators, and understand the underlying mechanisms.

Dynamic shapes allow PyTorch models to handle inputs with varying dimensions without recompilation. This enables more flexible models that can process different batch sizes, sequence lengths, or image dimensions in a single compiled artifact. Dynamic shapes work by symbolically tracing tensor dimensions rather than using concrete values, creating a computation graph that adapts to different input shapes at runtime. By default, PyTorch assumes all input shapes to be static.

Typically, deep learning compilers only support static shapes, requiring recompilation for input shape changes. While this approach covers many use cases, there are situations where this is insufficient:

- **Variable Dimensions** - Batch sizes or sequence lengths vary, such as in adaptive batching.
- **Data-Dependent Outputs** - Models produce outputs based on input data, like variable bounding boxes in detection models.
- **Sparse Representations** - Processing depends on data-varying sparse structures, such as in sparse tensors, jagged tensors, and graph neural networks.

Dynamic shapes do not support dynamic rank programs, programs which input tensors change in dimensionality, as this is uncommon and unnecessarily complex.

# Basic example

Here is a simple example of using dynamic shapes:

```python
# test.py
torch._logging.set_logs(graph_breaks=True, graph_code=True)

@torch.compile(dynamic=False)
def foo(x):
    for _ in range(10):
        x = x.sin()
    return x

for i in range(2, 100):
    x = torch.randn(i)
    # torch._dynamo.decorators.mark_dynamic(x, 0) # uncomment out for dynamic shapes
    foo(x)
```

This code defines a function `foo` that applies the sine operation to a tensor `x`
ten times. It iterates over tensors of increasing size, from 2 to 99. If we
use `@torch.compile(dynamic=False)`, this code will create 98 graphs (from 2 to 99),
recompiling the input shape every time, which would take a significant amount of time
and resources. However, if we uncomment `torch._dynamo.decorators.mark_dynamic(x, 0)`,
recompilation will be skipped and only one graph will be created, significantly
decreasing compilation time and the use of resources.

```{toctree}
:titlesonly:
dynamic_shapes_core_concepts
dynamic_shapes_beyond_the_basics
dynamic_shapes_troubleshooting
```
