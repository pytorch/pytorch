---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  execution_timeout: 30
  execution_show_tb: True
  merge_streams: True
---

(cond)=

# Control Flow - Cond

`torch.cond` is a structured control flow operator. It can be used to specify if-else like control flow
and can logically be seen as implemented as follows.

```python
def cond(
    pred: Union[bool, torch.Tensor],
    true_fn: Callable,
    false_fn: Callable,
    operands: Tuple[torch.Tensor]
):
    if pred:
        return true_fn(*operands)
    else:
        return false_fn(*operands)
```

Its unique power lies in its ability of expressing **data-dependent control flow**: it lowers to a conditional
operator (`torch.ops.higher_order.cond`), which preserves predicate, true function and false functions.
This unlocks great flexibility in writing and deploying models that change model architecture based on
the **value** or **shape** of inputs or intermediate outputs of tensor operations.

```{warning}
`torch.cond` is a prototype feature in PyTorch. It has limited support for input and output types.
Please look forward to a more stable implementation in a future version of PyTorch.
Read more about feature classification at: https://pytorch.org/blog/pytorch-feature-classification-changes/#prototype
```

## Examples

Below is an example that uses cond to branch based on input shape:

```{code-cell}
import torch

def true_fn(x: torch.Tensor):
    return x.cos()

def false_fn(x: torch.Tensor):
    return x.sin()

class DynamicShapeCondPredicate(torch.nn.Module):
    """
    A basic usage of cond based on dynamic shape predicate.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cond(x.shape[0] > 4, true_fn, false_fn, (x,))

dyn_shape_mod = DynamicShapeCondPredicate()
```

We can eagerly run the model and expect the results vary based on input shape:

```{code-cell}
inp = torch.randn(3)
inp2 = torch.randn(5)
print(dyn_shape_mod(inp), false_fn(inp))
print(dyn_shape_mod(inp2), true_fn(inp2))
```

We can export the model for further transformations and deployment. This gives
us an exported program as shown below:

```{code-cell}
inp = torch.randn(4, 3)
ep = torch.export.export(
    DynamicShapeCondPredicate(),
    (inp,),
    dynamic_shapes={"x": {0: torch.export.Dim.DYNAMIC}}
)
print(ep)
```

Notice that `torch.cond` is lowered to `torch.ops.higher_order.cond`, its predicate becomes a Symbolic expression over the shape of input,
and branch functions becomes two sub-graph attributes of the top level graph module.

Here is another example that showcases how to express a data-dependent control flow:

```{code-cell}
def true_fn(x: torch.Tensor):
    return x.cos() + x.sin()

def false_fn(x: torch.Tensor):
    return x.sin()

class DataDependentCondPredicate(torch.nn.Module):
    """
    A basic usage of cond based on data dependent predicate.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cond(x.sum() > 4.0, true_fn, false_fn, (x,))

inp = torch.randn(4, 3)
ep = torch.export.export(DataDependentCondPredicate(), (inp,), dynamic_shapes={"x": {0: torch.export.Dim.DYNAMIC}})
print(ep)
```

## Invariants of torch.ops.higher_order.cond

There are several useful invariants for `torch.ops.higher_order.cond`:

- For predicate:
    - Dynamicness of predicate is preserved (e.g. `gt` shown in the above example)
    - If the predicate in user-program is constant (e.g. a python bool constant), the `pred` of the operator will be a constant.

- For branches:
    - The input and output signature will be a flattened tuple.
    - They are `torch.fx.GraphModule`.
    - Closures in original function becomes explicit inputs. No closures.
    - No mutations on inputs or globals are allowed.

- For operands:
    - It will also be a flat tuple.

- Nesting of `torch.cond` in user program becomes nested graph modules.

## API Reference

```{eval-rst}
.. autofunction:: torch._higher_order_ops.cond.cond
```
