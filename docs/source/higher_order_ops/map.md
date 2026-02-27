---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  execution_timeout: 30
  execution_show_tb: True
  merge_streams: True
---

(map)=

# Control Flow - Map

`torch.map` is a structured control flow operator that applies a function over the leading dimension of
input tensors. It can logically be seen as implemented as follows:

```python
def map(
    f: Callable[[PyTree, ...], PyTree],
    xs: Union[PyTree, torch.Tensor],
    *args,
):
    out = []
    for idx in range(xs.size(0)):
        xs_sliced = xs.select(0, idx)
        out.append(f(xs_sliced, *args))
    return torch.stack(out)
```

```{warning}
`torch._higher_order_ops.map` is a prototype feature in PyTorch. You may run into miscompiles.
Read more about feature classification at: https://pytorch.org/blog/pytorch-feature-classification-changes/#prototype
```

## Examples

Below is an example that uses map to apply a function over a batch:

```{code-cell}
import torch
from torch._higher_order_ops import map

def f(x):
    return x.sin() + x.cos()

xs = torch.randn(3, 4, 5)  # batch of 3 tensors, each 4x5
# Applies f to each of the 3 slices
result = map(f, xs)  # returns tensor of shape [3, 4, 5]
print(result)
```

We can export the model with map for further transformations and deployment.
This example uses dynamic shapes to allow variable batch size:

```{code-cell}
class MapModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        def body_fn(x):
            return x.sin() + x.cos()

        return map(body_fn, xs)

mod = MapModule()
inp = torch.randn(3, 4)
ep = torch.export.export(mod, (inp,), dynamic_shapes={"xs": {0: torch.export.Dim.DYNAMIC}})
print(ep)
```

Notice that `torch.map` is lowered to `torch.ops.higher_order.map_impl`, and the body function becomes a
sub-graph attribute of the top-level graph module.

## Restrictions

- Mapped `xs` can only consist of tensors.

- Leading dimensions of all tensors in `xs` must be consistent and non-zero.

- The body function must not mutate inputs.

## API Reference

```{eval-rst}
.. autofunction:: torch._higher_order_ops.map.map
```
