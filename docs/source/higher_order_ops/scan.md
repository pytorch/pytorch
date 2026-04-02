---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  execution_timeout: 30
  execution_show_tb: True
  merge_streams: True
---

(scan)=

# Control Flow - Scan

`torch.scan` is a structured control flow operator that performs an inclusive scan with a combine function.
It is commonly used for cumulative operations like cumsum, cumprod, or more general recurrences.
It can logically be seen as implemented as follows:

```python
def scan(
    combine_fn: Callable[[PyTree, PyTree], tuple[PyTree, PyTree]],
    init: PyTree,
    xs: PyTree,
    *,
    dim: int = 0,
    reverse: bool = False,
) -> tuple[PyTree, PyTree]:
    carry = init
    ys = []
    for i in range(xs.size(dim)):
        x_slice = xs.select(dim, i)
        carry, y = combine_fn(carry, x_slice)
        ys.append(y)
    return carry, torch.stack(ys)
```

```{warning}
`torch.scan` is a prototype feature in PyTorch. You may run into miscompiles.
Read more about feature classification at:
https://pytorch.org/blog/pytorch-feature-classification-changes/#prototype
```

## Examples

Below is an example that uses scan to compute a cumulative sum:

```{code-cell}
import torch
from torch._higher_order_ops import scan

def add(carry: torch.Tensor, x: torch.Tensor):
    next_carry = carry + x
    y = next_carry.clone()  # clone to avoid output-output aliasing
    return next_carry, y

init = torch.zeros(1)
xs = torch.arange(5, dtype=torch.float32)

final_carry, cumsum = scan(add, init=init, xs=xs)
print(final_carry)
print(cumsum)
```

We can export the model with scan for further transformations and deployment.
This example uses dynamic shapes to allow variable sequence length:

```{code-cell}
class ScanModule(torch.nn.Module):
    def forward(self, xs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        def combine_fn(carry, x):
            next_carry = carry + x
            return next_carry, next_carry.clone()

        init = torch.zeros_like(xs[0])
        return scan(combine_fn, init=init, xs=xs)

mod = ScanModule()
inp = torch.randn(5, 3)
ep = torch.export.export(mod, (inp,), dynamic_shapes={"xs": {0: torch.export.Dim.DYNAMIC}})
print(ep)
```

Notice that the combine function becomes a sub-graph attribute of the top-level graph module.

## Restrictions

- `combine_fn` must return tensors with the same metadata (shape, dtype) for `next_carry` as `init`.

- `combine_fn` must not in-place mutate its inputs. A clone before mutation is required.

- `combine_fn` must not mutate Python variables (e.g., list/dict) created outside the function.

- `combine_fn`'s output cannot alias any of the inputs. A clone is required.

## API Reference

```{eval-rst}
.. autofunction:: torch._higher_order_ops.scan.scan
```
