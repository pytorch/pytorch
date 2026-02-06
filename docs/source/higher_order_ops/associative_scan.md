---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  execution_timeout: 30
  execution_show_tb: True
  merge_streams: True
---

(associative_scan)=

# Control Flow - Associative Scan

`torch.associative_scan` is a structured control flow operator that performs an inclusive scan with an
associative combine function. It can logically be seen as implemented as follows:

```python
def associative_scan(
    combine_fn: Callable[[pytree.PyTree, pytree.PyTree], pytree.PyTree],
    xs: pytree.PyTree,
    dim: int,
    reverse: bool = False,
) -> pytree.PyTree:
    result = []
    carry = xs.select(dim, 0)
    result.append(carry)
    for i in range(1, xs.size(dim)):
        carry = combine_fn(carry, xs.select(dim, i))
        result.append(carry)
    return torch.stack(result, dim=dim)
```

Because `combine_fn` is required to be associative, the computation can be parallelized using a
tree-reduction algorithm rather than executed sequentially. This enables efficient GPU implementations
for operations like cumulative sums, products, or other associative accumulations.

```{warning}
`torch.associative_scan` is a prototype feature in PyTorch. You may run into miscompiles.
Read more about feature classification at: https://pytorch.org/blog/pytorch-feature-classification-changes/#prototype
```

## Examples

Below is an example that uses `associative_scan` to compute a cumulative sum:

```{code-cell}
import torch
from torch._higher_order_ops.associative_scan import associative_scan

def add(x: torch.Tensor, y: torch.Tensor):
    return x + y

xs = torch.arange(1, 5, dtype=torch.float32)  # [1, 2, 3, 4]
cumsum = associative_scan(add, xs, dim=0, combine_mode="generic")
print(cumsum)
```

Here is an example computing a cumulative product:

```{code-cell}
def mul(x: torch.Tensor, y: torch.Tensor):
    return x * y

xs = torch.arange(1, 5, dtype=torch.float32)  # [1, 2, 3, 4]
cumprod = associative_scan(mul, xs, dim=0, combine_mode="generic")
print(cumprod)
```

We can export the model with associative_scan for further transformations and deployment.
This example uses dynamic shapes to allow variable sequence length:

```{code-cell}
class AssociativeScanModule(torch.nn.Module):
    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        def combine_fn(x, y):
            return x + y

        return associative_scan(combine_fn, xs, dim=0, combine_mode="piecewise")

mod = AssociativeScanModule()
inp = torch.randn(5, 3, device="cuda")
dim_seq = torch.export.Dim("seq", min=2)
ep = torch.export.export(mod, (inp,), dynamic_shapes={"xs": {0: dim_seq}})
print(ep)
```

Notice that `torch.associative_scan` is lowered to `torch.ops.higher_order.associative_scan`, and the
combine function becomes a sub-graph attribute of the top-level graph module.

## Restrictions

- `combine_fn` must be associative: `combine_fn(combine_fn(a, b), c) == combine_fn(a, combine_fn(b, c))`.

- `combine_fn` must not in-place mutate its inputs.

- `combine_fn` must not reference variables from an outer scope (closures are not supported).

- `combine_fn`'s output cannot alias any of the inputs.

## API Reference

```{eval-rst}
.. autofunction:: torch._higher_order_ops.associative_scan.associative_scan
```
