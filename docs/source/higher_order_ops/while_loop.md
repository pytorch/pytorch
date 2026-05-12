---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  execution_timeout: 30
  execution_show_tb: True
  merge_streams: True
---

(while_loop)=

# Control Flow - While Loop

`torch.while_loop` is a structured control flow operator that runs a body function while a condition is true.
It can logically be seen as implemented as follows:

```python
def while_loop(
    cond_fn: Callable[..., bool],
    body_fn: Callable[..., tuple],
    carried_inputs: tuple,
):
    val = carried_inputs
    while cond_fn(*val):
        val = body_fn(*val)
    return val
```

```{warning}
`torch.while_loop` is a prototype feature in PyTorch. It has limited support for input and output types.
Please look forward to a more stable implementation in a future version of PyTorch.
Read more about feature classification at: https://pytorch.org/blog/pytorch-feature-classification-changes/#prototype
```

## Examples

Below is a basic example that uses while_loop to iterate until a condition is met:

```{code-cell}
import torch
from torch._higher_order_ops import while_loop

class M(torch.nn.Module):

    def cond_fn(self, iter_count, x):
        return iter_count.sum() > 0

    def body_fn(self, iter_count, x):
        return iter_count - 1, x * 2

    def forward(self, init_iter, init_x):
        final_iter, final_x = while_loop(self.cond_fn, self.body_fn, (init_iter, init_x))
        return final_iter, final_x

m = M()
```

We can eagerly run the model and expect the results vary based on input shape:

```{code-cell}
_, final_x = m(torch.tensor([3]), torch.ones(3))
assert torch.equal(final_x, torch.ones(3) * 2**3)

_, final_x = m(torch.tensor([10]), torch.ones(3))
assert torch.equal(final_x, torch.ones(3) * 2**10)
```

We can export the model for further transformations and deployment. This gives us an exported program that preserves the while_loop structure:

```{code-cell}
ep = torch.export.export(M(), (torch.tensor([10]), torch.ones(3)))
print(ep)
```

Notice that both the condition and body functions become sub-graph attributes of
the top-level graph module.

## Restrictions

- `body_fn` must return tensors or integers with the same metadata (shape, dtype) as inputs.

- `body_fn` and `cond_fn` must not in-place mutate the `carried_inputs`. A clone before mutation is required.

- `body_fn` and `cond_fn` must not mutate Python variables (e.g., list/dict) created outside the function.

- `body_fn` and `cond_fn`'s output cannot alias any of the inputs. A clone is required.


## API Reference

```{eval-rst}
.. autofunction:: torch._higher_order_ops.while_loop.while_loop
```
