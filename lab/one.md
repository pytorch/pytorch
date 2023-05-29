## Part one - vmap operations as torch.mul operations

I have rewritten `vmap` operations as `torch.mul` operations here.

```python3
import torch
from torch import vmap

B = 2
B1 = 3

op = torch.mul
```

### Example a

```python3
x = torch.randn(B)
y = torch.randn(B)

torch.allclose(vmap(op)(x, y), op(x, y))
```

### Example b

Idea: If a tensor has batch dimension as None, give it a batch dimension at 0.
If a tensor has only batch dimension, broadcast it to match dimension of the other operand.
Finally, both operands should have same batch dimension and data dimensions before applying the operation.

```python3
x = torch.randn(3)
y = torch.randn(B)

vmap_result = vmap(op, in_dims=(None, 0))(x, y)
# x has batch dimension as None, give it a batch dimension at 0
x = x.unsqueeze(0)
# y only has batch dimension, broadcast it and give it a data dimension to match dimension of x
y = y.unsqueeze(1)
torch.allclose(torch.mul(x, y), vmap_result)
```

### Example c

```python3
x = torch.randn(4, 3)
y = torch.randn(3, B)

c = vmap(op, in_dims=(None, 1))(x, y)

in_dims = (None, 1)

# make batch dimension as first dimension
y1 = torch.moveaxis(y, in_dims[1], 0)

# add batch dimension to x since x has no batch dimension
# broadcast y to match data dimension of x
torch.allclose(torch.mul(x.unsqueeze(0), y1.unsqueeze(1)), c)
```

### Example d
We compose two vmap operations here. First we change dimension of x, y corresponsing to the external vmap. Since external vmap has `in_dims` of (None, 0), we add a batch dimension to x via `unsqueeze(0)` and then add a data dimension to y to match dimension of x. In the next step, we operate in the same lines for the internal vmap - adding a batch dimension to y and data dimension to x.

```python3
x = torch.randn(B)
y = torch.randn(B1)
c = vmap(vmap(op, (0, None)), (None, 0))(x, y)

torch.allclose(c, torch.mul(x.unsqueeze(0).unsqueeze(1), y.unsqueeze(1).unsqueeze(0)))
```
