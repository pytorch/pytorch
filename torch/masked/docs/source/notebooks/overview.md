---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.7
kernelspec:
  display_name: Python 3.9.7 ('pytorch_env')
  language: python
  name: python3
---

+++ {"id": "JF8WDyTRq0Hn"}

# Overview of MaskedTensors

+++

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pytorch/maskedtensor/blob/main/docs/source/notebooks/overview.ipynb)

```{code-cell} ipython3
:id: XHPLFh2Qf4ZL

import torch
import numpy as np
from maskedtensor import masked_tensor
from maskedtensor import as_masked_tensor
```

+++ {"id": "aSD_zzXcWLvK"}

## Basic masking semantics

+++

### MaskedTensor vs NumPy's MaskedArray semantics

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: FrNr_-yfjcsr
outputId: b1b03759-1e8c-4cf3-bdae-0be4900ee8ea
---
# First example of addition
data = torch.arange(5.)
mask = torch.tensor([True, True, False, True, False])
m0 = masked_tensor(data, mask)
m1 = masked_tensor(data, ~mask)

print(m0)
print(m1)
print(torch.cos(m0))
print(m0 + m0)

try:
  # For now the masks must match. We treat them like shapes.
  # We can relax this later on, but should have a good reason for it.
  # We'll revisit this once we have reductions.
  print(m0 + m1)
except ValueError as e:
  print(e)
```

+++ {"id": "RMHT1RebL8PR"}

NumPy's MaskedArray implements intersection semantics here. If one of two elements are masked out the resulting element will be masked out as well. Note that MaskedArray's factory function inverts the mask (similar to torch.nn.MHA). For MaskedTensor we'd apply the logical_and operator to both masks during a binary operation to get the semantics NumPy has. Since NumPy stores the inverted mask they [apply the logical_or operator](https://github.com/numpy/numpy/blob/68299575d8595d904aff6f28e12d21bf6428a4ba/numpy/ma/core.py#L1016-L1024). But to repeat this point we suggest to not support addition between MaskedTensors with masks that don't match. See the section on reductions for why we should have good reasons for this.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: -tmWxZ9NKEgE
outputId: 5997cae5-f4c4-4422-b6d1-c80fd59c4001
---
npm0 = np.ma.masked_array(data.numpy(), (~mask).numpy())
npm1 = np.ma.masked_array(data.numpy(), (mask).numpy())
print("npm0:       ", npm0)
print("npm1:       ", npm1)
print("npm0 + npm1:", npm0 + npm1)
```

+++ {"id": "SljV_QCfMEu7"}

MaskedTensor also supports these semantics by giving access to the masks and conveniently converting a MaskedTensor to a Tensor with masked values filled in with a particular value.

NumPy of course has the opportunity to avoid addition altogether in this case by check whether any results are not masked, but [chooses not to](https://github.com/numpy/numpy/blob/68299575d8595d904aff6f28e12d21bf6428a4ba/numpy/ma/core.py#L1013). Presumably it's more expensive to allreduce the mask every time to avoid the binary addition of the data in this case.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: D-TCtDEJJzeV
outputId: de7bd486-4a0f-4645-8c38-63b5b16eb55f
---
m0t = m0.to_tensor(0)
m1t = m1.to_tensor(0)

m2t = masked_tensor(m0t + m1t, m0.mask() & m1.mask())
print(m0t)
print(m1t)
print(m2t)
```

### MaskedTensor reduction semantics

+++ {"id": "zAbRZK3QZgge"}

Example of printing a 2d MaskedTensor and setup for reductions below

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: y9tv_SP8oI7Z
outputId: 006fbfc2-a59c-4205-919e-d866203aa840
---
data = torch.randn(8, 3).mul(10).int().float()
mask = torch.randint(2, (8, 3), dtype=torch.bool)
print(data)
print(mask)
m = masked_tensor(data, mask)
print(m)
```

+++ {"id": "8fUbL3yAZqZF"}

Reduction semantics based on https://github.com/pytorch/rfcs/pull/27

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: M2wGD5hRpVDV
outputId: f5867cef-70b6-427b-c4bc-8fc066a98469
---
print("sum:", torch.sum(m, 1))
print("mean:", torch.mean(m, 1))
print("prod:", torch.prod(m, 1))
print("min:", torch.amin(m, 1))
print("max:", torch.amax(m, 1))
```

+++ {"id": "TyLv4Nf4dVtS"}

Now that we have reductions, let's revisit as to why we'll probably want to have a good reason to allow addition of MaskedTensors with different masks.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: xMPyMU87fICJ
outputId: 910e01fd-0da9-4e56-f91e-4a2087b4377b
---
data0 = torch.arange(10.).reshape(2, 5)
data1 = torch.arange(10.).reshape(2, 5) + 10
mask0 = torch.tensor([[True, True, False, False, False], [False, False, False, True, True]])
mask1 = torch.tensor([[False, False, False, True, True], [True, True, False, False, False]])

npm0 = np.ma.masked_array(data0.numpy(), (mask0).numpy())
npm1 = np.ma.masked_array(data1.numpy(), (mask1).numpy())
print("\nnpm0:\n", npm0)
print("\nnpm1:\n", npm1)
print("\n(npm0 + npm1).sum(0):\n", (npm0 + npm1).sum(0))
print("\nnpm0.sum(0) + npm1.sum(0):\n", (npm0.sum(0) + npm1.sum(0)))
print("\n(data0 + data1).sum(0):\n", (data0 + data1).sum(0))
print("\n(data0 + data1).sum(0):\n", (data0.sum(0) + data1.sum(0)))
```

+++ {"id": "FjSyaqqKgvYh"}

Sum and addition should be associative. However with NumPy's semantics we allow them not to be. Instead of allowing these semantics, at least in the case of addition and sum, we could ask the user to fill the MaskedTensor's undefined elements with 0 values or as in the MaskedTensor addition examples above be very specific about the semantics used.

While it's obviously possible to support this, we think we should cover other operators first and really make sure we can't avoid this behavior via other means.

+++ {"id": "ADp2guJ6ZlMo"}

### Indexing and Advanced Indexing

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: c3ESKLl0pYEj
outputId: 41f3d277-177a-484a-9ae1-f89ef552a8d9
---
data = torch.randn(4, 5, 3).mul(5).float()
mask = torch.randint(2, (4, 5, 3), dtype=torch.bool)
m = masked_tensor(data, mask)
print(m)
```

+++ {"id": "LgHrnMB7ZtOo"}

Example of indexing and advanced indexing

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: RWGG2AA1r_R-
outputId: 20ad0002-bb61-4a3a-c700-541ce229718d
---
print(m[0])
print(m[torch.tensor([0, 2])])
print(m[m.mask()])
```

### MaskedTensor gradient examples

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: qEejo-1-sBMw
outputId: edd0d800-2c7c-4ea3-f67f-bbce070fc044
---
torch.manual_seed(22)
# Sum needs custom autograd, since the mask of the input should be maintained
data = torch.randn(2, 2, 3).mul(5).float()
mask = torch.randint(2, (2, 2, 3), dtype=torch.bool)
m = masked_tensor(data, mask, requires_grad=True)
print(m)
s = torch.sum(m)
print("s: ", s)
s.backward()
print("m.grad: ", m.grad)

# sum needs to return a scalar MaskedTensor because the input might be fully masked
data = torch.randn(2, 2, 3).mul(5).float()
mask = torch.zeros(2, 2, 3, dtype=torch.bool)
m = masked_tensor(data, mask, requires_grad=True)
print("\n", m)
s = torch.sum(m)
print("s: ", s)
s.backward()
print("m.grad: ", m.grad)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: 3N1Y7QFJMrdz
outputId: da5f6016-23cb-46e2-d949-ea145aea0037
---
# Grad of multiplication of MaskedTensor and Tensor
x = masked_tensor(torch.tensor([3.0, 4.0]), torch.tensor([True, False]), requires_grad=True)
print("x:\n", x)
y = torch.tensor([2., 1.]).requires_grad_()
print("y:\n", y)
# The mask broadcast in the sense that the result is masked.
# In general a MaskedTensor is considered a generalization of Tensor's shape.
# The mask is a more complex, higher dimensional shape and thus the Tensor
# broadcasts to it. I'd love to find a more rigorous definition of this.
z = x * y
print("x * y:\n", z)
z.sum().backward()
print("\nx.grad: ", x.grad)
# The regular torch.Tensor now has a MaskedTensor grad
print("y.grad: ", y.grad)
```
