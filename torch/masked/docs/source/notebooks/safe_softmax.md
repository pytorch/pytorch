---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.7
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Safe Softmax

+++

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pytorch/maskedtensor/blob/main/docs/source/notebooks/safe_softmax.ipynb)

+++

## Motivation

+++

One of the issues that commonly comes up is the necessity for a safe softmax -- that is, if there is an entire batch that is "masked out" or consists entirely of padding (which in the softmax case translates to being set to `-inf`, then this will result in NaNs, which can lead to training divergence. For more detail on why this functionality is helpful, please find [Issue 55056 - Feature Request for Safe Softmax](https://github.com/pytorch/pytorch/issues/55056).

Luckily, MaskedTensor has solved this issue already.

```{code-cell} ipython3
import torch
from maskedtensor import masked_tensor
```

```{code-cell} ipython3
data = torch.randn(3, 3)
mask = torch.tensor([
    [True, False, False],
    [True, False, True],
    [False, False, False]
])
x = data.masked_fill(~mask, float('-inf'))

m = masked_tensor(data, mask)
```

**PyTorch result**:

```{code-cell} ipython3
x.softmax(0)
```

**MaskedTensor result**:

```{code-cell} ipython3
m.softmax(0)
```
