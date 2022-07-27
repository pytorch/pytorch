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

# Implemented missing torch.nan\* operators

+++

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pytorch/maskedtensor/blob/main/docs/source/notebooks/nan_operators.ipynb)

```{code-cell} ipython3
import torch
from maskedtensor import masked_tensor
```

### [Issue 21987](https://github.com/pytorch/pytorch/issues/21987)

+++

This issue was closed by inclusion into [Issue 61474 - Implement missing torch.nan\* operators](https://github.com/pytorch/pytorch/issues/61474). This proposes an alternative, which is to use masked tensors instead of introducing additional operators. Since nanmean [has already landed](https://github.com/pytorch/pytorch/issues/21987), we can use it as a comparison point.

```{code-cell} ipython3
y = torch.arange(32).float()
x = y * y.fmod(4)
x = x.masked_fill(x == 0, float('nan'))
print(x)
print(torch.nanmean(x))
print(torch.mean(masked_tensor(x, ~torch.isnan(x))))
```

MaskedTensor can further support reduction when fully masked out, as would be the case when a given Tensor is completetely nan. nanmean on the other hand returns nan when the input is entirely nan.

```{code-cell} ipython3
x = torch.empty(32)
x.fill_(float('nan'))
print(x)
print(torch.nanmean(x))
print(torch.mean(masked_tensor(x, ~torch.isnan(x))))
```

Further [some users](https://github.com/pytorch/pytorch/issues/63870) already want to use nan reductions to encode masked semantics.
