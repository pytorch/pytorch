# torch.func._random

```{eval-rst}
.. currentmodule:: torch.func._random
```
```{eval-rst}
.. py:module:: torch.func._random
```

```{warning}
   These APIs are a prototype and may change or be removed without notice. The module is private and not covered
   by PyTorch's backward compatibility policy. Read more about feature classification at:
   https://pytorch.org/blog/pytorch-feature-classification-changes/#prototype
```

``torch.func._random`` provides JAX-style stateless PRNG APIs: randomness is a pure function of an explicit key
tensor rather than of global generator state. The module is not imported by ``import torch``; import it explicitly
with ``import torch.func._random`` before use.

These APIs do not yet support {func}`torch.func.vmap`. To generate a batch of independent samples, pass a batch of
keys with shape ``(*batch, K)`` (e.g. from {func}`split`) instead.

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

     key
     split
     fold_in
     uniform
     uniform_
     normal
     normal_
     randint
     randint_
     bits
     bits_
```
