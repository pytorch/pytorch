# torch.func._random

```{eval-rst}
.. currentmodule:: torch.func._random
```
```{eval-rst}
.. py:module:: torch.func._random
```

```{warning}
   These APIs are a prototype and may change or be removed without notice.
```

``torch.func._random`` provides JAX-style stateless PRNG APIs: randomness is a pure function of an explicit key
tensor rather than of global generator state.

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
