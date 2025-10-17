# Aliases in torch

The following are aliases in ``torch`` to their counterparts in the nested namespaces
in which they are defined. Feel free to use either the top-level version in ``torch``
(e.g. ``torch.broadcast_tensors()``) or the nested version ``torch.functional.broadcast_tensors()``.

```{eval-rst}
.. automodule:: torch.functional
.. currentmodule:: torch.functional
.. autosummary::
   :toctree: generated
   :nosignatures:

    align_tensors
    atleast_1d
    atleast_2d
    atleast_3d
    block_diag
    broadcast_shapes
    broadcast_tensors
    cartesian_prod
    cdist
    chain_matmul
    einsum
    lu
    meshgrid
    norm
    split
    stft
    tensordot
    unique
    unique_consecutive
    unravel_index
```
