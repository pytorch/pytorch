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

```{eval-rst}
.. automodule:: torch.serialization
.. currentmodule:: torch.serialization
.. autosummary::
   :toctree: generated
   :nosignatures:

    check_module_version_greater_or_equal
    default_restore_location
    load
    location_tag
    mkdtemp
    normalize_storage_type
    save
    storage_to_tensor_type
    validate_cuda_device
    validate_hpu_device
```
