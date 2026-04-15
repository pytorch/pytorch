"""Stateless PRNG APIs.

These are experimental and subject to change without notice.
Access via ``torch.func._random``.
"""

from collections.abc import Sequence

import torch


def key(
    seed: int, impl: str = "philox4x32-10", device: torch.device | None = None
) -> torch.Tensor:
    r"""Create a PRNG key from a seed.

    A key is a tensor that encodes the state needed to deterministically
    produce random values. Keys are consumed by generation functions to produce
    reproducible random tensors without any global state. The internal
    representation of the key depends on the chosen PRNG algorithm.

    Args:
        seed (int): The seed value for the PRNG.
        impl (str): PRNG algorithm. Currently only ``"philox4x32-10"`` is
            supported.
        device (:class:`torch.device`, optional): The desired device for the
            returned key. Default: ``cpu``.

    Returns:
        A tensor representing the PRNG key.

    .. note::

        For the ``"philox4x32-10"`` algorithm, the key is a uint64 tensor of
        shape ``(2,)`` encoding a ``(seed, offset)`` pair. The offset determines
        the starting position in the Philox output stream.

    Example::

        >>> key = torch.func._random.key(42, device="cuda")  # doctest: +SKIP
    """
    if impl != "philox4x32-10":
        raise NotImplementedError(f"key() does not support PRNG impl '{impl}'")

    # (seed, offset)
    return torch.tensor([seed, 0], dtype=torch.uint64, device=device)


def split(key: torch.Tensor, num: int = 2) -> torch.Tensor:
    r"""Split a PRNG key into ``num`` new independent keys.

    Each returned key produces a different, deterministic random sequence.
    This is the primary mechanism for deriving multiple independent keys from
    a single parent key without mutating any state.

    Supports batched keys: if ``key`` has shape ``(*batch, K)``, each key in the
    batch is split independently and the result has shape ``(num, *batch, K)``.

    Args:
        key (Tensor): A PRNG key returned by :func:`key`, :func:`split`, or
            :func:`fold_in`.
        num (int): Number of keys to produce. Default: ``2``.

    Returns:
        A tensor of shape ``(num, *key.shape)`` containing the derived keys.

    Example::

        >>> key = torch.func._random.key(42, device="cuda")  # doctest: +SKIP
        >>> k1, k2 = torch.func._random.split(key)  # doctest: +SKIP
    """
    return torch.ops.aten._philox_key_split(key, num)


def unbind(
    key: torch.Tensor,
    shape: tuple,
    splits: tuple,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    r"""Unbind a key into a grid of sub-keys for tiled generation.

    Each returned sub-key generates a contiguous tile of the output that the
    original key would produce. Unlike :func:`split`, which derives
    statistically independent keys, ``unbind`` preserves the relationship
    between sub-keys and the parent key: generating with all sub-keys and
    reassembling the tiles exactly reconstructs the full output::

        keys = unbind(key, (100,), (10,))
        full = uniform(key, (100,))
        tile_size = 100 // 10  # = 10
        tiled = torch.cat([uniform(keys[i], (tile_size,)) for i in range(10)])
        assert torch.equal(full, tiled)

    For N-D, each tile key is a batched key with per-row sub-keys. The tile
    shape is ``shape[i] // splits[i]`` along each dimension, and
    ``uniform(keys[t0, ..., t_{n-1}], tile_shape)`` reproduces the
    corresponding sub-block of the full generation.

    Args:
        key (Tensor): A PRNG key of shape ``(..., 2)`` with dtype ``torch.uint64``.
        shape (tuple): Shape of the full tensor to be generated.
        splits (tuple): Number of tiles along each dimension. Must evenly
            divide the corresponding element of ``shape``.
        dtype (:class:`torch.dtype`, optional): The dtype that will be generated.
            Needed because float64 consumes 2 Philox outputs per element vs 1
            for other types.

    Returns:
        Tensor: Batched key tensor. For 1D: shape ``(*splits, 2)``.
        For N-D: shape ``(*splits, *tile_shape[:-1], 2)``, where each tile
        key carries one sub-key per row of the tile.

    .. note::

        For the Philox algorithm, ``unbind`` works by shifting the offset
        component of the key so that each sub-key points to the start of
        its tile within the same PRNG stream.
    """
    if len(shape) != len(splits):
        raise ValueError(
            f"shape and splits must have the same length, got {len(shape)} and {len(splits)}"
        )
    for i, (s, sp) in enumerate(zip(shape, splits)):
        if s % sp != 0:
            raise ValueError(f"splits[{i}]={sp} does not evenly divide shape[{i}]={s}")
    # Elements produced per Philox 4x32 call: 2 for float64, 4 otherwise.
    epc = 2 if dtype is not None and dtype == torch.float64 else 4
    tile_shape = tuple(s // sp for s, sp in zip(shape, splits))
    align_dim = tile_shape[-1] if len(shape) > 1 else tile_shape[0]
    if align_dim % epc != 0:
        kind = "float64" if epc == 2 else "float32/float16/bfloat16"
        raise ValueError(
            f"tile size along the innermost dimension ({align_dim}) must be a "
            f"multiple of {epc} (elements per Philox call for {kind})"
        )
    return _philox_unbind(key, shape, splits, epc)


def _philox_unbind(
    key: torch.Tensor, shape: tuple, splits: tuple, epc: int
) -> torch.Tensor:
    ndim = len(shape)
    tile_shape = tuple(s // sp for s, sp in zip(shape, splits))
    data = key.view(torch.int64)
    seed = data[..., 0]
    base_offset = data[..., 1]

    if ndim == 1:
        flat_indices = torch.arange(splits[0], dtype=torch.int64, device=key.device)
        offsets = base_offset + flat_indices * (tile_shape[0] // epc)
        seeds = seed.expand_as(offsets)
        return torch.stack([seeds, offsets], dim=-1).view(torch.uint64)

    # N-D: tiles are not contiguous in the flat stream. Each "row" (innermost
    # slice of size tile_shape[-1]) IS contiguous, so we emit one key per row
    # within each tile. Returned shape: (*splits, *tile_shape[:-1], 2).

    # Row-major strides of the full shape (in Philox outputs).
    strides = []
    s = outputs_per_elem
    for d in reversed(shape):
        strides.append(s)
        s *= d
    strides.reverse()

    # Build range tensors for tile indices and inner-tile row indices.
    ranges = []
    for j in range(ndim - 1):
        t = torch.arange(splits[j], dtype=torch.int64, device=key.device)
        i = torch.arange(tile_shape[j], dtype=torch.int64, device=key.device)
        global_j = (t * tile_shape[j]).unsqueeze(1) + i.unsqueeze(0)
        ranges.append(global_j)
    # Last dim: just tile index * tile_shape[-1]
    t_last = (
        torch.arange(splits[-1], dtype=torch.int64, device=key.device) * tile_shape[-1]
    )
    ranges.append(t_last.unsqueeze(1))

    # Broadcast all ranges to compute flat offsets.
    # Layout: (splits[0], tile_shape[0], ..., splits[n-2], tile_shape[n-2], splits[n-1], 1)
    total_dims = 2 * (ndim - 1) + 2
    offset = torch.zeros(1, dtype=torch.int64, device=key.device)
    for j in range(ndim - 1):
        view_shape = [1] * total_dims
        view_shape[2 * j] = splits[j]
        view_shape[2 * j + 1] = tile_shape[j]
        offset = offset + ranges[j].reshape(view_shape) * strides[j]
    view_shape = [1] * total_dims
    view_shape[2 * (ndim - 1)] = splits[-1]
    offset = offset + ranges[-1].reshape(view_shape)

    offset = offset + base_offset
    offset = offset.squeeze(-1)
    target_shape = []
    for j in range(ndim - 1):
        target_shape.extend([splits[j], tile_shape[j]])
    target_shape.append(splits[-1])
    offset = offset.reshape(target_shape)
    # Permute: (sp0, ts0, sp1, ts1, ..., sp_{n-1}) -> (*splits, *tile_shape[:-1])
    tile_perm = list(range(0, 2 * (ndim - 1), 2))
    tile_perm.append(2 * (ndim - 1))
    inner_perm = list(range(1, 2 * (ndim - 1), 2))
    offset = offset.permute(tile_perm + inner_perm).contiguous()

    seeds = seed.expand_as(offset)
    return torch.stack([seeds, offset], dim=-1).view(torch.uint64)


def fold_in(key: torch.Tensor, data: int) -> torch.Tensor:
    r"""Deterministically derive a new key by folding in an integer.

    Equivalent to ``split(key, data + 1)[data]``, but more efficient when
    only a single derived key is needed. Useful for associating a key with
    a loop iteration, layer index, or other integer identifier.

    Supports batched keys: if ``key`` has shape ``(*batch, K)``, each key in
    the batch is folded independently.

    Args:
        key (Tensor): A PRNG key returned by :func:`key`, :func:`split`, or
            :func:`fold_in`.
        data (int): An integer to fold into the key, interpreted as uint64.

    Returns:
        A new key tensor with the same shape as ``key``.

    Example::

        >>> key = torch.func._random.key(42, device="cuda")  # doctest: +SKIP
        >>> k0 = torch.func._random.fold_in(key, 0)  # doctest: +SKIP
        >>> k1 = torch.func._random.fold_in(key, 1)  # doctest: +SKIP
        >>> # Equivalent to split:
        >>> keys = torch.func._random.split(key, 2)  # doctest: +SKIP
        >>> assert torch.equal(k0, keys[0])  # doctest: +SKIP
        >>> assert torch.equal(k1, keys[1])  # doctest: +SKIP
    """
    return torch.ops.aten._philox_key_fold_in(key, data)


def normal_(
    key: torch.Tensor,
    result: torch.Tensor,
    *,
    mean: float = 0.0,
    std: float = 1.0,
    portable: bool = True,
) -> torch.Tensor:
    r"""Fill ``result`` in-place with normal random values from a PRNG key.

    The values are drawn from a normal distribution with the specified ``mean``
    and ``std``. The output is fully determined by the key, so calling with the
    same key always produces the same result.

    Supports batched keys: if ``key`` has shape ``(*batch, K)``, the leading
    dimensions of ``result`` must be broadcastable with ``*batch`` and each key
    independently generates its slice of the output.

    Args:
        key (Tensor): A PRNG key returned by :func:`key`, :func:`split`, or
            :func:`fold_in`.
        result (Tensor): The output tensor to fill in-place.
        mean (float): Mean of the normal distribution. Default: ``0.0``.
        std (float): Standard deviation of the normal distribution. Default: ``1.0``.
        portable (bool): If ``True`` (default), the output is identical
            across GPU types for the same key. If ``False``, device-specific
            optimizations may produce different values but may offer better
            performance.

    Returns:
        ``result``, filled with normal random values.

    Example::

        >>> key = torch.func._random.key(42, device="cuda")  # doctest: +SKIP
        >>> result = torch.empty(1000, device="cuda")  # doctest: +SKIP
        >>> torch.func._random.normal_(key, result)  # doctest: +SKIP
    """
    return torch.ops.aten._philox_normal_(result, key, mean, std, portable)


def normal(
    key: torch.Tensor,
    *shape: tuple[int, ...],
    mean: float = 0.0,
    std: float = 1.0,
    dtype: torch.dtype | None = None,
    portable: bool = True,
) -> torch.Tensor:
    r"""Generate normally distributed random values from a PRNG key.

    Produces a tensor of the given shape filled with values drawn from a normal
    distribution with the specified ``mean`` and ``std``. The output is fully
    determined by the key, so calling with the same key always returns the same
    result. The output is placed on the same device as ``key``.

    Supports batched keys: if ``key`` has shape ``(*batch, K)``, the leading
    dimensions of ``shape`` must be broadcastable with ``*batch`` and each key
    independently generates its slice of the output.

    Args:
        key (Tensor): A PRNG key returned by :func:`key`, :func:`split`, or
            :func:`fold_in`.
        *shape (int): The desired output shape.
        mean (float): Mean of the normal distribution. Default: ``0.0``.
        std (float): Standard deviation of the normal distribution. Default: ``1.0``.
        dtype (:class:`torch.dtype`, optional): The desired dtype. Default: ``torch.float32``.
        portable (bool): If ``True`` (default), the output is identical
            across GPU types for the same key. CPU and CUDA outputs are close
            but may not be bitwise identical due to different transcendental
            function implementations used in the Box-Muller transform. If
            ``False``, device-specific optimizations may produce more
            significantly different values across devices but may offer
            better performance.

    Returns:
        A tensor of the given shape filled with normal random values.

    Example::

        >>> key = torch.func._random.key(42, device="cuda")  # doctest: +SKIP
        >>> torch.func._random.normal(key, (1000,))  # doctest: +SKIP
    """
    if len(shape) == 1 and isinstance(shape[0], Sequence):
        # pyrefly: ignore [bad-argument-type]
        shape = tuple(shape[0])
    if dtype is None:
        dtype = torch.float32
    # pyrefly: ignore [no-matching-overload]
    result = torch.empty(shape, dtype=dtype, device=key.device)
    return normal_(key, result, mean=mean, std=std, portable=portable)


def uniform_(
    key: torch.Tensor,
    result: torch.Tensor,
    *,
    low: float = 0.0,
    high: float = 1.0,
    portable: bool = True,
) -> torch.Tensor:
    r"""Fill ``result`` in-place with uniform random values from a PRNG key.

    The values are drawn uniformly from the interval ``[low, high)``. The output
    is fully determined by the key, so calling with the same key always produces
    the same result.

    Supports batched keys: if ``key`` has shape ``(*batch, K)``, the leading
    dimensions of ``result`` must be broadcastable with ``*batch`` and each key
    independently generates its slice of the output.

    Args:
        key (Tensor): A PRNG key returned by :func:`key`, :func:`split`, or
            :func:`fold_in`.
        result (Tensor): The output tensor to fill in-place.
        low (float): Lower bound (inclusive) of the uniform distribution. Default: ``0.0``.
        high (float): Upper bound (exclusive) of the uniform distribution. Default: ``1.0``.
        portable (bool): If ``True`` (default), the output is identical
            across CPU, CUDA, and different GPU types for the same key. If
            ``False``, device-specific optimizations may produce different
            values across devices but may offer better performance.

    Returns:
        ``result``, filled with uniform random values.

    Example::

        >>> key = torch.func._random.key(42, device="cuda")  # doctest: +SKIP
        >>> result = torch.empty(1000, device="cuda")  # doctest: +SKIP
        >>> torch.func._random.uniform_(key, result)  # doctest: +SKIP
    """
    return torch.ops.aten._philox_uniform_(result, key, low, high, portable)


def uniform(
    key: torch.Tensor,
    *shape: tuple[int, ...],
    low: float = 0.0,
    high: float = 1.0,
    dtype: torch.dtype | None = None,
    portable: bool = True,
) -> torch.Tensor:
    r"""Generate uniformly distributed random values from a PRNG key.

    Produces a tensor of the given shape filled with values drawn uniformly
    from the interval ``[low, high)``. The output is fully determined by the
    key, so calling with the same key always returns the same result. The output
    is placed on the same device as ``key``.

    Supports batched keys: if ``key`` has shape ``(*batch, K)``, the leading
    dimensions of ``shape`` must be broadcastable with ``*batch`` and each key
    independently generates its slice of the output.

    Args:
        key (Tensor): A PRNG key returned by :func:`key`, :func:`split`, or
            :func:`fold_in`.
        *shape (int): The desired output shape.
        low (float): Lower bound (inclusive) of the uniform distribution. Default: ``0.0``.
        high (float): Upper bound (exclusive) of the uniform distribution. Default: ``1.0``.
        dtype (:class:`torch.dtype`, optional): The desired dtype. Default: ``torch.float32``.
        portable (bool): If ``True`` (default), the output is identical
            across CPU, CUDA, and different GPU types for the same key. If
            ``False``, device-specific optimizations may produce different
            values across devices but may offer better performance.

    Returns:
        A tensor of the given shape filled with uniform random values.

    Example::

        >>> key = torch.func._random.key(42, device="cuda")  # doctest: +SKIP
        >>> torch.func._random.uniform(key, (1000,))  # doctest: +SKIP
    """
    if len(shape) == 1 and isinstance(shape[0], Sequence):
        # pyrefly: ignore [bad-argument-type]
        shape = tuple(shape[0])
    if dtype is None:
        dtype = torch.float32
    # pyrefly: ignore [no-matching-overload]
    result = torch.empty(shape, dtype=dtype, device=key.device)
    return uniform_(key, result, low=low, high=high, portable=portable)
