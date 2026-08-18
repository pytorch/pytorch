"""Stateless PRNG APIs.

These are experimental and subject to change without notice.
Access via ``torch.func._random``.
"""

from collections.abc import Sequence

import torch


# Inclusive lower and exclusive upper bound of each dtype randint() supports.
_RANDINT_DTYPE_RANGE = {
    torch.uint8: (0, 2**8),
    torch.uint16: (0, 2**16),
    torch.uint32: (0, 2**32),
    torch.uint64: (0, 2**64),
    torch.int8: (-(2**7), 2**7),
    torch.int16: (-(2**15), 2**15),
    torch.int32: (-(2**31), 2**31),
    torch.int64: (-(2**63), 2**63),
}


def _as_int64(value: int) -> int:
    # Bounds are passed to ATen as int64; reinterpret wider values so the
    # kernel's unsigned arithmetic recovers the intended range.
    return ((value + 2**63) % 2**64) - 2**63


class PRNGKey(torch.Tensor):
    """Base tensor subclass for typed PRNG keys.

    Uses _make_wrapper_subclass with __tensor_flatten__/__tensor_unflatten__
    so torch.compile can decompose the key into a plain tensor for tracing.
    __torch_dispatch__ unwraps the key for all ops, so the dispatcher always
    sees plain tensors.
    """

    _data: torch.Tensor

    __torch_function__ = torch._C._disabled_torch_function_impl

    @staticmethod
    def __new__(cls, data: torch.Tensor):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            data.shape,
            dtype=data.dtype,
            device=data.device,
            strides=data.stride(),
        )

    def __init__(self, data: torch.Tensor):
        self._data = data

    def __tensor_flatten__(self):
        return ["_data"], {}

    @classmethod
    def __tensor_unflatten__(cls, inner_tensors, metadata, outer_size, outer_stride):
        return cls(inner_tensors["_data"])

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        def unwrap(x):
            return x._data if isinstance(x, PRNGKey) else x

        args = torch.utils._pytree.tree_map(unwrap, args)
        kwargs = torch.utils._pytree.tree_map(unwrap, kwargs)
        return func(*args, **kwargs)

    def __repr__(self):
        return f"{type(self).__name__}({self._data})"

    def _unbind(
        self, shape: tuple, splits: tuple, outputs_per_elem: int
    ) -> "PRNGKey":
        raise NotImplementedError

    def _split(self, num: int) -> "PRNGKey":
        raise NotImplementedError

    def _fold_in(self, data: int | torch.Tensor) -> "PRNGKey":
        raise NotImplementedError

    def _uniform(
        self, out: torch.Tensor, low: float, high: float
    ) -> torch.Tensor:
        raise NotImplementedError

    def _normal(
        self, out: torch.Tensor, mean: float, std: float
    ) -> torch.Tensor:
        raise NotImplementedError

    def _randint(
        self, out: torch.Tensor, low: int | None, high: int | None
    ) -> torch.Tensor:
        raise NotImplementedError


class Philox4x32_10Key(PRNGKey):
    """Philox 4x32-10 PRNG key. Data layout: (*batch, 2) uint64 [seed, offset]."""

    @classmethod
    def __tensor_unflatten__(cls, inner_tensors, metadata, outer_size, outer_stride):
        return cls(inner_tensors["_data"])

    def _unbind(self, shape, splits, outputs_per_elem):
        return Philox4x32_10Key(
            _philox_unbind(self._data, shape, splits, outputs_per_elem)
        )

    def _split(self, num):
        return Philox4x32_10Key(torch.ops.aten._philox_key_split(self, num))

    def _fold_in(self, data):
        if isinstance(data, torch.Tensor):
            result = torch.ops.aten._philox_key_fold_in.Tensor(self, data)
        else:
            result = torch.ops.aten._philox_key_fold_in(self, data)
        return Philox4x32_10Key(result)

    def _uniform(self, out, low, high):
        return torch.ops.aten._philox_uniform_(out, self, low, high)

    def _normal(self, out, mean, std):
        return torch.ops.aten._philox_normal_(out, self, mean, std)

    def _randint(self, out, low, high):
        return torch.ops.aten._philox_randint_(out, self, low, high)


_IMPLS: dict[str, type[PRNGKey]] = {"philox4x32-10": Philox4x32_10Key}


def key(
    seed: int, *, device: torch.device | None = None, impl: str = "philox4x32-10"
) -> torch.Tensor:
    r"""Create a PRNG key from a seed.

    A key is a tensor that encodes the state needed to deterministically
    produce random values. Keys are consumed by generation functions to produce
    reproducible random tensors without any global state. The internal
    representation of the key depends on the chosen PRNG algorithm.

    Args:
        seed (int): The seed value for the PRNG.
        device (:class:`torch.device`, optional): The desired device for the
            returned key. Default: ``cpu``.
        impl (str): PRNG algorithm. Currently only ``"philox4x32-10"`` is
            supported.

    Returns:
        A tensor representing the PRNG key.

    .. note::

        For the ``"philox4x32-10"`` algorithm, the key is a uint64 tensor of
        shape ``(2,)`` encoding a ``(seed, offset)`` pair. The offset determines
        the starting position in the Philox output stream.

    Example::

        >>> key = torch.func._random.key(42, device="cuda")  # doctest: +SKIP
    """
    cls = _IMPLS.get(impl)
    if cls is None:
        raise NotImplementedError(f"key() does not support PRNG impl '{impl}'")
    data = torch.tensor([seed, 0], dtype=torch.uint64, device=device)
    return cls(data)


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
    if isinstance(key, PRNGKey):
        return key._split(num)
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
    outputs_per_elem = 2 if dtype is not None and dtype == torch.float64 else 1
    if isinstance(key, PRNGKey):
        return key._unbind(shape, splits, outputs_per_elem)
    return _philox_unbind(key, shape, splits, outputs_per_elem)


def _philox_unbind(
    key: torch.Tensor, shape: tuple, splits: tuple, outputs_per_elem: int
) -> torch.Tensor:
    ndim = len(shape)
    tile_shape = tuple(s // sp for s, sp in zip(shape, splits))
    data = key.view(torch.int64)
    seed = data[..., 0]
    base_offset = data[..., 1]

    if ndim == 1:
        flat_indices = torch.arange(splits[0], dtype=torch.int64, device=key.device)
        offsets = base_offset + flat_indices * (tile_shape[0] * outputs_per_elem)
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


def fold_in(key: torch.Tensor, data: int | torch.Tensor) -> torch.Tensor:
    r"""Deterministically derive a new key by folding in an integer value.

    ``data`` may be a Python ``int`` or a single-item ``uint64`` tensor on the
    same device as ``key``. Note that passing ``data`` as a tensor prevents it
    from being baked into a captured CUDA graph, so the graph can be replayed
    with a different value without recapture.

    Equivalent to ``split(key, data + 1)[data]``, but more efficient when
    only a single derived key is needed. Useful for associating a key with
    a loop iteration, layer index, or other integer identifier.

    Supports batched keys: if ``key`` has shape ``(*batch, K)``, ``data`` is
    folded into each key independently.

    Args:
        key (Tensor): A PRNG key returned by :func:`key`, :func:`split`, or
            :func:`fold_in`.
        data (int or Tensor): The value to fold into the key, interpreted as
            uint64. An ``int`` must be within the inclusive range
            ``[-0x8000_0000_0000_0000, 0xffff_ffff_ffff_ffff]``. Negative inputs
            are remapped to positive values with the formula
            ``0x1_0000_0000_0000_0000 + data``. A tensor must have dtype
            ``uint64``, contain a single value, and reside on the same device
            as ``key``.

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
    if not isinstance(data, torch.Tensor):
        data = int(data)
        if not -(1 << 63) <= data <= (1 << 64) - 1:
            raise ValueError(
                f"fold_in: int data must be in [-2**63, 2**64 - 1], got {data}"
            )
        # Reinterpret as signed int64 due to ATen op schema; kernel will cast back
        if data >= (1 << 63):
            data -= 1 << 64
    if isinstance(key, PRNGKey):
        return key._fold_in(data)
    if isinstance(data, torch.Tensor):
        return torch.ops.aten._philox_key_fold_in.Tensor(key, data)
    return torch.ops.aten._philox_key_fold_in(key, data)


def normal_(
    key: torch.Tensor,
    result: torch.Tensor,
    *,
    mean: float = 0.0,
    std: float = 1.0,
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

    Returns:
        ``result``, filled with normal random values.

    Example::

        >>> key = torch.func._random.key(42, device="cuda")  # doctest: +SKIP
        >>> result = torch.empty(1000, device="cuda")  # doctest: +SKIP
        >>> torch.func._random.normal_(key, result)  # doctest: +SKIP
    """
    if isinstance(key, PRNGKey):
        return key._normal(result, mean, std)
    return torch.ops.aten._philox_normal_(result, key, mean, std)


def normal(
    key,
    *shape: tuple[int, ...],
    mean: float = 0.0,
    std: float = 1.0,
    dtype: torch.dtype | None = None,
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
    return normal_(key, result, mean=mean, std=std)


def uniform_(
    key: torch.Tensor,
    result: torch.Tensor,
    *,
    low: float = 0.0,
    high: float = 1.0,
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

    Returns:
        ``result``, filled with uniform random values.

    Example::

        >>> key = torch.func._random.key(42, device="cuda")  # doctest: +SKIP
        >>> result = torch.empty(1000, device="cuda")  # doctest: +SKIP
        >>> torch.func._random.uniform_(key, result)  # doctest: +SKIP
    """
    if isinstance(key, PRNGKey):
        return key._uniform(result, low, high)
    return torch.ops.aten._philox_uniform_(result, key, low, high)


def uniform(
    key,
    *shape: tuple[int, ...],
    low: float = 0.0,
    high: float = 1.0,
    dtype: torch.dtype | None = None,
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
    return uniform_(key, result, low=low, high=high)


def randint_(
    key: torch.Tensor,
    result: torch.Tensor,
    *,
    low: int | None = 0,
    high: int | None = None,
) -> torch.Tensor:
    r"""Fill ``result`` in-place with uniform random integers in ``[low, high)``.

    ``low`` defaults to ``0`` and ``high`` to the dtype's largest value plus one.
    Passing ``None`` for either selects the corresponding limit of the dtype, so
    ``randint_(key, result, low=None, high=None)`` draws from the dtype's full
    range.

    .. warning::

        The two in-place analogues in core PyTorch both take bounds
        positionally, while these are keyword-only. Versus
        :func:`torch.randint` with ``out=``, which needs the size even though
        ``out`` already fixes it::

            torch.randint(10, (2, 3), out=result)  # -> [0, 10)
            randint_(key, result, high=10)  # size comes from result

            torch.randint(1, 5, (2, 3), out=result)  # -> [1, 5)
            randint_(key, result, low=1, high=5)

        Versus :meth:`torch.Tensor.random_`, which names the bounds ``from`` /
        ``to`` and treats a lone bound as the upper one::

            result.random_(10)  # -> [0, 10)
            randint_(key, result, high=10)

            result.random_(1, 5)  # -> [1, 5)
            randint_(key, result, low=1, high=5)

            result.random_()  # -> [0, DTYPE_MAX]
            randint_(key, result)

            result.random_(DTYPE_MIN, None)  # whole dtype range
            randint_(key, result, low=None, high=None)

        The bound defaults agree: both fill ``[0, DTYPE_MAX]`` when given no
        bounds, and both need an explicit lower bound to reach negative values.
        The spelling differs in that ``None`` here means the dtype's limit for
        either bound, so ``low=None`` replaces ``random_``'s ``DTYPE_MIN``.

    The output is fully determined by the key, so calling with the same key
    always produces the same result.

    Supports batched keys: if ``key`` has shape ``(*batch, K)``, the leading
    dimensions of ``result`` must be broadcastable with ``*batch`` and each key
    independently generates its slice of the output.

    Args:
        key (Tensor): A PRNG key returned by :func:`key`, :func:`split`, or
            :func:`fold_in`.
        result (Tensor): The output tensor to fill in-place. Must have an
            integer dtype other than ``torch.bool``.
        low (int, optional): Lower bound (inclusive) of the range. ``None`` means
            the dtype's smallest value. Default: ``0``.
        high (int, optional): Upper bound (exclusive) of the range. ``None`` means
            the dtype's largest value plus one. For 32-bit dtypes,
            ``high - low`` must be less than ``2 ** 28``; see the note below.
            Default: ``None``.

    Returns:
        ``result``, filled with random integers.

    .. note::

        Values are reduced into ``[low, high)`` with a modulo, which is exactly
        uniform only when ``high - low`` is a power of two. Ranges of ``2 ** 28``
        or more are rejected for 32-bit dtypes; use a 64-bit dtype if you need a
        range that large. See :func:`randint` for the bias in detail.

    Example::

        >>> key = torch.func._random.key(42, device="cuda")  # doctest: +SKIP
        >>> result = torch.empty(1000, dtype=torch.int64, device="cuda")  # doctest: +SKIP
        >>> torch.func._random.randint_(key, result, high=10)  # doctest: +SKIP
    """
    dtype_range = _RANDINT_DTYPE_RANGE.get(result.dtype)
    if dtype_range is None:
        # Unsupported dtype; let the kernel report it.
        if isinstance(key, PRNGKey):
            return key._randint(result, None, None)
        return torch.ops.aten._philox_randint_(result, key)
    dtype_low, dtype_high = dtype_range
    lo = dtype_low if low is None else low
    hi = dtype_high if high is None else high
    if hi <= lo:
        raise ValueError(
            f"randint: high must be greater than low, got low={lo}, high={hi}"
        )
    if lo < dtype_low or hi > dtype_high:
        raise ValueError(
            f"randint: [low, high) = [{lo}, {hi}) is out of range for dtype "
            f"{result.dtype}, which covers [{dtype_low}, {dtype_high})"
        )
    # The dtype's exclusive upper limit is not always representable as int64, so
    # the kernel takes it as None.
    high_arg = None if hi == dtype_high else _as_int64(hi)
    if isinstance(key, PRNGKey):
        return key._randint(result, _as_int64(lo), high_arg)
    return torch.ops.aten._philox_randint_(result, key, _as_int64(lo), high_arg)


def randint(
    key: torch.Tensor,
    *shape: tuple[int, ...],
    low: int | None = 0,
    high: int | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    r"""Generate uniform random integers in ``[low, high)`` from a PRNG key.

    Produces a tensor of the given shape filled with integers drawn uniformly
    from ``[low, high)``. The output is fully determined by the key, so calling
    with the same key always returns the same result. The output is placed on the
    same device as ``key``.

    ``low`` defaults to ``0`` and ``high`` to the dtype's largest value plus
    one. Passing ``None`` for either selects the corresponding limit of
    ``dtype``, so ``low=None`` with no ``high`` draws from the full range.

    .. warning::

        The signature differs from :func:`torch.randint`. Bounds are
        keyword-only and follow the shape, matching :func:`uniform` and
        :func:`normal` in this module, whereas :func:`torch.randint` takes them
        positionally before the size::

            torch.randint(10, (2, 3))  # -> [0, 10)
            randint(key, (2, 3), high=10)

            torch.randint(1, 5, (2, 3))  # -> [1, 5)
            randint(key, (2, 3), low=1, high=5)

        Note that a positional integer is a shape dimension here, not a bound,
        so ``randint(key, 10, high=5)`` produces ten values in ``[0, 5)`` rather
        than the ``[0, 10)`` that :func:`torch.randint` would give.

        This function also accepts ``None`` bounds for the dtype's limits, which
        :func:`torch.randint` does not.

    Supports batched keys: if ``key`` has shape ``(*batch, K)``, the leading
    dimensions of ``shape`` must be broadcastable with ``*batch`` and each key
    independently generates its slice of the output.

    Args:
        key (Tensor): A PRNG key returned by :func:`key`, :func:`split`, or
            :func:`fold_in`.
        *shape (int): The desired output shape.
        low (int, optional): Lower bound (inclusive) of the range. ``None`` means
            the dtype's smallest value. Default: ``0``.
        high (int, optional): Upper bound (exclusive) of the range. ``None`` means
            the dtype's largest value plus one. For 32-bit dtypes,
            ``high - low`` must be less than ``2 ** 28``; see the note below.
            Default: ``None``.
        dtype (:class:`torch.dtype`, optional): The desired dtype: any integer
            dtype other than ``torch.bool``. Default: ``torch.int32``.

    Returns:
        A tensor of the given shape filled with random integers.

    .. note::

        **Sampling width.** Each element draws a fixed ``nbits`` of randomness:
        64 for the 8-byte dtypes (``torch.int64`` / ``torch.uint64``) and 32 for
        every narrower one. Dtypes smaller than 32 bits are *not* packed several
        to a word; each element still consumes a full 32-bit sample, the range
        reduction is done at 32 bits, and only the final store narrows to the
        output dtype (keeping the low bits). Sampling ``torch.int8`` therefore
        matches sampling ``torch.int32`` and casting down with
        :meth:`~torch.Tensor.to`, and its bias is measured against ``2 ** 32``
        rather than ``2 ** 8``.

        **Sampling bias.** The ``nbits`` sample is reduced into ``[low, high)``
        with a modulo. That is exactly uniform only when ``high - low`` divides ``2 ** nbits``,
        i.e. when the range is a power of two; otherwise the lowest
        ``2 ** nbits % (high - low)`` values of the range are returned slightly
        more often than the rest.

        With ``q = 2 ** nbits // (high - low)``, the most frequent value occurs
        ``1 + 1 / q`` times as often as the least frequent, so the worst case
        grows with the range::

            q = 2**nbits // (high - low)
            bias = 0.0 if 2**nbits % (high - low) == 0 else 1.0 / q

        Away from the exact cases this is close to ``(high - low) / 2 ** nbits``.
        It is a step function of the range, changing only where ``q`` does.
        Exact powers of two are unbiased, so the figures below bound every range
        up to the listed size, for a 32-bit dtype:

        ==============  ==================
        Range up to     Max relative bias
        ==============  ==================
        ``2 ** 8``      0.000006%
        ``2 ** 16``     0.0015%
        ``2 ** 20``     0.024%
        ``2 ** 24``     0.39%
        ``2 ** 26``     1.6%
        ``2 ** 27``     3.1%
        ``2 ** 28``     6.25%
        ==============  ==================

        Ranges of ``2 ** 28`` or more are rejected for dtypes sampled at 32 bits,
        to keep the bias under roughly 6%; use a 64-bit dtype if you need a range
        that large. Ranges that divide ``2 ** 32`` evenly (powers of two) are
        exact at any size and are always allowed. Dtypes narrower than 32 bits can never reach that limit, since
        their ranges cap at ``2 ** 16``, so their bias stays below 0.0015%.
        64-bit dtypes are not restricted, since the same formula over
        ``2 ** 64`` keeps the bias near ``2e-10`` even at a range of ``2 ** 32``.

        Using the dtype's full range (omitting both bounds) is always exactly
        uniform, since that range is a power of two and no reduction is applied.

    Example::

        >>> r = torch.func._random  # doctest: +SKIP
        >>> key = r.key(42, device="cuda")  # doctest: +SKIP
        >>> r.randint(key, (1000,), high=10)  # [0, 10)  # doctest: +SKIP
        >>> r.randint(key, (1000,), low=-5, high=5)  # [-5, 5)  # doctest: +SKIP
        >>> r.randint(key, 2, 3, high=10, dtype=torch.int64)  # shape (2, 3)  # doctest: +SKIP
        >>> r.randint(key, (1000,), low=None)  # the dtype's full range  # doctest: +SKIP
    """
    if len(shape) == 1 and isinstance(shape[0], Sequence):
        # pyrefly: ignore [bad-argument-type]
        shape = tuple(shape[0])
    if dtype is None:
        dtype = torch.int32
    # pyrefly: ignore [no-matching-overload]
    result = torch.empty(shape, dtype=dtype, device=key.device)
    return randint_(key, result, low=low, high=high)


def bits_(key: torch.Tensor, result: torch.Tensor) -> torch.Tensor:
    r"""Fill ``result`` in-place with raw random bits from a PRNG key.

    Equivalent to :func:`randint_` over the full range of ``result``'s dtype.
    That range is a power of two, so unlike a bounded draw this is exactly
    uniform. Signed dtypes receive negative values as well.

    Supports batched keys: if ``key`` has shape ``(*batch, K)``, the leading
    dimensions of ``result`` must be broadcastable with ``*batch`` and each key
    independently generates its slice of the output.

    Args:
        key (Tensor): A PRNG key returned by :func:`key`, :func:`split`, or
            :func:`fold_in`.
        result (Tensor): The output tensor to fill in-place. Must have an
            integer dtype other than ``torch.bool``.

    Returns:
        ``result``, filled with raw random bits.

    Example::

        >>> key = torch.func._random.key(42, device="cuda")  # doctest: +SKIP
        >>> result = torch.empty(1000, dtype=torch.uint32, device="cuda")  # doctest: +SKIP
        >>> torch.func._random.bits_(key, result)  # doctest: +SKIP
    """
    return randint_(key, result, low=None, high=None)


def bits(
    key: torch.Tensor,
    *shape: tuple[int, ...],
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    r"""Generate raw random bits from a PRNG key.

    Equivalent to :func:`randint` over the full range of ``dtype``. That range
    is a power of two, so unlike a bounded draw this is exactly uniform. Signed
    dtypes receive negative values as well.

    Dtypes narrower than 32 bits take the low bits of a full 32-bit sample, so
    ``bits(key, shape, dtype=torch.int8)`` matches ``bits(key, shape,
    dtype=torch.int32).to(torch.int8)``.

    Supports batched keys: if ``key`` has shape ``(*batch, K)``, the leading
    dimensions of ``shape`` must be broadcastable with ``*batch`` and each key
    independently generates its slice of the output.

    Args:
        key (Tensor): A PRNG key returned by :func:`key`, :func:`split`, or
            :func:`fold_in`.
        *shape (int): The desired output shape.
        dtype (:class:`torch.dtype`, optional): The desired dtype: any integer
            dtype other than ``torch.bool``. Default: ``torch.int32``.

    Returns:
        A tensor of the given shape filled with raw random bits.

    Example::

        >>> key = torch.func._random.key(42, device="cuda")  # doctest: +SKIP
        >>> torch.func._random.bits(key, (1000,), dtype=torch.uint64)  # doctest: +SKIP
    """
    if dtype is None:
        dtype = torch.int32
    return randint(key, *shape, low=None, high=None, dtype=dtype)
