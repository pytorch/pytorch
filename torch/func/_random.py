"""Stateless PRNG APIs.

These are experimental and subject to change without notice.
Access via ``torch.func._random``.
"""

import math
import operator
from collections.abc import Sequence

import torch


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


def fold_in(key: torch.Tensor, data: int | torch.Tensor) -> torch.Tensor:
    r"""Deterministically derive a new key by folding in an integer value.

    ``data`` may be a Python ``int`` or a ``uint64`` tensor on the same device
    as ``key``. Tensor data is broadcast against the key's batch dimensions,
    deriving one key for each broadcasted ``(key, data)`` pair. Passing
    ``data`` as a tensor also prevents it from being baked into a captured CUDA
    graph, so the graph can be replayed with different values without
    recapture.

    For scalar ``data``, this is equivalent to
    ``split(key, data + 1)[data]``, but more efficient when only a single
    derived key is needed. Useful for associating a key with a loop iteration,
    layer index, or other integer identifier.

    Supports batched keys and tensor data. If ``key`` has shape
    ``(*key_batch, K)`` and tensor ``data`` has shape ``*data_batch``, the
    result has shape ``(*broadcast_shapes(key_batch, data_batch), K)``.

    Args:
        key (Tensor): A PRNG key returned by :func:`key`, :func:`split`, or
            :func:`fold_in`.
        data (int or Tensor): The value to fold into the key, interpreted as
            uint64. An ``int`` must be within the inclusive range
            ``[-0x8000_0000_0000_0000, 0xffff_ffff_ffff_ffff]``. Negative inputs
            are remapped to positive values with the formula
            ``0x1_0000_0000_0000_0000 + data``. A tensor must have dtype
            ``uint64`` and reside on the same device as ``key``. Tensor data is
            broadcast against the key's batch dimensions.

    Returns:
        A new key tensor. Python integer data preserves the shape of ``key``;
        tensor data produces the broadcasted batch shape followed by the key
        dimension.

    Example::

        >>> key = torch.func._random.key(42, device="cuda")  # doctest: +SKIP
        >>> k0 = torch.func._random.fold_in(key, 0)  # doctest: +SKIP
        >>> k1 = torch.func._random.fold_in(key, 1)  # doctest: +SKIP
        >>> # Equivalent to split:
        >>> keys = torch.func._random.split(key, 2)  # doctest: +SKIP
        >>> assert torch.equal(k0, keys[0])  # doctest: +SKIP
        >>> assert torch.equal(k1, keys[1])  # doctest: +SKIP
    """
    if isinstance(data, torch.Tensor):
        return torch.ops.aten._philox_key_fold_in.Tensor(key, data)
    data = int(data)
    if not -(1 << 63) <= data <= (1 << 64) - 1:
        raise ValueError(
            f"fold_in: int data must be in [-2**63, 2**64 - 1], got {data}"
        )
    # Reinterpret as signed int64 due to ATen op schema; kernel will cast back
    if data >= (1 << 63):
        data -= 1 << 64
    return torch.ops.aten._philox_key_fold_in(key, data)


class StatefulPRNG:
    r"""Own a mutable root key for stateless random operations.

    ``take_key()`` splits the current state into two independent keys, returns
    the first, and stores the second as the next state. One call therefore
    represents one state transition regardless of how much work consumes the
    returned key.

    Args:
        seed (int): Seed used to create the initial key.
        device (:class:`torch.device`, optional): Device on which to store the
            key. Default: ``cpu``.

    .. note::

        Mutating state-owner methods are intended for single-threaded host
        control flow outside ``torch.compile`` and accelerator graph capture.
        The keys returned by :meth:`take_key` may be consumed by compiled or
        captured stateless operations.
    """

    def __init__(self, seed: int, *, device: torch.device | None = None) -> None:
        self._state = key(seed, device=device)

    @property
    def device(self) -> torch.device:
        """Return the device storing this PRNG's state."""
        return self._state.device

    def take_key(self) -> torch.Tensor:
        """Return one child key and advance to the other child."""
        children = split(self._state, 2)
        self._state = children[1].clone()
        return children[0]

    def get_state(self) -> torch.Tensor:
        """Return a copy of the current key state."""
        return self._state.clone()

    def set_state(self, state: torch.Tensor) -> "StatefulPRNG":
        """Restore state, copying it onto this PRNG's construction device."""
        if not isinstance(state, torch.Tensor):
            raise TypeError(
                f"StatefulPRNG state must be a torch.Tensor, got {type(state).__name__}"
            )
        if state.dtype != torch.uint64 or state.shape != (2,):
            raise ValueError("StatefulPRNG state must be a uint64 tensor of shape (2,)")
        if state.layout != torch.strided:
            raise ValueError("StatefulPRNG state must have strided layout")
        new_state = state.to(device=self.device, copy=True)
        self._state = new_state
        return self


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
    return torch.ops.aten._philox_normal_(result, key, mean, std)


def normal(
    key: torch.Tensor,
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
    return torch.ops.aten._philox_uniform_(result, key, low, high)


def uniform(
    key: torch.Tensor,
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


def _as_index_tuple(values: Sequence[int], name: str) -> tuple[int, ...]:
    try:
        return tuple(operator.index(value) for value in values)
    except TypeError as error:
        raise TypeError(f"{name} must contain integers") from error


def _rectangles_overlap(
    first_offset: tuple[int, ...],
    first_size: tuple[int, ...],
    second_offset: tuple[int, ...],
    second_size: tuple[int, ...],
) -> bool:
    return all(
        first < second + second_extent and second < first + first_extent
        for first, first_extent, second, second_extent in zip(
            first_offset,
            first_size,
            second_offset,
            second_size,
        )
    )


def _validate_shard_metadata(
    key: torch.Tensor,
    result: torch.Tensor,
    global_shape: Sequence[int],
    global_offsets: Sequence[Sequence[int]],
    local_offsets: Sequence[Sequence[int]],
    local_sizes: Sequence[Sequence[int]],
) -> tuple[
    tuple[int, ...],
    tuple[tuple[int, ...], ...],
    tuple[tuple[int, ...], ...],
    tuple[tuple[int, ...], ...],
]:
    op_name = "stateless shard distribution"
    if key.dtype != torch.uint64 or key.shape != (2,):
        raise ValueError(
            f"{op_name}: key must be an unbatched uint64 key of shape (2,)"
        )
    if key.device != result.device:
        raise ValueError(
            f"{op_name}: key and result must be on the same device, "
            f"got {key.device} and {result.device}"
        )
    if result.layout != torch.strided:
        raise ValueError(f"{op_name}: result must have strided layout")
    if not result.is_floating_point():
        raise ValueError(f"{op_name}: result must have a floating point dtype")
    if torch._debug_has_internal_overlap(result) == 1:
        raise ValueError(f"{op_name}: result must not have internal overlap")
    if torch._C._overlaps(key, result):
        raise ValueError(f"{op_name}: key and result must not overlap")

    shape = _as_index_tuple(global_shape, "global_shape")
    if len(shape) != result.ndim:
        raise ValueError(f"{op_name}: global_shape and result must have the same rank")
    if any(size < 0 for size in shape):
        raise ValueError(f"{op_name}: global_shape must be non-negative")
    if math.prod(shape) > torch.iinfo(torch.int64).max:
        raise ValueError(f"{op_name}: global tensor has more than int64 elements")

    global_rects = tuple(
        _as_index_tuple(offset, f"global_offsets[{index}]")
        for index, offset in enumerate(global_offsets)
    )
    local_rects = tuple(
        _as_index_tuple(offset, f"local_offsets[{index}]")
        for index, offset in enumerate(local_offsets)
    )
    sizes = tuple(
        _as_index_tuple(size, f"local_sizes[{index}]")
        for index, size in enumerate(local_sizes)
    )
    if len(global_rects) != len(local_rects) or len(global_rects) != len(sizes):
        raise ValueError(
            f"{op_name}: global_offsets, local_offsets, and local_sizes "
            "must have the same length"
        )

    local_shape = tuple(result.shape)
    nonempty: list[int] = []
    for index, (global_offset, local_offset, size) in enumerate(
        zip(global_rects, local_rects, sizes)
    ):
        if len(global_offset) != result.ndim:
            raise ValueError(
                f"{op_name}: global_offsets[{index}] must have {result.ndim} dimensions"
            )
        if len(local_offset) != result.ndim:
            raise ValueError(
                f"{op_name}: local_offsets[{index}] must have {result.ndim} dimensions"
            )
        if len(size) != result.ndim:
            raise ValueError(
                f"{op_name}: local_sizes[{index}] must have {result.ndim} dimensions"
            )
        for dim, (offset, extent, bound) in enumerate(zip(global_offset, size, shape)):
            if offset < 0 or extent < 0 or offset > bound - extent:
                raise ValueError(
                    f"{op_name}: global rectangle {index} dimension {dim} "
                    "is outside global_shape"
                )
        for dim, (offset, extent, bound) in enumerate(
            zip(local_offset, size, local_shape)
        ):
            if offset < 0 or offset > bound - extent:
                raise ValueError(
                    f"{op_name}: local rectangle {index} dimension {dim} "
                    "is outside result shape"
                )
        if all(extent > 0 for extent in size):
            nonempty.append(index)

    for position, first in enumerate(nonempty):
        for second in nonempty[position + 1 :]:
            if _rectangles_overlap(
                global_rects[first], sizes[first], global_rects[second], sizes[second]
            ):
                raise ValueError(
                    f"{op_name}: global rectangles {first} and {second} overlap"
                )
            if _rectangles_overlap(
                local_rects[first], sizes[first], local_rects[second], sizes[second]
            ):
                raise ValueError(
                    f"{op_name}: local rectangles {first} and {second} overlap"
                )

    return shape, global_rects, local_rects, sizes


def _materialized_shards_(
    key: torch.Tensor,
    result: torch.Tensor,
    *,
    global_shape: Sequence[int],
    global_offsets: Sequence[Sequence[int]],
    local_offsets: Sequence[Sequence[int]],
    local_sizes: Sequence[Sequence[int]],
    distribution: str,
    params: tuple[float, float],
) -> torch.Tensor:
    shape, global_rects, local_rects, sizes = _validate_shard_metadata(
        key,
        result,
        global_shape,
        global_offsets,
        local_offsets,
        local_sizes,
    )
    if distribution == "normal":
        if not params[1] >= 0:
            raise ValueError(f"normal expects std >= 0.0, but found std {params[1]}")
    elif distribution == "uniform":
        if not params[0] <= params[1]:
            raise ValueError(
                f"uniform expects low <= high, but found {params[0]} > {params[1]}"
            )
    else:
        raise ValueError(f"unsupported distribution: {distribution}")

    global_strides = [0] * len(shape)
    stride = 1
    for dim in range(len(shape) - 1, -1, -1):
        global_strides[dim] = stride
        stride *= shape[dim]

    for global_offset, local_offset, size in zip(global_rects, local_rects, sizes):
        if any(extent == 0 for extent in size):
            continue
        indices = torch.zeros(size, dtype=torch.int64, device=result.device)
        for dim, (offset, extent, global_stride) in enumerate(
            zip(global_offset, size, global_strides)
        ):
            coordinate = torch.arange(
                offset,
                offset + extent,
                dtype=torch.int64,
                device=result.device,
            )
            coordinate_shape = [1] * len(size)
            coordinate_shape[dim] = extent
            indices.add_(coordinate.reshape(coordinate_shape), alpha=global_stride)

        element_keys = fold_in(key, indices.to(torch.uint64))
        local_slices = tuple(
            slice(offset, offset + extent) for offset, extent in zip(local_offset, size)
        )
        local_result = result[local_slices]
        if distribution == "normal":
            normal_(element_keys, local_result, mean=params[0], std=params[1])
        else:
            uniform_(element_keys, local_result, low=params[0], high=params[1])
    return result


def _fused_shards_(
    key: torch.Tensor,
    result: torch.Tensor,
    *,
    global_shape: Sequence[int],
    global_offsets: Sequence[Sequence[int]],
    local_offsets: Sequence[Sequence[int]],
    local_sizes: Sequence[Sequence[int]],
    distribution: str,
    params: tuple[float, float],
) -> torch.Tensor:
    shape, global_rects, local_rects, sizes = _validate_shard_metadata(
        key,
        result,
        global_shape,
        global_offsets,
        local_offsets,
        local_sizes,
    )
    if distribution == "normal":
        if not params[1] >= 0:
            raise ValueError(f"normal expects std >= 0.0, but found std {params[1]}")
        distribution_id = 0
    elif distribution == "uniform":
        if not params[0] <= params[1]:
            raise ValueError(
                f"uniform expects low <= high, but found {params[0]} > {params[1]}"
            )
        distribution_id = 1
    else:
        raise ValueError(f"unsupported distribution: {distribution}")

    return torch.ops.aten._philox_keyed_distribution_shards_(
        result,
        key,
        shape,
        tuple(value for rectangle in global_rects for value in rectangle),
        tuple(value for rectangle in local_rects for value in rectangle),
        tuple(value for rectangle in sizes for value in rectangle),
        len(sizes),
        distribution_id,
        params,
    )


def normal_shards_(
    key: torch.Tensor,
    result: torch.Tensor,
    *,
    global_shape: Sequence[int],
    global_offsets: Sequence[Sequence[int]],
    local_offsets: Sequence[Sequence[int]],
    local_sizes: Sequence[Sequence[int]],
    mean: float = 0.0,
    std: float = 1.0,
) -> torch.Tensor:
    r"""Fill logical tensor shards with values from a stateless normal distribution.

    Each local element is keyed by its row-major index in ``global_shape``, so
    the logical values are independent of how rectangles are partitioned or
    placed in ``result``. Padding and holes outside the rectangles are not
    modified.
    """
    implementation = (
        _fused_shards_ if result.device.type == "cuda" else _materialized_shards_
    )
    return implementation(
        key,
        result,
        global_shape=global_shape,
        global_offsets=global_offsets,
        local_offsets=local_offsets,
        local_sizes=local_sizes,
        distribution="normal",
        params=(mean, std),
    )


def uniform_shards_(
    key: torch.Tensor,
    result: torch.Tensor,
    *,
    global_shape: Sequence[int],
    global_offsets: Sequence[Sequence[int]],
    local_offsets: Sequence[Sequence[int]],
    local_sizes: Sequence[Sequence[int]],
    low: float = 0.0,
    high: float = 1.0,
) -> torch.Tensor:
    r"""Fill logical tensor shards with values from a stateless uniform distribution.

    Each local element is keyed by its row-major index in ``global_shape``, so
    the logical values are independent of how rectangles are partitioned or
    placed in ``result``. Padding and holes outside the rectangles are not
    modified.
    """
    implementation = (
        _fused_shards_ if result.device.type == "cuda" else _materialized_shards_
    )
    return implementation(
        key,
        result,
        global_shape=global_shape,
        global_offsets=global_offsets,
        local_offsets=local_offsets,
        local_sizes=local_sizes,
        distribution="uniform",
        params=(low, high),
    )
