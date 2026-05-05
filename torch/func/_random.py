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
