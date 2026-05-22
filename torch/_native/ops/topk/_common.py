"""Shared eligibility checks and reshape helpers for topk overrides.

Both the CuTeDSL and Triton paths operate on a 2D ``(M, N)`` fp32 row-major
view of ``self`` where ``dim`` points at the last axis. This module hosts the
checks the conds share (contiguity, dim normalisation, COW) and the reshape
helpers the impls share.
"""

import math

import torch


def any_cow(*tensors: torch.Tensor) -> bool:
    return any(
        torch._C._is_cow_tensor(t)  # pyrefly: ignore[missing-attribute]
        for t in tensors
    )


def normalise_dim(self: torch.Tensor, dim: int) -> int:
    """Translate a possibly-negative dim into its non-negative form.

    Returns ``-1`` if the dim is out of range (cond callers should reject
    in that case).
    """
    if self.ndim == 0:
        return -1
    if dim < -self.ndim or dim >= self.ndim:
        return -1
    return dim % self.ndim


def last_dim_row_major_ok(self: torch.Tensor, dim: int) -> bool:
    """True iff reducing over ``dim`` is equivalent to reducing over the last
    axis of a 2D ``(prod(leading dims), N)`` row-major view.

    Requires ``dim`` to be the last axis and ``self`` to be contiguous in the
    standard row-major sense; together they ensure the flatten-to-2D view we
    pass to the kernel is a view (not a copy) and has stride-1 on N.
    """
    norm = normalise_dim(self, dim)
    if norm != self.ndim - 1:
        return False
    return self.is_contiguous()


def flatten_last_dim(self: torch.Tensor) -> torch.Tensor:
    """View nD self as 2D ``(M, N)`` with ``N = self.shape[-1]``.

    Caller must have verified ``last_dim_row_major_ok`` so this is a view.
    """
    if self.ndim == 2:
        return self
    if self.ndim == 1:
        return self.view(1, self.shape[0])
    N = self.shape[-1]
    M = math.prod(self.shape[:-1])
    return self.view(M, N)


def unflatten_last_dim(
    flat_values: torch.Tensor, flat_indices: torch.Tensor, self: torch.Tensor, k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reshape the 2D ``(M, K)`` kernel outputs back into ``self``'s shape."""
    if self.ndim == 2:
        return flat_values, flat_indices
    if self.ndim == 1:
        return flat_values.view(k), flat_indices.view(k)
    out_shape = self.shape[:-1] + (k,)
    return flat_values.view(out_shape), flat_indices.view(out_shape)
