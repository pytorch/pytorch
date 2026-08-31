"""Helpers shared by the FlyDSL RMSNorm dispatcher and kernel wrapper.

Deliberately free of flydsl imports: the dispatcher predicate runs on every
_fused_rms_norm call, including ones that fall back to ATen, so it must not
pull in the kernel module.
"""

# mypy: allow-untyped-defs

from __future__ import annotations

import torch


# Single source of truth for which dtypes the kernel handles: the dispatcher
# predicate tests membership, the wrapper reads the FlyDSL type name. Splitting
# the two would let a dtype pass the predicate and then fail the lookup.
SUPPORTED_DTYPES: dict[torch.dtype, str] = {
    torch.float32: "f32",
    torch.float16: "f16",
    torch.bfloat16: "bf16",
}


def normalized_shape_1d(normalized_shape) -> int | None:
    """Return N for one-dimensional normalization, or None if unsupported.

    The operator schema declares normalized_shape as SymInt[], but this also
    accepts a bare int and other sequences for callers that reach the kernel
    wrapper directly.
    """

    if isinstance(normalized_shape, int):
        return normalized_shape
    try:
        if len(normalized_shape) != 1:
            return None
        return int(normalized_shape[0])
    except (TypeError, ValueError):
        return None
