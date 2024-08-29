# mypy: allow-untyped-defs
"""
These are the creation ops using the ``*_like`` API that need to support the
`MaskedTensor`. The wrapper just applies the function to the masked data then convert
it to a masked tensor using the mask from the given tensor.
"""

import torch

from .core import _get_data, _maybe_get_mask, MaskedTensor


__all__ = [
    "LIKE_NAMES",
    "LIKE_FNS",
]


LIKE_NAMES = [
    "empty_like",
    "full_like",
    "ones_like",
    "rand_like",
    "randint_like",
    "randn_like",
    "zeros_like",
]

LIKE_FNS = [getattr(torch.ops.aten, name) for name in LIKE_NAMES]


def _is_like_fn(fn):
    return fn in LIKE_FNS


def _apply_like_fn(func, *args, **kwargs):
    result_data = func(_get_data(args[0]), *args[1:], **kwargs)
    return MaskedTensor(result_data, _maybe_get_mask(args[0]))
