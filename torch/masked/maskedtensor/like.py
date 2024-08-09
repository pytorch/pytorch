# mypy: allow-untyped-defs

import torch
from torch.masked.maskedtensor.core import _get_data, MaskedTensor, _maybe_get_mask

LIKE_NAMES = [
    'zeros_like',
    'ones_like',
    'empty_like',
    'full_like',
    'rand_like',
    'randn_like',
    'randint_like'
]

LIKE_FNS = [getattr(torch.ops.aten, name) for name in LIKE_NAMES]


def _is_like_fn(fn):
    return fn in LIKE_FNS


def _apply_like_fn(func, *args, **kwargs):
    result_data = func(_get_data(args[0]), *args[1:], **kwargs)
    return MaskedTensor(result_data, _maybe_get_mask(args[0]))
