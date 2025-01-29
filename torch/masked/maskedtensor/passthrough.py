# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates
"""
These are functions that should simply be applied to both mask and data.
Take select or stack as an example. This operation can be applied to
both the mask and data of a MaskedTensor and the result wrapped into
a new MaskedTensor as a result.
"""

import torch

from .core import _map_mt_args_kwargs, _wrap_result


__all__ = []  # type: ignore[var-annotated]


PASSTHROUGH_FNS = [
    torch.ops.aten.select,
    torch.ops.aten.transpose,
    torch.ops.aten.split,
    torch.ops.aten.t,
    torch.ops.aten.slice,
    torch.ops.aten.slice_backward,
    torch.ops.aten.select_backward,
    torch.ops.aten.index,
    torch.ops.aten.expand,
    torch.ops.aten.view,
    torch.ops.aten._unsafe_view,
    torch.ops.aten._reshape_alias,
    torch.ops.aten._lazy_clone_alias,
    torch.ops.aten.cat,
    torch.ops.aten.unsqueeze,
    torch.ops.aten.unfold,
    torch.ops.aten.unfold_backward,
    torch.ops.aten.im2col,
    torch.ops.aten.col2im,
    torch.ops.aten.stack,
]


def _is_pass_through_fn(fn):
    return fn in PASSTHROUGH_FNS


def _apply_pass_through_fn(fn, *args, **kwargs):
    data_args, data_kwargs = _map_mt_args_kwargs(args, kwargs, lambda x: x.get_data())
    result_data = fn(*data_args, **data_kwargs)
    mask_args, mask_kwargs = _map_mt_args_kwargs(args, kwargs, lambda x: x.get_mask())
    result_mask = fn(*mask_args, **mask_kwargs)
    return _wrap_result(result_data, result_mask)
