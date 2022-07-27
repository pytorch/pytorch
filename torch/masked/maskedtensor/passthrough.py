# Copyright (c) Meta Platforms, Inc. and affiliates
"""
These are functions that should simply be applied to both mask and data.

Take select or stack as an example. This operation can be applied to
both the mask and data of a MaskedTensor and the result wrapped into
a new MaskedTensor as a result.
"""

import torch

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
    torch.ops.aten.cat,
]


# TODO: tree_map doesn't cut it due to kwargs support for torch_function
def _map_mt_args_kwargs(args, kwargs, map_fn):
    if kwargs is None:
        kwargs = {}
    impl_args = []
    for a in args:
        from maskedtensor import is_masked_tensor

        if is_masked_tensor(a):
            impl_args.append(map_fn(a))
        elif torch.is_tensor(a):
            impl_args.append(a)
        elif isinstance(a, list):
            a_impl, _ = _map_mt_args_kwargs(a, {}, map_fn)
            impl_args.append(a_impl)
        elif isinstance(a, tuple):
            a_impl, _ = _map_mt_args_kwargs(a, {}, map_fn)
            impl_args.append(tuple(a_impl))
        else:
            impl_args.append(a)
    impl_kwargs = {
        k: map_fn(v) if is_masked_tensor(v) else v for (k, v) in kwargs.items()
    }
    return impl_args, impl_kwargs


def _wrap_result(result_data, result_mask):
    if isinstance(result_data, list):
        return list(_wrap_result(r, m) for (r, m) in zip(result_data, result_mask))
    if isinstance(result_data, tuple):
        return tuple(_wrap_result(r, m) for (r, m) in zip(result_data, result_mask))
    if torch.is_tensor(result_data):
        from maskedtensor import MaskedTensor

        return MaskedTensor(result_data, result_mask)
    # Expect result_data and result_mask to be Tensors only
    return NotImplemented


def is_pass_through_fn(fn):
    return fn in PASSTHROUGH_FNS


def apply_pass_through_fn(fn, *args, **kwargs):
    mask_args, mask_kwargs = _map_mt_args_kwargs(args, kwargs, lambda x: x.masked_mask)
    result_mask = fn(*mask_args, **mask_kwargs)
    data_args, data_kwargs = _map_mt_args_kwargs(args, kwargs, lambda x: x.masked_data)
    result_data = fn(*data_args, **data_kwargs)
    return _wrap_result(result_data, result_mask)
