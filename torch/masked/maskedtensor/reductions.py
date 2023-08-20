# Copyright (c) Meta Platforms, Inc. and affiliates

import warnings

import torch

from .core import is_masked_tensor
from .creation import as_masked_tensor, masked_tensor

__all__ = []  # type: ignore[var-annotated]


def _masked_all_all(data, mask=None):
    if mask is None:
        return data.all()
    return data.masked_fill(~mask, True).all()


def _masked_all_dim(data, dim, keepdim=False, mask=None):
    if mask is None:
        return torch.all(data, dim=dim, keepdim=keepdim)
    return torch.all(data.masked_fill(~mask, True), dim=dim, keepdim=keepdim)


def _masked_all(*args, **kwargs):
    if len(args) == 1 and len(kwargs) == 1:
        return _masked_all_all(args[0], mask=kwargs["mask"])
    return _masked_all_dim(*args, **kwargs)


def _multidim_any(mask, dim, keepdim):
    if isinstance(dim, int):
        return _multidim_any(mask, [dim], keepdim)
    for d in sorted(dim, reverse=True):
        mask = torch.any(mask, dim=d, keepdim=keepdim)
    return mask


def _get_masked_fn(fn):
    if fn == "all":
        return _masked_all
    return getattr(torch.masked, fn)


def _torch_reduce_all(fn):
    def reduce_all(self):
        masked_fn = _get_masked_fn(fn)
        data = self.get_data()
        mask = self.get_mask().values() if self.is_sparse() else self.get_mask()
        # When reduction is "all", then torch.argmin/torch.argmax needs to return the index of the
        # element corresponding to the min/max, but this operation isn't supported correctly for sparse layouts.
        # Therefore, this implementation calculates it using the strides.
        if fn == "all":
            result_data = masked_fn(data, mask=mask)

        elif fn in {"argmin", "argmax"} and self.is_sparse_coo():
            sparse_idx = masked_fn(data.values(), mask=mask).to(dtype=torch.int)
            indices = (
                data.to_sparse_coo().indices()
                if not self.is_sparse_coo()
                else data.indices()
            )
            idx = indices.unbind(1)[sparse_idx]
            stride = data.size().numel() / torch.tensor(
                data.size(), device=data.device
            ).cumprod(0)
            result_data = torch.sum(idx * stride)

        # we simply pass in the values for sparse COO/CSR tensors
        elif self.is_sparse():
            result_data = masked_fn(masked_tensor(data.values(), mask))

        else:
            result_data = masked_fn(self, mask=mask)

        return as_masked_tensor(result_data, torch.any(mask))

    return reduce_all


def _torch_reduce_dim(fn):
    def reduce_dim(self, dim, keepdim=False, dtype=None):
        if self.is_sparse():
            msg = (
                f"The sparse version of {fn} is not implemented in reductions.\n"
                "If you would like this operator to be supported, please file an issue for a feature request at "
                "https://github.com/pytorch/maskedtensor/issues with a minimal reproducible code snippet.\n"
                "In the case that the semantics for the operator are not trivial, it would be appreciated "
                "to also include a proposal for the semantics."
            )
            warnings.warn(msg, stacklevel=TO_BE_DETERMINED)
            return NotImplemented
        if not is_masked_tensor(self):
            raise TypeError("Input to reduce_dim must be a MaskedTensor")

        masked_fn = _get_masked_fn(fn)
        data = self.get_data()
        mask = self.get_mask()
        if fn == "all":
            result_data = masked_fn(data, dim=dim, keepdim=keepdim, mask=mask)
        else:
            result_data = masked_fn(
                self, dim=dim, keepdim=keepdim, dtype=dtype, mask=self.get_mask()
            )
        return as_masked_tensor(result_data, _multidim_any(mask, dim, keepdim))

    return reduce_dim


def _torch_reduce(fn):
    def reduce_fn(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0:
            return _torch_reduce_all(fn)(args[0])
        return _torch_reduce_dim(fn)(*args, **kwargs)

    return reduce_fn


def _reduce_dim_args(input, dim, keepdim=False, dtype=None):
    return input, dim, keepdim, dtype


def _torch_grad_reduce(fn):
    def grad_reduce(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0:
            return _torch_reduce_all(fn)(args[0])
        # TODO: autograd.Function doesn't support kwarg
        input, dim, keepdim, dtype = _reduce_dim_args(*args, **kwargs)
        return _torch_reduce_dim(fn)(input, dim, keepdim, dtype)

    return grad_reduce


REDUCE_NAMES = [
    "sum",
    "mean",
    "amin",
    "amax",
    "argmin",
    "argmax",
    "prod",
    "all",
    "norm",
    "var",
    "std",
]

NATIVE_REDUCE_MAP = {
    getattr(torch.ops.aten, name): _torch_reduce(name) for name in REDUCE_NAMES
}
TORCH_REDUCE_MAP = {
    getattr(torch, name): _torch_grad_reduce(name) for name in REDUCE_NAMES
}
TENSOR_REDUCE_MAP = {
    getattr(torch.Tensor, name): _torch_grad_reduce(name) for name in REDUCE_NAMES
}

NATIVE_REDUCE_FNS = list(NATIVE_REDUCE_MAP.keys())
TORCH_REDUCE_FNS = list(TORCH_REDUCE_MAP.keys())
TENSOR_REDUCE_FNS = list(TENSOR_REDUCE_MAP.keys())

def _is_reduction(fn):
    return fn in NATIVE_REDUCE_MAP or fn in TORCH_REDUCE_MAP or fn in TENSOR_REDUCE_MAP


def _apply_reduction(fn, *args, **kwargs):
    if fn in NATIVE_REDUCE_MAP:
        return NATIVE_REDUCE_MAP[fn](*args, **kwargs)
    if fn in TORCH_REDUCE_MAP:
        return TORCH_REDUCE_MAP[fn](*args, **kwargs)
    if fn in TENSOR_REDUCE_MAP:
        return TENSOR_REDUCE_MAP[fn](*args, **kwargs)
    return NotImplemented
