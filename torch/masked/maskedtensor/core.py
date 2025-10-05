# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates

import warnings
from typing import Any
from typing_extensions import TypeIs

import torch
from torch.overrides import get_default_nowrap_functions


__all__ = [
    "MaskedTensor",
    "is_masked_tensor",
]


def is_masked_tensor(obj: Any, /) -> TypeIs["MaskedTensor"]:
    r"""Returns True if the input is a MaskedTensor, else False

    Args:
        a: any input

    Examples:

        >>> # xdoctest: +SKIP
        >>> from torch.masked import MaskedTensor
        >>> data = torch.arange(6).reshape(2, 3)
        >>> mask = torch.tensor([[True, False, False], [True, True, False]])
        >>> mt = MaskedTensor(data, mask)
        >>> is_masked_tensor(mt)
        True
    """
    return isinstance(obj, MaskedTensor)


def _tensors_match(a, b, exact=True, rtol=1e-05, atol=1e-08):
    if is_masked_tensor(a) or is_masked_tensor(b):
        raise ValueError("Neither `a` nor `b` can be a MaskedTensor.")
    if a.layout != b.layout:
        raise ValueError(
            f"`a` and `b` must have the same layout. Got {a.layout} and {b.layout}"
        )

    if a.dtype != b.dtype:
        b = b.type(a.dtype)
    if a.layout == b.layout == torch.sparse_coo:
        return _tensors_match(a.values(), b.values(), exact) and _tensors_match(
            a.indices(), b.indices(), exact
        )
    elif a.layout == b.layout == torch.sparse_csr:
        return (
            _tensors_match(a.crow_indices(), b.crow_indices(), exact)
            and _tensors_match(a.col_indices(), b.col_indices(), exact)
            and _tensors_match(a.values(), b.values(), exact)
        )
    if exact:
        return (a.dim() == b.dim()) and torch.eq(a, b).all().item()
    return (a.dim() == b.dim()) and torch.allclose(a, b, rtol=rtol, atol=atol)


def _masks_match(a, b):
    if is_masked_tensor(a) and is_masked_tensor(b):
        mask_a = a.get_mask()
        mask_b = b.get_mask()
        return _tensors_match(mask_a, mask_b, exact=True)
    return True


def _map_mt_args_kwargs(args, kwargs, map_fn):
    def _helper(a, map_fn):
        if is_masked_tensor(a):
            return map_fn(a)
        elif torch.is_tensor(a):
            return a
        elif isinstance(a, list):
            a_impl, _ = _map_mt_args_kwargs(a, {}, map_fn)
            return a_impl
        elif isinstance(a, tuple):
            a_impl, _ = _map_mt_args_kwargs(a, {}, map_fn)
            return tuple(a_impl)
        else:
            return a

    if kwargs is None:
        kwargs = {}
    impl_args = []
    for a in args:
        impl_args.append(_helper(a, map_fn))
    impl_kwargs = {}
    for k in kwargs.keys():
        impl_kwargs[k] = _helper(a, map_fn)
    return impl_args, impl_kwargs


def _wrap_result(result_data, result_mask):
    if isinstance(result_data, list):
        return [_wrap_result(r, m) for (r, m) in zip(result_data, result_mask)]
    if isinstance(result_data, tuple):
        return tuple(_wrap_result(r, m) for (r, m) in zip(result_data, result_mask))
    if torch.is_tensor(result_data):
        return MaskedTensor(result_data, result_mask)
    # Expect result_data and result_mask to be Tensors only
    return NotImplemented


def _masked_tensor_str(data, mask, formatter):
    if data.layout in {torch.sparse_coo, torch.sparse_csr}:
        data = data.to_dense()
        mask = mask.to_dense()
    if data.dim() == 1:
        formatted_elements = [
            formatter.format(d.item()) if isinstance(d.item(), float) else str(d.item())
            for d in data
        ]
        max_len = max(8 if x[1] else len(x[0]) for x in zip(formatted_elements, ~mask))
        return (
            "["
            + ", ".join(
                [
                    "--".rjust(max_len) if m else e
                    for (e, m) in zip(formatted_elements, ~mask)
                ]
            )
            + "]"
        )
    sub_strings = [_masked_tensor_str(d, m, formatter) for (d, m) in zip(data, mask)]
    sub_strings = ["\n".join(["  " + si for si in s.split("\n")]) for s in sub_strings]
    return "[\n" + ",\n".join(sub_strings) + "\n]"


def _get_data(a):
    if is_masked_tensor(a):
        return a._masked_data
    return a


def _maybe_get_mask(a):
    if is_masked_tensor(a):
        return a.get_mask()
    return None


class MaskedTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, data, mask, requires_grad=False):
        if is_masked_tensor(data) or not torch.is_tensor(data):
            raise TypeError("data must be a Tensor")
        if is_masked_tensor(mask) or not torch.is_tensor(mask):
            raise TypeError("mask must be a Tensor")
        # Use a Tensor that of the give size for the wrapper.
        kwargs = {
            "device": data.device,
            "dtype": data.dtype,
            "layout": data.layout,
            "requires_grad": requires_grad,
            "dispatch_sizes_strides_policy": "strides",
            "dispatch_layout": True,
        }
        warnings.warn(
            (
                "The PyTorch API of MaskedTensors is in prototype stage "
                "and will change in the near future. Please open a Github issue "
                "for features requests and see our documentation on the torch.masked "
                "module for further information about the project."
            ),
            UserWarning,
            stacklevel=2,
        )
        if data.requires_grad:
            warnings.warn(
                "It is not recommended to create a MaskedTensor with a tensor that requires_grad. "
                "To avoid this, you can use data.detach().clone()",
                UserWarning,
                stacklevel=2,
            )
        return torch.Tensor._make_wrapper_subclass(cls, data.size(), **kwargs)

    def _preprocess_data(self, data, mask):
        from .._ops import _sparse_coo_where, _sparse_csr_where

        if data.layout != mask.layout:
            raise TypeError("data and mask must have the same layout.")
        if data.layout == torch.sparse_coo:
            data = data.coalesce()
            mask = mask.coalesce()
            if data._nnz() != mask._nnz():
                data = _sparse_coo_where(mask, data, torch.tensor(0))
        elif data.layout == torch.sparse_csr:
            if data._nnz() != mask._nnz():
                data = _sparse_csr_where(mask, data, torch.tensor(0))

        # Have to pick awkward names to not conflict with existing fields such as data
        self._masked_data = data.clone()
        self._masked_mask = mask.clone()

    def _validate_members(self):
        data = self._masked_data
        mask = self.get_mask()
        if type(data) != type(mask):
            raise TypeError(
                f"data and mask must have the same type. Got {type(data)} and {type(mask)}"
            )
        if data.layout not in {torch.strided, torch.sparse_coo, torch.sparse_csr}:
            raise TypeError(f"data layout of {data.layout} is not supported.")
        if data.layout == torch.sparse_coo:
            if not _tensors_match(data.indices(), mask.indices(), exact=True):
                raise ValueError(
                    "data and mask are both sparse COO tensors but do not have the same indices."
                )
        elif data.layout == torch.sparse_csr:
            if not _tensors_match(
                data.crow_indices(), mask.crow_indices(), exact=True
            ) or not _tensors_match(data.col_indices(), mask.col_indices(), exact=True):
                raise ValueError(
                    "data and mask are both sparse CSR tensors but do not share either crow or col indices."
                )
        if mask.dtype != torch.bool:
            raise TypeError("mask must have dtype bool.")
        if not (
            data.dtype == torch.float16
            or data.dtype == torch.float32
            or data.dtype == torch.float64
            or data.dtype == torch.bool
            or data.dtype == torch.int8
            or data.dtype == torch.int16
            or data.dtype == torch.int32
            or data.dtype == torch.int64
        ):
            raise TypeError(f"{data.dtype} is not supported in MaskedTensor.")
        if data.dim() != mask.dim():
            raise ValueError("data.dim() must equal mask.dim()")
        if data.size() != mask.size():
            raise ValueError("data.size() must equal mask.size()")

    def __init__(self, data, mask, requires_grad=False):
        self._preprocess_data(data, mask)
        self._validate_members()

    @staticmethod
    def _from_values(data, mask):
        """Differentiable constructor for MaskedTensor"""

        class Constructor(torch.autograd.Function):
            @staticmethod
            def forward(ctx, data, mask):
                return MaskedTensor(data, mask)

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output, None

        result = Constructor.apply(data, mask)
        return result

    def _set_data_mask(self, data, mask):
        self._masked_data = data
        self._masked_mask = mask
        self._validate_members()

    def __repr__(self):  # type: ignore[override]
        formatter = "{0:8.4f}"
        if self.dim() == 0:
            scalar_data = self.get_data().item()
            data_formatted = (
                formatter.format(scalar_data)
                if isinstance(scalar_data, float)
                else str(scalar_data)
            )
            if not self.get_mask().item():
                data_formatted = "--"
            return (
                "MaskedTensor("
                + data_formatted
                + ", "
                + str(self.get_mask().item())
                + ")"
            )
        s = _masked_tensor_str(self.get_data(), self.get_mask(), formatter)
        s = "\n".join("  " + si for si in s.split("\n"))
        return "MaskedTensor(\n" + s + "\n)"

    # Seems like this needs to be defined before torch_dispatch to work
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}

        from ._ops_refs import _MASKEDTENSOR_FUNCTION_TABLE

        if func in _MASKEDTENSOR_FUNCTION_TABLE:
            return _MASKEDTENSOR_FUNCTION_TABLE[func](*args, **kwargs)

        if not all(issubclass(cls, t) for t in types):
            return NotImplemented
        with torch._C.DisableTorchFunctionSubclass():
            ret = func(*args, **kwargs)
            if func in get_default_nowrap_functions():
                return ret
            else:
                return torch._tensor._convert(ret, cls)

    @classmethod
    def unary(cls, fn, data, mask):
        return MaskedTensor(fn(data), mask)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):  # type: ignore[override]
        func = func.overloadpacket

        from ._ops_refs import _MASKEDTENSOR_DISPATCH_TABLE

        if func in _MASKEDTENSOR_DISPATCH_TABLE:
            return _MASKEDTENSOR_DISPATCH_TABLE[func](*args, **kwargs)

        msg = (
            f"{func.__name__} is not implemented in __torch_dispatch__ for MaskedTensor.\n"
            "If you would like this operator to be supported, please file an issue for a feature request at "
            "https://github.com/pytorch/maskedtensor/issues with a minimal reproducible code snippet.\n"
            "In the case that the semantics for the operator are not trivial, it would be appreciated "
            "to also include a proposal for the semantics."
        )
        warnings.warn(msg)
        return NotImplemented

    def __lt__(self, other):
        if is_masked_tensor(other):
            return MaskedTensor(self.get_data() < _get_data(other), self.get_mask())
        return MaskedTensor(self.get_data() < other, self.get_mask())

    def to_tensor(self, value):
        return self.get_data().masked_fill(~self.get_mask(), value)

    def get_data(self):
        class GetData(torch.autograd.Function):
            @staticmethod
            def forward(ctx, self):
                return self._masked_data.detach()

            @staticmethod
            def backward(ctx, grad_output):
                if is_masked_tensor(grad_output):
                    return grad_output
                return MaskedTensor(grad_output, self.get_mask())

        return GetData.apply(self)

    def get_mask(self):
        return self._masked_mask

    def is_sparse_coo(self):
        return self.layout == torch.sparse_coo

    def is_sparse_csr(self):  # type: ignore[override]
        return self.layout == torch.sparse_csr

    # Update later to support more sparse layouts
    @property
    def is_sparse(self):  # type: ignore[override]
        return self.is_sparse_coo() or self.is_sparse_csr()
