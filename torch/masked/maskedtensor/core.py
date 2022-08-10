# Copyright (c) Meta Platforms, Inc. and affiliates

import warnings

import torch
from torch._masked import _sparse_coo_where, _sparse_csr_where
from torch.overrides import get_default_nowrap_functions


__all__ = [
    "MaskedTensor",
    "is_masked_tensor",
]


def is_masked_tensor(a):
    r""" Returns True if the input is a MaskedTensor, else False

    Args:
        a: input MaskedTensor

    Shape:
        a: :math:`(*)`, where :math:`*` means any number of dimensions.

    Examples:

        >>> data = torch.arange(6).reshape(2,3)
        >>> mask = torch.tensor([[True, False, False], [True, True, False]])
        >>> mt = masked_tensor(data, mask)
        >>> is_masked_tensor(mt)
        True
    """
    return isinstance(a, MaskedTensor)


def _tensors_match(a, b, exact=True):
    if is_masked_tensor(a) or is_masked_tensor(b):
        raise ValueError("Neither `a` nor `b` can be a MaskedTensor.")
    if a.layout != b.layout:
        raise ValueError(f"`a` and `b` must have the same layout. Got {a.layout} and {b.layout}")

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
    return (a.dim() == b.dim()) and torch.allclose(a, b)


def _masks_match(a, b):
    if is_masked_tensor(a) and is_masked_tensor(b):
        mask_a = a._masked_mask
        mask_b = b._masked_mask
        return _tensors_match(mask_a, mask_b, exact=True)
    return True


def _check_args_kwargs_length(args, kwargs, error_prefix, len_args=None, len_kwargs=None):
    if len_args is not None and len_args != len(args):
        raise ValueError(f"{error_prefix}: len(args) must be {len_args} but got {len(args)}")
    if len_kwargs is not None and len_kwargs != len(kwargs):
        raise ValueError(f"{error_prefix}: len(kwargs) must be {len_kwargs} but got {len(kwargs)}")


def _masked_tensor_str(data, mask, formatter):
    if data.layout in {torch.sparse_coo, torch.sparse_csr}:
        data = data.to_dense()
        mask = mask.to_dense()
    if data.dim() == 1:
        formatted_elements = [
            formatter.format(d.item()) if isinstance(d.item(), float) else str(d.item())
            for d in data
        ]
        max_len = max(
            map(lambda x: 8 if x[1] else len(x[0]), zip(formatted_elements, ~mask))
        )
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
        return a._masked_mask
    return None


class _MaskedContiguous(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        if not is_masked_tensor(input):
            raise ValueError("MaskedContiguous forward: input must be a MaskedTensor.")

        if input.is_contiguous():
            return input

        data = input.get_data()
        mask = input.get_mask()

        return MaskedTensor(data.contiguous(), mask.contiguous())

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class _MaskedToDense(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        if not is_masked_tensor(input):
            raise ValueError("MaskedToDense forward: input must be a MaskedTensor.")

        if input.layout() == torch.strided:
            return input

        ctx.layout = input.layout()
        data = input.get_data()
        mask = input.get_mask()

        return MaskedTensor(data.to_dense(), mask.to_dense())

    @staticmethod
    def backward(ctx, grad_output):
        layout = ctx.layout

        if layout == torch.sparse_coo:
            return grad_output.to_sparse_coo()
        elif layout == torch.sparse_csr:
            return grad_output.to_sparse_csr()
        elif layout == torch.strided:
            return grad_output.to_dense()
        raise ValueError("to_dense: Unsupported input layout: ", layout)


class _MaskedToSparse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        if not is_masked_tensor(input):
            raise ValueError("MaskedToSparse forward: input must be a MaskedTensor.")

        # Following the convention from sparse tensors that to_sparse always means that we convert to sparse_coo
        if input.layout() == torch.sparse_coo:
            return input

        data = input.get_data()
        mask = input.get_mask()
        sparse_mask = mask.to_sparse_coo().coalesce()
        sparse_data = data.sparse_mask(sparse_mask)

        return MaskedTensor(sparse_data, sparse_mask)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.to_dense()


class _MaskedToSparseCsr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        if not is_masked_tensor(input):
            raise ValueError("MaskedToSparseCsr forward: input must be a MaskedTensor.")

        if input._masked_data.ndim != 2:
            raise ValueError(f"Only 2D tensors can be converted to the SparseCsr layout but got shape: {input._masked_data.size()}")

        if input.layout() == torch.sparse_csr:
            return input

        data = input.get_data()
        mask = input.get_mask()
        sparse_mask = mask.to_sparse_csr()
        sparse_data = data.sparse_mask(sparse_mask)

        return MaskedTensor(sparse_data, sparse_mask)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.to_dense()


class _MaskedWhere(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cond, self, other):
        ctx.mark_non_differentiable(cond)
        ctx.save_for_backward(cond)
        return torch.ops.aten.where(cond, self, other)

    @staticmethod
    def backward(ctx, grad_output):
        (cond,) = ctx.saved_tensors

        def masked_out_like(mt):
            return MaskedTensor(mt.get_data(), torch.zeros_like(mt.get_mask()).bool())

        return (
            None,
            torch.ops.aten.where(cond, grad_output, masked_out_like(grad_output)),
            torch.ops.aten.where(cond, masked_out_like(grad_output), grad_output),
        )


class MaskedTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, data, mask, requires_grad=False):
        if not torch.is_tensor(data):
            raise TypeError("data must be a Tensor")
        if not torch.is_tensor(mask):
            raise TypeError("mask must be a Tensor")
        # Use a Tensor that of the give size for the wrapper.
        kwargs = {}
        kwargs["device"] = data.device
        kwargs["dtype"] = data.dtype
        kwargs["layout"] = data.layout
        kwargs["requires_grad"] = requires_grad
        kwargs["dispatch_sizes_strides_policy"] = "strides"
        kwargs["dispatch_layout"] = True
        return torch.Tensor._make_wrapper_subclass(cls, data.size(), **kwargs)  # type: ignore[attr-defined]

    def _preprocess_data(self, data, mask):
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
        self._masked_data = data
        self._masked_mask = mask

    def _validate_members(self):
        data = self._masked_data
        mask = self._masked_mask
        if type(data) != type(mask):
            raise TypeError(f"data and mask must have the same type. Got {type(data)} and {type(mask)}")
        if data.layout not in {torch.strided, torch.sparse_coo, torch.sparse_csr}:
            raise TypeError(f"data layout of {data.layout} is not supported.")
        if data.layout == torch.sparse_coo:
            self.masked_layout = torch.sparse_coo
            if not _tensors_match(data.indices(), mask.indices(), exact=True):
                raise ValueError("data and mask are both sparse COO tensors but do not have the same indices.")
        elif data.layout == torch.sparse_csr:
            self.masked_layout = torch.sparse_csr
            if not _tensors_match(
                data.crow_indices(), mask.crow_indices(), exact=True
            ) or not _tensors_match(data.col_indices(), mask.col_indices(), exact=True):
                raise ValueError("data and mask are both spares CSR tensors but do not share either crow or col indices.")
        else:
            self.masked_layout = torch.strided
        if not torch.is_tensor(data):
            raise TypeError("data must be a tensor.")
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
            raise TypeError("{data.dtype} is not supported in MaskedTensor.")
        if data.dim() != mask.dim():
            raise ValueError("data.dim() must equal mask.dim()")
        if data.size() != mask.size():
            raise ValueError("data.size() must equal mask.size()")
        if mask.requires_grad:
            raise ValueError("mask cannot have requires_grad=True")

    def __init__(self, data, mask, requires_grad=False):
        self._preprocess_data(data, mask)
        self._validate_members()

    def _set_data_mask(self, data, mask):
        self._masked_data = data
        self._masked_mask = mask
        self._validate_members()

    def __repr__(self):
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

        if func in [torch.Tensor.where, torch.where]:
            _check_args_kwargs_length(args, kwargs, "__torch_function__, torch.where", len_args=3, len_kwargs=0)
            return _MaskedWhere.apply(*args)
        if func is torch.Tensor.contiguous:
            return _MaskedContiguous.apply(args[0])
        if func is torch.Tensor.to_dense:
            return _MaskedToDense.apply(args[0])
        if func is torch.Tensor.to_sparse:
            return _MaskedToSparse.apply(args[0])
        if func is torch.Tensor.to_sparse_csr:
            return _MaskedToSparseCsr.apply(args[0])
        if not all(issubclass(cls, t) for t in types):
            return NotImplemented
        with torch._C.DisableTorchFunction():
            ret = func(*args, **kwargs)
            if func in get_default_nowrap_functions():
                return ret
            else:
                return torch._tensor._convert(ret, cls)

    @classmethod
    def unary(cls, fn, data, mask):
        return MaskedTensor(fn(data), mask)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        func = func.overloadpacket

        from .passthrough import apply_pass_through_fn, is_pass_through_fn

        if is_pass_through_fn(func):
            return apply_pass_through_fn(func, *args, **kwargs)

        from .unary import apply_native_unary, is_native_unary

        if is_native_unary(func):
            return apply_native_unary(func, *args, **kwargs)

        from .binary import apply_native_binary, is_native_binary

        if is_native_binary(func):
            return apply_native_binary(func, *args, **kwargs)

        if func in [torch.ops.aten.mm, torch.ops.aten.bmm]:
            _check_args_kwargs_length(args, kwargs, f"__torch_dispatch__, {func}", len_args=2, len_kwargs=0)
            return cls.matmul(args[0], args[1], func)  # type: ignore[call-arg]

        # Doesn't work for addmm where the first argument is a Tensor
        data = _get_data(args[0])
        mask = _maybe_get_mask(args[0])
        if func is torch.ops.aten.stride:
            return None
        if func is torch.ops.prim.layout:
            return data.layout
        if func is torch.ops.aten.is_contiguous:
            if data.is_sparse:
                raise ValueError(
                    "MaskedTensors with sparse data do not have is_contiguous"
                )
            return data.is_contiguous()
        if func is torch.ops.aten.contiguous:
            if data.is_sparse:
                raise ValueError(
                    "MaskedTensors with sparse data do not have contiguous"
                )
            return _MaskedContiguous.apply(args[0])
        if func is torch.ops.aten.new_empty_strided:
            _check_args_kwargs_length(args, kwargs, f"__torch_dispatch__, {func}", len_args=3)
            if tuple(args[1]) != tuple(data.size()):
                raise ValueError(f"__torch_dispatch__, {func.name}: args[1] expected to be the same as data.size()")
            if tuple(args[2]) != tuple(data.stride()):
                raise ValueError(f"__torch_dispatch__, {func.name}: args[2] expected to be the same as data.stride()")
            return MaskedTensor(func(data, args[1], args[2], **kwargs), mask)
        if func is torch.ops.aten._local_scalar_dense:
            if not mask:
                raise ValueError("__torch_dispatch__, {func.name}: expected a mask tensor")
            return func(data)
        if func is torch.ops.aten._to_copy:
            return MaskedTensor(func(data, *args[1:], **kwargs), mask)
        if func is torch.ops.aten.new_empty_strided:
            _check_args_kwargs_length(args, kwargs, f"__torch_dispatch__, {func}", len_args=3)
            if tuple(args[1]) != tuple(data.size()):
                raise ValueError(f"__torch_dispatch__, {func.name}: args[1] expected to be the same as data.size()")
            if tuple(args[2]) != tuple(data.stride()):
                raise ValueError(f"__torch_dispatch__, {func.name}: args[2] expected to be the same as data.stride()")
            return MaskedTensor(func(data, args[1], args[2], **kwargs), mask)
        if func in [torch.ops.aten.detach, torch.ops.aten.clone]:
            _check_args_kwargs_length(args, kwargs, f"__torch_dispatch__, {func}", len_args=1, len_kwargs=0)
            return MaskedTensor(func(data), mask)
        if func is torch.ops.aten._softmax:
            _check_args_kwargs_length(args, kwargs, f"__torch_dispatch__, {func}", len_args=3, len_kwargs=0)
            input_data = data.masked_fill(~mask, float("-inf"))
            result_data = func(input_data, args[1], args[2])
            return MaskedTensor(result_data, mask)
        if func in [torch.ops.aten.ones_like]:
            _check_args_kwargs_length(args, kwargs, f"__torch_dispatch__, {func}", len_args=1)
            res_data = func(data, **kwargs)
            return MaskedTensor(res_data, mask)
        if func is torch.ops.aten._softmax_backward_data:
            _check_args_kwargs_length(args, kwargs, f"__torch_dispatch__, {func}", len_args=4)
            grad = args[0]
            output = args[1]
            dim = args[2]
            input_dtype = args[3]
            if is_masked_tensor(grad) and is_masked_tensor(output):
                if not _masks_match(grad, output):
                    raise ValueError("__torch_dispatch__, {func}: expected the masks of grad and output to match")
                grad_data = _get_data(grad).masked_fill(~_maybe_get_mask(grad), 1)
                output_data = _get_data(output).masked_fill(~_maybe_get_mask(output), 0)
                new_grad_data = torch.ops.aten._softmax_backward_data(
                    grad_data, output_data, dim, input_dtype
                )
                res = MaskedTensor(new_grad_data, _maybe_get_mask(grad))
                return res
        if func is torch.ops.aten.copy_:
            _check_args_kwargs_length(args, kwargs, f"__torch_dispatch__, {func}", len_args=2)
            if not _masks_match(mask, _maybe_get_mask(args[1])):
                raise ValueError("args[0] mask and args[1] mask must match but do not")
            func(data, _get_data(args[1]))
            return args[0]
        if func in [torch.ops.aten.where]:
            _check_args_kwargs_length(args, kwargs, f"__torch_dispatch__, {func}", len_args=3, len_kwargs=0)
            if not torch.is_tensor(args[0]):
                raise ValueError("__torch_dispatch__, {func}: expected args[0] to be a tensor")
            mx = args[1]
            my = args[2]
            if not is_masked_tensor(mx):
                mx = MaskedTensor(mx, torch.ones_like(mx, dtype=torch.bool))
            if not is_masked_tensor(my):
                my = MaskedTensor(my, torch.ones_like(my, dtype=torch.bool))
            new_data = func(args[0], mx.get_data(), my.get_data())
            new_mask = func(args[0], mx.get_mask(), my.get_mask())
            return MaskedTensor(new_data, new_mask)
        if func is torch.ops.aten.to_sparse:
            _check_args_kwargs_length(args, kwargs, f"__torch_dispatch__, {func}", len_args=1, len_kwargs=0)
            if not torch.is_tensor(args[0]):
                raise TypeError("__torch_dispatch__, {func}: expected args[0] to be a tensor")
            mt = args[0]
            if not is_masked_tensor(mt):
                mt = MaskedTensor(mt, torch.ones_like(mt, dtype=torch.bool))
            if mt.is_sparse_coo():
                return mt
            new_mask = func(mask).coalesce()
            new_data = data.sparse_mask(new_mask)
            return MaskedTensor(new_data, new_mask)
        if func is torch.ops.aten.to_sparse_csr:
            _check_args_kwargs_length(args, kwargs, f"__torch_dispatch__, {func}", len_args=1, len_kwargs=0)
            if not torch.is_tensor(args[0]):
                raise ValueError("__torch_dispatch__, {func}: expected args[0] to be a tensor")
            mt = args[0]
            if not is_masked_tensor(mt):
                mt = MaskedTensor(mt, torch.ones_like(mt).bool())
            if mt.is_sparse_csr():
                return mt
            new_mask = func(mask)
            new_data = data.sparse_mask(new_mask)
            return MaskedTensor(new_data, new_mask)
        if func in [torch.ops.aten._to_dense]:
            _check_args_kwargs_length(args, kwargs, f"__torch_dispatch__, {func}", len_args=1, len_kwargs=0)
            if not torch.is_tensor(args[0]):
                raise ValueError("__torch_dispatch__, {func}: expected args[0] to be a tensor")
            mt = args[0]
            if not is_masked_tensor(mt):
                mt = MaskedTensor(mt, torch.ones_like(mt).bool())
            new_data = func(data)
            new_mask = func(mask)
            return MaskedTensor(new_data, new_mask)
        if func is torch.ops.aten._indices:
            # Assumes data is sparse
            _check_args_kwargs_length(args, kwargs, f"__torch_dispatch__, {func}", len_args=1, len_kwargs=0)
            return MaskedTensor(data.indices(), torch.ones_like(data.indices()).bool())
        if func is torch.ops.aten._values:
            # Assumes data is sparse
            _check_args_kwargs_length(args, kwargs, f"__torch_dispatch__, {func}", len_args=1, len_kwargs=0)
            data = data.values()
            return MaskedTensor(data, torch.ones_like(data).bool())
        if func is torch.ops.aten._sparse_coo_tensor_with_dims_and_tensors:
            new_args = list(args)
            if is_masked_tensor(args[-1]):
                new_args[-1] = args[-1]._masked_data
            if is_masked_tensor(args[-2]):
                new_args[-2] = args[-2]._masked_data

            new_data = func(*new_args, **kwargs)
            new_args[-1] = torch.ones_like(new_args[-1])
            new_mask = func(*new_args, **kwargs).bool()

            return MaskedTensor(new_data, new_mask)
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
        return self._masked_data

    def get_mask(self):
        return self._masked_mask

    def layout(self):
        return self.masked_layout

    def is_sparse_coo(self):
        return self.layout() == torch.sparse_coo

    def is_sparse_csr(self):
        return self.layout() == torch.sparse_csr

    # Update later to support more sparse layouts
    def is_sparse(self):
        return self.is_sparse_coo() or self.is_sparse_csr()
