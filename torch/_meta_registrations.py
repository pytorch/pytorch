# mypy: allow-untyped-defs
import math
from collections.abc import Sequence
from enum import Enum
from functools import wraps
from typing import Callable, Optional, TypeVar, Union
from typing_extensions import ParamSpec

import torch
import torch._prims_common as utils
from torch import SymBool, SymFloat, Tensor
from torch._decomp import (
    _add_op_to_registry,
    _convert_out_params,
    global_decomposition_table,
    meta_table,
)
from torch._ops import OpOverload
from torch._prims import _prim_elementwise_meta, ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND
from torch._prims_common import (
    BoolLike,
    corresponding_complex_dtype,
    corresponding_real_dtype,
    elementwise_dtypes,
    ELEMENTWISE_TYPE_PROMOTION_KIND,
    FloatLike,
    IntLike,
    make_contiguous_strides_for,
    Number,
    suggest_memory_format,
    TensorLike,
)
from torch._prims_common.wrappers import (
    _maybe_convert_to_dtype,
    _maybe_resize_out,
    _resize_output_check,
    _safe_copy_out,
    out_wrapper,
)
from torch._refs import _broadcast_shapes, _maybe_broadcast
from torch.fx.experimental import _config as exp_config
from torch.utils import _pytree as pytree


_T = TypeVar("_T")
_P = ParamSpec("_P")

aten = torch.ops.aten

_meta_lib_dont_use_me_use_register_meta = torch.library.Library("aten", "IMPL", "Meta")
MODE_SUM, MODE_MEAN, MODE_MAX = range(3)


def register_meta(op) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    def wrapper(fn):
        fn = _convert_out_params(fn)

        def register(op):
            _add_op_to_registry(meta_table, op, fn)

        pytree.tree_map_(register, op)
        return fn

    return wrapper


def elementwise_meta(
    *args,
    type_promotion: ELEMENTWISE_TYPE_PROMOTION_KIND,
):
    # Perform type promotion, as this is expected from prim_metafunction
    _, result_dtype = utils.elementwise_dtypes(
        *args,
        type_promotion_kind=type_promotion,
    )
    args = [_maybe_convert_to_dtype(x, result_dtype) for x in args]

    # Broadcast
    args = _maybe_broadcast(*args)

    # Perform prim checks
    return _prim_elementwise_meta(
        *args, type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT
    )


def toRealValueType(dtype):
    from_complex = {
        torch.complex32: torch.half,
        torch.cfloat: torch.float,
        torch.cdouble: torch.double,
    }
    return from_complex.get(dtype, dtype)


def check_inplace_broadcast(self_shape, *args_shape):
    broadcasted_shape = tuple(_broadcast_shapes(self_shape, *args_shape))
    torch._check(
        broadcasted_shape == self_shape,
        lambda: f"output with shape {self_shape} doesn't match the broadcast shape {broadcasted_shape}",
    )


@register_meta([aten.linspace, aten.logspace])
@out_wrapper()
def meta_linspace_logspace(
    start,
    end,
    steps,
    base=None,
    dtype=None,
    device=None,
    layout=torch.strided,
    pin_memory=False,
    requires_grad=False,
):
    if isinstance(start, torch.Tensor):
        torch._check(
            start.dim() == 0,
            lambda: "linspace only supports 0-dimensional start and end tensors",
        )
    if isinstance(end, torch.Tensor):
        torch._check(
            end.dim() == 0,
            lambda: "linspace only supports 0-dimensional start and end tensors",
        )

    if any(isinstance(arg, complex) for arg in (start, end, steps)):
        default_complex_dtype = utils.corresponding_complex_dtype(
            torch.get_default_dtype()
        )
        if dtype is None:
            dtype = default_complex_dtype
        else:
            torch._check(
                utils.is_complex_dtype(dtype),
                lambda: f"linspace(): inferred dtype {default_complex_dtype} can't be safely cast to passed dtype {dtype}",
            )
    else:
        dtype = dtype or torch.get_default_dtype()
    assert isinstance(dtype, torch.dtype)

    # steps does not participate in the computation of the dtype
    torch._check_type(
        isinstance(steps, IntLike),
        lambda: f"received an invalid combination of arguments - got \
({type(start).__name__}, {type(end).__name__}, {type(steps).__name__})",
    )
    assert isinstance(steps, IntLike)  # for mypy
    torch._check(steps >= 0, lambda: "number of steps must be non-negative")

    return torch.empty(
        (steps,),  # type: ignore[arg-type]
        dtype=dtype,
        layout=layout,
        device="meta",
        pin_memory=pin_memory,
        requires_grad=requires_grad,
    )


@register_meta([aten.take.default, aten.take.out])
@out_wrapper()
def meta_take(self, index):
    # Type and device checks
    torch._check(
        index.dtype == torch.long,
        lambda: f"take(): Expected a long tensor for index, but got {index.dtype}",
    )
    # Index checks
    torch._check_index(
        not (self.numel() == 0 and index.numel() != 0),
        lambda: "take(): tried to take from an empty tensor",
    )
    return self.new_empty(index.shape)


@register_meta([aten.linalg_cross.default, aten.linalg_cross.out])
@out_wrapper()
def linalg_cross(self, other, *, dim=-1):
    x_d = self.ndim
    y_d = other.ndim
    torch._check(
        x_d == y_d,
        lambda: "linalg.cross: inputs must have the same number of dimensions.",
    )
    torch._check(
        self.size(dim) == 3 and other.size(dim) == 3,
        lambda: (
            f"linalg.cross: inputs dimension {dim} must have length 3. "
            f"Got {self.size(dim)} and {other.size(dim)}"
        ),
    )
    out_shape = _broadcast_shapes(self.shape, other.shape)
    return self.new_empty(out_shape)


@register_meta(aten.linalg_matrix_exp)
@out_wrapper()
def linalg_matrix_exp(self):
    squareCheckInputs(self, "linalg.matrix_exp")
    checkFloatingOrComplex(self, "linalg.matrix_exp")
    return torch.empty_like(self, memory_format=torch.contiguous_format)


@register_meta(
    [aten.cummax.default, aten.cummax.out, aten.cummin.default, aten.cummin.out]
)
@out_wrapper("values", "indices")
def cummaxmin(self, dim):
    values = torch.empty(self.shape, device=self.device, dtype=self.dtype)
    indices = torch.empty(self.shape, device=self.device, dtype=torch.int64)
    if self.numel() != 0 and self.ndim != 0:
        # Checks that dim is within bounds
        maybe_wrap_dim(dim, self.ndim)
    return values, indices


@register_meta([aten.logcumsumexp.default, aten.logcumsumexp.out])
@out_wrapper()
def logcumsumexp(self, dim):
    # Checks that dim is within bounds
    maybe_wrap_dim(dim, self.ndim)
    return torch.empty_like(self, memory_format=torch.contiguous_format)


# Stride-related code from _exec_fft in aten/src/ATen/native/mkl/SpectralOps.cpp
# and aten/src/ATen/cuda/SpectralOps.cpp
#
# Although the actual FFT launch is different, all the permuting code appears
# to be the same
def _exec_fft(out, self, out_sizes, dim, *, forward):
    ndim = self.ndim
    signal_ndim = len(dim)
    batch_dims = ndim - signal_ndim

    # Permute dimensions so batch dimensions come first, and in stride order
    dim_permute = list(range(ndim))

    is_transformed_dim = [False for _ in range(ndim)]
    for d in dim:
        is_transformed_dim[d] = True

    # std::partition
    left, right = [], []
    for d in dim_permute:
        if not is_transformed_dim[d]:
            left.append(d)
        else:
            right.append(d)
    dim_permute = left + right
    batch_end = len(left)

    self_strides = self.stride()
    tmp = dim_permute[:batch_end]
    tmp.sort(key=lambda x: self_strides[x], reverse=True)
    dim_permute = tmp + dim_permute[batch_end:]
    input = self.permute(dim_permute)

    # Collapse batch dimensions into a single dimension
    batched_sizes = [-1] + list(input.shape[batch_dims:])
    input = input.reshape(batched_sizes)

    batch_size = input.size(0)
    batched_sizes[0] = batch_size
    batched_out_sizes = list(batched_sizes)
    for i in range(len(dim)):
        batched_out_sizes[i + 1] = out_sizes[dim[i]]
    out.resize_(batched_out_sizes, memory_format=torch.contiguous_format)

    # Inplace reshaping to original batch shape and inverting the dimension permutation
    out_strides = [0 for _ in range(ndim)]
    batch_numel = 1
    i = batch_dims - 1
    while i >= 0:
        out_strides[dim_permute[i]] = batch_numel * out.stride(0)
        batch_numel *= out_sizes[dim_permute[i]]
        i -= 1
    for i in range(batch_dims, ndim):
        out_strides[dim_permute[i]] = out.stride(1 + (i - batch_dims))
    out.as_strided_(out_sizes, out_strides, out.storage_offset())

    return out


def _sort_dims(self: Tensor, dim: list[int], exclude_last: bool = False):
    sorted_dims = list(dim)
    self_strides = self.stride()
    sorted_dims[: len(sorted_dims) - int(exclude_last)].sort(
        key=lambda i: self_strides[i]
    )
    return sorted_dims


# See _fft_c2c_cufft in aten/src/ATen/native/cuda/SpectralOps.cpp
# and _fft_c2c_mkl in aten/src/ATen/native/mkl/SpectralOps.cpp
@register_meta([aten._fft_c2c.default, aten._fft_c2c.out])
@out_wrapper()
def meta_fft_c2c(self, dim, normalization, forward):
    torch._check(self.dtype.is_complex)
    if not dim:
        return self.clone()

    sorted_dims = _sort_dims(self, dim)
    out = self.new_empty(self.size())
    return _exec_fft(out, self, self.size(), sorted_dims, forward=forward)


cufft_max_ndim = 3


def use_optimized_cufft_path(dim: list[int]):
    if len(dim) > cufft_max_ndim or (len(dim) >= 2 and dim[0] == 0 and dim[1] == 1):
        return False
    else:
        return True


@register_meta([aten._fft_r2c.default, aten._fft_r2c.out])
@out_wrapper()
def meta_fft_r2c(self, dim, normalization, onesided):
    torch._check(self.dtype.is_floating_point)
    input_sizes = list(self.size())
    out_sizes = list(input_sizes)
    last_dim = dim[-1]
    last_dim_halfsize = input_sizes[last_dim] // 2 + 1
    onesided_sizes = list(input_sizes)
    onesided_sizes[last_dim] = last_dim_halfsize

    if onesided:
        out_sizes[last_dim] = last_dim_halfsize

    if device_hint(self) == "cuda" or device_hint(self) == "xpu":
        # _fft_r2c_cufft in aten/src/ATen/native/cuda/SpectralOps.cpp
        # _fft_r2c_xpu in torch-xpu-ops/src/ATen/native/xpu/SpectralOps.cpp
        output = self.new_empty(
            out_sizes, dtype=utils.corresponding_complex_dtype(self.dtype)
        )

        working_tensor = self
        if device_hint(self) == "cuda" and use_optimized_cufft_path(dim):
            _exec_fft(output, working_tensor, out_sizes, dim, forward=True)
        else:
            # First do the R2C transform on the last dimension
            target_sizes = out_sizes if len(dim) == 1 else onesided_sizes
            _exec_fft(output, working_tensor, target_sizes, [last_dim], forward=True)
            if len(dim) > 1:
                working_tensor = self.new_empty(
                    out_sizes, dtype=utils.corresponding_complex_dtype(self.dtype)
                )

            # Then any remaining C2C transforms
            sorted_dims = dim[:-1]
            while sorted_dims:
                output, working_tensor = working_tensor, output
                strides = working_tensor.stride()
                sorted_dims.sort(
                    key=lambda i: strides[i], reverse=True
                )  # NB reverse!  Not sure if this is og bug
                max_dims = min(cufft_max_ndim, len(sorted_dims))
                last_dims = sorted_dims[len(sorted_dims) - max_dims :]
                _exec_fft(
                    output, working_tensor, onesided_sizes, last_dims, forward=True
                )
                sorted_dims = sorted_dims[: len(sorted_dims) - max_dims]

        if not onesided:
            if output.size(last_dim) != out_sizes[last_dim]:
                working_tensor.resize_(out_sizes, memory_format=torch.contiguous_format)
                output = working_tensor

        return output

    else:
        return self.new_empty(
            out_sizes, dtype=utils.corresponding_complex_dtype(self.dtype)
        )


@register_meta(aten.randperm.generator_out)
def meta_randperm(n, *, generator=None, out):
    return _maybe_resize_out(out, torch.Size([n]))


@register_meta(aten.randperm.default)
def meta_randperm_default(
    n,
    *,
    dtype=torch.long,
    layout=None,
    device=None,
    pin_memory=None,
):
    return torch.empty(
        n, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory
    )


@register_meta([aten.randint.default, aten.randint.out])
@out_wrapper()
def meta_randint(
    high,
    size,
    *,
    dtype=torch.long,
    layout=None,
    device=None,
    pin_memory=None,
):
    low = 0
    torch._check(
        high > low,
        lambda: f"random_ expects 'from' to be less than 'to', but got from={low} >= to={high}",
    )
    return torch.empty(
        size, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory
    )


@register_meta([aten.randint.low, aten.randint.low_out])
@out_wrapper()
def meta_randint_low(
    low,
    high,
    size,
    *,
    dtype=torch.long,
    layout=None,
    device=None,
    pin_memory=None,
):
    torch._check(
        high > low,
        lambda: f"random_ expects 'from' to be less than 'to', but got from={low} >= to={high}",
    )
    return torch.empty(
        size, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory
    )


@register_meta([aten.rand.default, aten.rand.out])
@out_wrapper()
def meta_rand_default(size, *, dtype=None, layout=None, device=None, pin_memory=None):
    return torch.empty(
        size, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory
    )


@register_meta([aten._fft_c2r.default, aten._fft_c2r.out])
@out_wrapper()
def meta_fft_c2r(self: Tensor, dim: list[int], normalization: int, lastdim: int):
    # _fft_c2r_mkl
    torch._check(self.dtype.is_complex)

    if device_hint(self) == "cuda":
        out_sizes = list(self.size())
        out_sizes[dim[-1]] = lastdim

        output = self.new_empty(out_sizes, dtype=toRealValueType(self.dtype))

        if use_optimized_cufft_path(dim):
            return _exec_fft(
                output,
                self.clone(memory_format=torch.contiguous_format),
                out_sizes,
                dim,
                forward=False,
            )
        else:
            # First complete any C2C transforms
            if len(dim) > 1:
                temp = meta_fft_c2c(self, dim[:-1], 0, lastdim)  # fft_norm_mode::none
            else:
                temp = self.clone(memory_format=torch.contiguous_format)
            return _exec_fft(output, temp, out_sizes, [dim[-1]], forward=False)

    else:
        input = self
        if len(dim) > 1:
            c2c_dims = dim[:-1]
            input = meta_fft_c2c(self, c2c_dims, normalization, forward=False)
            dim = dim[-1:]

        out_sizes = list(input.size())
        out_sizes[dim[-1]] = lastdim
        out = self.new_empty(out_sizes, dtype=toRealValueType(self.dtype))
        return _exec_fft(out, input, out_sizes, dim, forward=False)


@register_meta(aten.copy_.default)
def meta_copy_(self, src, non_blocking=False):
    # This code simulates the original decomp from inductor,
    # which runs most of the meta checks that we care about.
    # In theory, we should make this more robust by carefully
    # auditing our C++ copy_() kernel and copying the checks here.
    from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols

    # TODO: Ideally, we'd insert a deferred runtime assert here, but if we are
    # calling an actual copy_, you'll get that automatically
    # https://github.com/pytorch/pytorch/issues/122477
    if (
        not free_unbacked_symbols(self) and torch._debug_has_internal_overlap(self) == 1
    ):  # 1 == MemOverlap::Yes
        raise RuntimeError(
            "more than one element of the written-to tensor refers to a single memory location"
        )

    if isinstance(src, Tensor):
        intermediate = src.to(self, non_blocking)
        if self.size() != intermediate.size():
            aten.expand_copy.default(intermediate, self.size())
    return self


def inferUnsqueezeGeometry(tensor, dim):
    result_sizes = list(tensor.size())
    result_strides = list(tensor.stride())
    new_stride = 1 if dim >= tensor.dim() else result_sizes[dim] * result_strides[dim]
    result_sizes.insert(dim, 1)
    result_strides.insert(dim, new_stride)
    return result_sizes, result_strides


@register_meta(aten.unsqueeze_.default)
def meta_unsqueeze_(self, dim):
    dim = maybe_wrap_dim(dim, self.dim() + 1)
    g_sizes, g_strides = inferUnsqueezeGeometry(self, dim)
    self.as_strided_(g_sizes, g_strides)
    return self


@register_meta(aten._sparse_semi_structured_linear)
def meta_sparse_structured_linear(
    input: Tensor,
    weight: Tensor,
    _meta: Tensor,
    bias: Optional[Tensor] = None,
    _activation_opt: Optional[str] = None,
    out_dtype: Optional[torch.dtype] = None,
):
    output_sizes = list(input.shape)
    if bias is not None:
        assert weight.size(0) == bias.size(0), "output size mismatch"
    assert weight.size(1) == input.size(-1) / 2
    output_sizes[-1] = weight.size(0)

    # see: https://github.com/pytorch/pytorch/pull/114477#issuecomment-1830121375
    # We assume that we have already squashed the inputs into a 2-D tensor
    # Then, as the output is transposed, we need to propagate the transposed
    # stride information to the output tensor
    assert len(input.shape) == 2, "we can only handle the squashed input case"
    transposed_strides = (1, input.size(0))

    if out_dtype is not None:
        assert input.dtype == torch.int8 and out_dtype == torch.int32, (
            "out_dtype is only supported for i8i8->i32 linear operator"
        )
    output = input.new_empty(
        output_sizes,
        dtype=input.dtype if out_dtype is None else out_dtype,
    ).as_strided(output_sizes, transposed_strides)

    return output


@register_meta(aten._sparse_semi_structured_mm)
def meta_sparse_structured_mm(
    mat1: Tensor,
    mat1_meta: Tensor,
    mat2: Tensor,
    out_dtype: Optional[torch.dtype] = None,
):
    assert len(mat1.shape) == 2
    assert len(mat1_meta.shape) == 2
    assert len(mat2.shape) == 2
    assert mat1.size(1) == mat2.size(0) / 2
    output_sizes = [mat1.size(0), mat2.size(1)]

    if out_dtype is not None:
        assert mat2.dtype == torch.int8 and out_dtype == torch.int32, (
            "out_dtype is only supported for i8i8->i32 linear operator"
        )
    output = mat2.new_empty(
        output_sizes,
        dtype=mat2.dtype if out_dtype is None else out_dtype,
    )

    return output


@register_meta(aten._sparse_semi_structured_addmm)
def meta_sparse_structured_addmm(
    input: Tensor,
    mat1: Tensor,
    mat1_meta: Tensor,
    mat2: Tensor,
    *,
    alpha=1,
    beta=1,
    out_dtype: Optional[torch.dtype] = None,
):
    assert len(input.shape) == 1, (
        "only input broadcasted to columns of mat1 * mat2 product is supported"
    )
    assert len(mat1.shape) == 2
    assert len(mat1_meta.shape) == 2
    assert len(mat2.shape) == 2
    assert input.size(0) == mat1.size(0), (
        "only input broadcasted to columns of mat1 * mat2 product is supported"
    )
    assert mat1.size(1) == mat2.size(0) / 2
    output_sizes = [mat1.size(0), mat2.size(1)]

    if out_dtype is not None:
        assert mat2.dtype == torch.int8 and out_dtype == torch.int32, (
            "out_dtype is only supported for i8i8->i32 linear operator"
        )
    output = mat2.new_empty(
        output_sizes,
        dtype=mat2.dtype if out_dtype is None else out_dtype,
    )

    return output


@register_meta(aten._cslt_sparse_mm)
def meta__cslt_sparse_mm(
    compressed_A: torch.Tensor,
    dense_B: torch.Tensor,
    bias: Optional[Tensor] = None,
    alpha: Optional[Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    transpose_result: bool = False,
    alg_id: int = 0,
    split_k: int = 1,
    split_k_mode: int = -1,
):
    assert dense_B.dtype in {
        torch.float32,
        torch.float16,
        torch.bfloat16,
        torch.int8,
        torch.float8_e4m3fn,
    }, "_cslt_sparse_mm only supports fp16, bf16, int8, and fp8e4m3"
    assert compressed_A.dtype == dense_B.dtype, "inputs must have the same dtype"
    assert len(dense_B.shape) == 2, "_cslt_sparse_mm only supports 2d inputs"

    is_8bit_input_type = compressed_A.dtype in [torch.int8, torch.float8_e4m3fn]
    compression_factor = 10 if is_8bit_input_type else 9

    if is_8bit_input_type:
        assert not dense_B.is_contiguous(), (
            "dense input must be transposed for 8bit dtypes"
        )

    k = dense_B.size(0)
    n = dense_B.size(1)
    m = (compressed_A.numel() * 16) // (compression_factor * k)
    if bias is not None:
        assert m == bias.size(0)

    if out_dtype is not None:
        assert is_8bit_input_type and out_dtype in {
            torch.float16,
            torch.bfloat16,
            torch.int32,
            torch.float8_e4m3fn,
        }, (
            "out_dtype is not supported for {compressed_A.dtype} x {dense_B.dtype} -> {out_dtype} matmul!"
        )
    output_shape = (n, m) if transpose_result else (m, n)
    return dense_B.new_empty(output_shape, dtype=out_dtype)


@register_meta(aten.index_reduce.default)
def meta_index_reduce(
    self: Tensor,
    dim: int,
    index: Tensor,
    source: torch.Tensor,
    reduce: str,
    *,
    include_self: bool = True,
) -> Tensor:
    return torch.empty_like(self, memory_format=torch.contiguous_format)


@register_meta(aten.index_reduce_.default)
def meta_index_reduce_(
    self: Tensor,
    dim: int,
    index: Tensor,
    source: torch.Tensor,
    reduce: str,
    *,
    include_self: bool = True,
) -> Tensor:
    return self


# Implementations below are taken from https://github.com/albanD/subclass_zoo/blob/main/python_meta_tensor.py
@out_wrapper()
@register_meta(aten.index_select.default)
def meta_index_select(self, dim, index):
    result_size = list(self.size())
    if self.dim() > 0:
        result_size[dim] = index.numel()
    return self.new_empty(result_size)


@register_meta(aten.segment_reduce.default)
def meta_segment_reduce(
    data: Tensor,
    reduce: str,
    *,
    lengths: Optional[Tensor] = None,
    indices: Optional[Tensor] = None,
    offsets: Optional[Tensor] = None,
    axis: int = 0,
    unsafe: bool = False,
    initial=None,
) -> Tensor:
    if indices is not None:
        raise NotImplementedError(
            "segment_reduce(): indices based reduction is not supported yet."
        )

    def segment_reduce_lengths_tensor(lengths_shape):
        return torch.empty(
            lengths_shape + data.shape[axis + 1 :],
            dtype=data.dtype,
            device="meta",
            memory_format=torch.contiguous_format,
        )

    if lengths is not None:
        return segment_reduce_lengths_tensor(lengths.shape)
    # FIXME should probably check that lengths and offset aren't both set, but
    # the ATen implementation neglects this too
    if offsets is not None:
        # lengths == torch.diff(offsets)
        lengths_shape = offsets.shape[:-1] + (offsets.shape[-1] - 1,)
        return segment_reduce_lengths_tensor(lengths_shape)
    raise RuntimeError("segment_reduce(): Either lengths or offsets must be defined.")


@register_meta([aten.max.default, aten.max.unary_out])
@out_wrapper()
def meta_max(self):
    return self.new_empty(())


@register_meta(aten.max.dim)
def meta_max_dim(self, dim, keepdim=False):
    dim = utils.reduction_dims(self.shape, (dim,))
    output_shape = _compute_reduction_shape(self, dim, keepdim)
    return (
        self.new_empty(output_shape),
        self.new_empty(output_shape, dtype=torch.long),
    )


@register_meta([aten.min.default, aten.min.unary_out])
@out_wrapper()
def meta_min(self):
    return self.new_empty(())


@register_meta(aten.min.dim)
def meta_min_dim(self, dim, keepdim=False):
    dim = utils.reduction_dims(self.shape, (dim,))
    output_shape = _compute_reduction_shape(self, dim, keepdim)
    return (
        self.new_empty(output_shape),
        self.new_empty(output_shape, dtype=torch.long),
    )


@register_meta(aten.angle.default)
def meta_angle(self):
    if self.is_complex():
        result_dtype = corresponding_real_dtype(self.dtype)
    else:
        _, result_dtype = elementwise_dtypes(
            self,
            type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        )
    return torch.empty_like(self, dtype=result_dtype)


@register_meta(aten.angle.out)
def meta_angle_out(self, out):
    torch._resize_output_(out, self.size(), self.device)
    return out.copy_(torch.angle(self))


@register_meta(aten._assert_async.default)
def assert_async(val):
    return


@register_meta(aten._assert_async.msg)
def assert_async_meta(val, assert_msg):
    return


@register_meta(aten._print.default)
def print_meta(s):
    return


@register_meta(aten._make_dep_token.default)
def make_dep_token(
    *,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
    memory_format=None,
):
    return torch.empty(0, device="meta")


@register_meta(aten.sym_constrain_range.default)
def sym_constrain_range(size, min=None, max=None):
    # Avoid importing sympy at a module level
    from torch.fx.experimental.symbolic_shapes import constrain_range

    if isinstance(size, (SymFloat, SymBool)):
        raise ValueError("Constraining SymFloat or Symbool is nyi")
    constrain_range(size, min=min, max=max)


@register_meta(aten._functional_sym_constrain_range.default)
def functional_sym_constrain_range(size, min=None, max=None, dep_token=None):
    aten.sym_constrain_range(size, min=min, max=max)
    return dep_token


@register_meta(aten.sym_constrain_range_for_size.default)
def sym_constrain_range_for_size(size, min=None, max=None):
    # Avoid importing sympy at a module level
    from torch.fx.experimental.symbolic_shapes import _constrain_range_for_size

    if min is None and max is None:
        torch._check_is_size(size)
        return

    if isinstance(size, (SymFloat, SymBool)):
        raise ValueError("Constraining SymFloat or Symbool is nyi")
    if type(size) is int:
        if min is not None:
            torch._check(size >= min)
        if max is not None:
            torch._check(size <= max)
        return
    _constrain_range_for_size(size, min=min, max=max)


@register_meta(aten._functional_sym_constrain_range_for_size.default)
def functional_sym_constrain_range_for_size(size, min, max, dep_token):
    aten.sym_constrain_range_for_size(size, min=min, max=max)
    return dep_token


@register_meta(aten._functional_assert_async.msg)
def functional_assert_async_meta(val, assert_msg, dep_token):
    return dep_token


# From aten/src/ATen/native/LinearAlgebraUtils.h
def squareCheckInputs(self: Tensor, f_name: str):
    assert self.dim() >= 2, (
        f"{f_name}: The input tensor must have at least 2 dimensions."
    )
    assert self.size(-1) == self.size(-2), (
        f"{f_name}: A must be batches of square matrices, but they are {self.size(-2)} by {self.size(-1)} matrices"
    )


# Validates input shapes and devices
# for linear solve methods (solve, cholesky_solve, lu_solve, triangular_solve)
# From aten/src/ATen/native/LinearAlgebraUtils.h
def linearSolveCheckInputs(self: Tensor, A: Tensor, name: str):
    torch._check(
        self.device == A.device,
        lambda: (
            f"Expected b and A to be on the same device, but found b on "
            f"{self.device} and A on {A.device} instead."
        ),
    )

    torch._check(
        self.dtype == A.dtype,
        lambda: (
            f"Expected b and A to have the same dtype, but found b of type "
            f"{self.dtype} and A of type {A.dtype} instead."
        ),
    )

    torch._check(
        A.size(-1) == A.size(-2),
        lambda: (
            f"A must be batches of square matrices, "
            f"but they are {A.size(-2)} by {A.size(-1)} matrices"
        ),
    )

    torch._check(
        A.size(-1) == self.size(-2),
        lambda: (
            f"Incompatible matrix sizes for {name}: each A "
            f"matrix is {A.size(-1)} by {A.size(-1)}"
            f" but each b matrix is {self.size(-2)} by {self.size(-1)}"
        ),
    )


# From aten/src/ATen/native/LinearAlgebraUtils.h
def checkFloatingOrComplex(
    t: Tensor,
    f_name: str,
    allow_low_precision_dtypes: bool = True,
):
    dtype = t.dtype
    torch._check(
        t.is_floating_point() or t.is_complex(),
        lambda: f"{f_name}: Expected a floating point or complex tensor as input. Got {dtype}",
    )
    if not allow_low_precision_dtypes:
        torch._check(
            dtype in (torch.float, torch.double, torch.cfloat, torch.cdouble),
            lambda: f"{f_name}: Low precision dtypes not supported. Got {dtype}",
        )


# From aten/src/ATen/native/LinearAlgebraUtils.h
def checkIsMatrix(A: Tensor, f_name: str, arg_name: str = "A"):
    torch._check(
        A.dim() >= 2,
        lambda: f"{f_name}: The input tensor {arg_name} must have at least 2 dimensions.",
    )


def checkInputsSolver(A: Tensor, B: Tensor, left: bool, f_name: str):
    squareCheckInputs(A, f_name)
    checkIsMatrix(B, f_name)
    torch._check(
        A.size(-2) == B.size(-2) if left else A.size(-1) == B.size(-1),
        lambda: (
            f"{f_name}: Incompatible shapes of A and B for the equation "
            f"{'AX = B' if left else 'XA = B'}"
            f" ({A.size(-2)}x{A.size(-1)} and {B.size(-2)}x{B.size(-1)})"
        ),
    )


def checkSameDevice(
    fn_name: str,
    result: Tensor,
    input: Tensor,
    result_name: str = "result",
):
    torch._check(
        result.device == input.device,
        lambda: (
            f"{fn_name}: Expected {result_name} and input tensors to be on the same device, but got "
            f"{result_name} on {result.device} and input on {input.device}"
        ),
    )


def checkUplo(UPLO: str):
    UPLO_uppercase = UPLO.upper()
    torch._check(
        len(UPLO) == 1 and (UPLO_uppercase == "U" or UPLO_uppercase == "L"),
        lambda: f"Expected UPLO argument to be 'L' or 'U', but got {UPLO}",
    )


@register_meta([aten._linalg_eigh.default, aten._linalg_eigh.eigenvalues])
@out_wrapper("eigenvalues", "eigenvectors")
def meta__linalg_eigh(A: Tensor, UPLO: str = "L", compute_v: bool = True):
    squareCheckInputs(A, "linalg.eigh")
    checkUplo(UPLO)

    shape = list(A.shape)
    if compute_v:
        vecs = A.new_empty(shape)
        vecs.as_strided_(shape, make_contiguous_strides_for(shape, row_major=False))
    else:
        vecs = A.new_empty([0])

    shape.pop()
    vals = A.new_empty(shape, dtype=toRealValueType(A.dtype))

    return vals, vecs


@register_meta([aten._linalg_eigvals.default, aten.linalg_eigvals.out])
@out_wrapper()
def meta__linalg_eigvals(input: Tensor) -> Tensor:
    squareCheckInputs(input, "linalg.eigvals")
    complex_dtype = (
        input.dtype
        if utils.is_complex_dtype(input.dtype)
        else utils.corresponding_complex_dtype(input.dtype)
    )
    return input.new_empty(input.shape[:-1], dtype=complex_dtype)


@register_meta([aten.linalg_eig])
@out_wrapper("eigenvalues", "eigenvectors")
def meta_linalg_eig(input: Tensor):
    squareCheckInputs(input, "linalg.eig")
    complex_dtype = (
        input.dtype
        if utils.is_complex_dtype(input.dtype)
        else utils.corresponding_complex_dtype(input.dtype)
    )
    values = input.new_empty(input.shape[:-1], dtype=complex_dtype)
    vectors = input.new_empty(input.shape, dtype=complex_dtype)
    return values, vectors


def cloneBatchedColumnMajor(src: Tensor) -> Tensor:
    return src.mT.clone(memory_format=torch.contiguous_format).transpose(-2, -1)


@register_meta(aten._cholesky_solve_helper)
@out_wrapper()
def _cholesky_solve_helper(self: Tensor, A: Tensor, upper: bool) -> Tensor:
    return cloneBatchedColumnMajor(self)


@register_meta(aten.cholesky_solve)
@out_wrapper()
def cholesky_solve(self: Tensor, A: Tensor, upper: bool = False) -> Tensor:
    torch._check(
        self.ndim >= 2,
        lambda: f"b should have at least 2 dimensions, but has {self.ndim} dimensions instead",
    )
    torch._check(
        A.ndim >= 2,
        lambda: f"u should have at least 2 dimensions, but has {A.ndim} dimensions instead",
    )
    self_broadcasted, A_broadcasted = _linalg_broadcast_batch_dims_name(
        self, A, "cholesky_solve"
    )
    return _cholesky_solve_helper(self_broadcasted, A_broadcasted, upper)


@register_meta(aten.cholesky)
@out_wrapper()
def cholesky(self: Tensor, upper: bool = False) -> Tensor:
    if self.numel() == 0:
        return torch.empty_like(self, memory_format=torch.legacy_contiguous_format)
    squareCheckInputs(self, "cholesky")
    return cloneBatchedColumnMajor(self)


@register_meta(aten.cholesky_inverse)
@out_wrapper()
def cholesky_inverse(self: Tensor, upper: bool = False) -> Tensor:
    squareCheckInputs(self, "cholesky_inverse")
    return cloneBatchedColumnMajor(self)


# From aten/src/ATen/native/BatchLinearAlgebra.cpp
@register_meta(aten.linalg_cholesky_ex.default)
def linalg_cholesky_ex(A: Tensor, upper: bool = False, check_errors: bool = False):
    squareCheckInputs(A, "linalg.cholesky")
    checkFloatingOrComplex(A, "linalg.cholesky")

    A_shape = A.shape
    ndim = len(A_shape)

    # L
    L_strides = make_contiguous_strides_for(A_shape, False)
    L = A.new_empty(A_shape)
    L.as_strided_(A_shape, L_strides)

    # infos
    infos = A.new_empty(A_shape[0 : ndim - 2], dtype=torch.int32)
    return L, infos


@register_meta(
    [aten.linalg_householder_product.default, aten.linalg_householder_product.out]
)
@out_wrapper()
def linalg_householder_product(input: Tensor, tau: Tensor) -> Tensor:
    torch._check(
        input.ndim >= 2,
        lambda: "torch.linalg.householder_product: input must have at least 2 dimensions.",
    )
    torch._check(
        input.size(-2) >= input.size(-1),
        lambda: "torch.linalg.householder_product: input.shape[-2] must be greater than or equal to input.shape[-1]",
    )
    torch._check(
        input.size(-1) >= tau.size(-1),
        lambda: "torch.linalg.householder_product: input.shape[-1] must be greater than or equal to tau.shape[-1]",
    )

    torch._check(
        input.ndim - tau.ndim == 1,
        lambda: (
            f"torch.linalg.householder_product: Expected tau to have one dimension less than input, "
            f"but got tau.ndim equal to {tau.ndim} and input.ndim is equal to {input.ndim}"
        ),
    )
    if input.ndim > 2:
        expected_batch_tau_shape = input.shape[:-2]
        actual_batch_tau_shape = tau.shape[:-1]
        torch._check(
            actual_batch_tau_shape == expected_batch_tau_shape,
            lambda: (
                f"torch.linalg.householder_product: Expected batch dimensions of tau to be "
                f"equal to input.shape[:-2], but got {actual_batch_tau_shape}"
            ),
        )

    torch._check(
        tau.dtype == input.dtype,
        lambda: (
            f"torch.linalg.householder_product: tau dtype {tau.dtype}"
            f" does not match input dtype {input.dtype}"
        ),
    )
    checkSameDevice("torch.linalg.householder_product", tau, input, "tau")

    return torch.empty_strided(
        size=input.shape,
        stride=make_contiguous_strides_for(input.shape, row_major=False),
        dtype=input.dtype,
        device=input.device,
    )


# From aten/src/ATen/native/BatchLinearAlgebra.cpp
@register_meta(aten.linalg_inv_ex.default)
def linalg_inv_ex_meta(A: Tensor, check_errors: bool = False):
    squareCheckInputs(A, "linalg.inv_ex")
    checkFloatingOrComplex(A, "linalg.inv_ex", allow_low_precision_dtypes=False)

    L = A.new_empty(A.shape)
    L.as_strided_(A.shape, make_contiguous_strides_for(A.shape, row_major=False))

    infos = A.new_empty(A.shape[:-2], dtype=torch.int32)
    return L, infos


@register_meta([aten.linalg_ldl_factor_ex.default, aten.linalg_ldl_factor_ex.out])
@out_wrapper("LD", "pivots", "info")
def linalg_ldl_factor_ex_meta(
    self: Tensor,
    *,
    hermitian: bool = False,
    check_errors: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    squareCheckInputs(self, "torch.linalg.ldl_factor_ex")
    checkFloatingOrComplex(self, "torch.linalg.ldl_factor_ex")
    LD = torch.empty_strided(
        size=self.shape,
        stride=make_contiguous_strides_for(self.shape, row_major=False),
        dtype=self.dtype,
        device=self.device,
    )
    pivots = self.new_empty(self.shape[:-1], dtype=torch.int)
    info = self.new_empty(self.shape[:-2], dtype=torch.int)
    return LD, pivots, info


@register_meta([aten.linalg_ldl_solve.default, aten.linalg_ldl_solve.out])
@out_wrapper()
def linalg_ldl_solve_meta(
    LD: Tensor,
    pivots: Tensor,
    B: Tensor,
    *,
    hermitian: bool = False,
) -> Tensor:
    squareCheckInputs(LD, "torch.linalg.ldl_solve")
    checkFloatingOrComplex(LD, "torch.linalg.ldl_solve")
    linearSolveCheckInputs(B, LD, "torch.linalg.ldl_solve")
    torch._check(
        B.ndim >= 2,
        lambda: (
            f"torch.linalg.ldl_solve: Expected B to have at least 2 dimensions, "
            f"but it has {B.ndim} dimensions instead"
        ),
    )
    expected_pivots_shape = LD.shape[:-1]
    torch._check(
        expected_pivots_shape == pivots.shape,
        lambda: (
            f"torch.linalg.ldl_solve: Expected LD.shape[:-1] and pivots.shape to be the same, "
            f"but got pivots with shape {pivots.shape} instead"
        ),
    )
    torch._check(
        utils.is_integer_dtype(pivots.dtype),
        lambda: f"torch.linalg.ldl_solve: Expected pivots to be integers. Got {pivots.dtype}",
    )
    torch._check(
        LD.dtype == B.dtype,
        lambda: f"torch.linalg.ldl_solve: LD dtype {LD.dtype} does not match b dtype {B.dtype}",
    )
    B_broadcast_size, _ = _linalg_broadcast_batch_dims(B, LD)
    return torch.empty_strided(
        size=B_broadcast_size,
        stride=make_contiguous_strides_for(B_broadcast_size, row_major=False),
        dtype=B.dtype,
        device=B.device,
    )


@register_meta([aten.linalg_lu.default, aten.linalg_lu.out])
@out_wrapper("P", "L", "U")
def linalg_lu_meta(A: Tensor, *, pivot: bool = True) -> tuple[Tensor, Tensor, Tensor]:
    torch._check(
        A.ndim >= 2,
        lambda: f"linalg.lu: Expected tensor with 2 or more dimensions. Got size: {A.shape} instead",
    )

    sizes = list(A.shape)
    m = sizes[-2]
    n = sizes[-1]
    k = min(m, n)

    sizes[-1] = m
    if pivot:
        P = A.new_empty(sizes)
    else:
        P = A.new_empty([0])

    sizes[-1] = k
    L = A.new_empty(sizes)

    sizes[-2] = k
    sizes[-1] = n
    U = A.new_empty(sizes)
    return P, L, U


@register_meta([aten.linalg_lu_factor_ex.default, aten.linalg_lu_factor_ex.out])
@out_wrapper("LU", "pivots", "info")
def linalg_lu_factor_ex_meta(
    A: Tensor,
    *,
    pivot: bool = True,
    check_errors: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    torch._check(
        A.ndim >= 2,
        lambda: f"torch.lu_factor: Expected tensor with 2 or more dimensions. Got size: {A.shape} instead",
    )

    sizes = list(A.shape)
    m = sizes[-2]
    n = sizes[-1]

    LU = torch.empty_strided(
        size=sizes,
        stride=make_contiguous_strides_for(sizes, row_major=False),
        dtype=A.dtype,
        device=A.device,
    )

    # Sets sizes to the size of pivots
    sizes.pop()
    sizes[-1] = min(m, n)
    pivots = A.new_empty(sizes, dtype=torch.int)

    # Sets sizes to the size of info
    sizes.pop()
    info = A.new_empty(sizes, dtype=torch.int)

    return LU, pivots, info


@register_meta([aten.linalg_lu_solve.default, aten.linalg_lu_solve.out])
@out_wrapper()
def linalg_lu_solve_meta(
    LU: Tensor,
    pivots: Tensor,
    B: Tensor,
    *,
    left: bool = True,
    adjoint: bool = False,
) -> Tensor:
    # dtype
    checkFloatingOrComplex(LU, "torch.linalg.lu_solve")
    torch._check(
        LU.dtype == B.dtype,
        lambda: (
            f"linalg.lu_solve: Expected LU and B to have the same dtype, "
            f"but found LU of type {LU.dtype} and B of type {B.dtype} instead"
        ),
    )
    torch._check(
        pivots.dtype == torch.int,
        lambda: "linalg.lu_solve: pivots should be a Tensor of scalar type torch.int32",
    )

    # matrix shapes
    squareCheckInputs(LU, "torch.linalg.lu_solve")
    checkInputsSolver(LU, B, left, "linalg.lu_solve")
    torch._check(
        LU.size(-1) == pivots.size(-1),
        lambda: "linalg.lu_solve: Number of pivots per batch should be same as the dimension of the matrix",
    )

    # batches
    torch._check(
        LU.shape[:-1] == pivots.shape,
        lambda: (
            f"linalg.lu_solve: Expected LU.shape[:-1] and pivots.shape to be the same, "
            f"but got pivots with shape {pivots.shape} instead"
        ),
    )

    B_broadcast_size, _ = _linalg_broadcast_batch_dims(B, LU)

    result = torch.empty_strided(
        size=B_broadcast_size,
        stride=make_contiguous_strides_for(B_broadcast_size, row_major=not left),
        dtype=B.dtype,
        device=B.device,
    )

    if result.numel() != 0 and not left:
        if result.is_complex():
            result = result.conj()

    return result


@register_meta(aten.lu_unpack)
@out_wrapper("P", "L", "U")
def lu_unpack_meta(
    LU: Tensor,
    pivots: Tensor,
    unpack_data: bool = True,
    unpack_pivots: bool = True,
) -> tuple[Tensor, Tensor, Tensor]:
    torch._check(
        LU.ndim >= 2,
        lambda: f"torch.lu_unpack: Expected tensor with 2 or more dimensions. Got size: {LU.shape} instead",
    )
    if unpack_pivots:
        torch._check(
            pivots.dtype == torch.int32,
            lambda: (
                "torch.lu_unpack: LU_pivots is expected to be a contiguous tensor of torch.int32 dtype.\n"
                "Note: this function is intended to be used with the output produced by torch.linalg.lu_factor"
            ),
        )
    sizes = list(LU.shape)
    m = sizes[-2]
    n = sizes[-1]
    k = min(m, n)
    sizes[-1] = m
    if unpack_pivots:
        P = LU.new_empty(sizes)
    else:
        P = LU.new_empty([0])
    if unpack_data:
        sizes[-1] = k
        L = LU.new_empty(sizes)
        sizes[-2] = k
        sizes[-1] = n
        U = LU.new_empty(sizes)
    else:
        L = LU.new_empty([0])
        U = LU.new_empty([0])
    return P, L, U


# parse the "mode" param in linalg_qr: return a tuple of bools (compute_q, reduced)
def _parse_qr_mode(mode: str) -> tuple[bool, bool]:
    if mode == "reduced":
        compute_q = True
        reduced = True
    elif mode == "complete":
        compute_q = True
        reduced = False
    elif mode == "r":
        compute_q = False
        reduced = True  # this is actually irrelevant in this mode
    else:
        torch._check(
            False,
            lambda: (
                f"qr received unrecognized mode '{mode}' "
                f"but expected one of 'reduced' (default), 'r', or 'complete'"
            ),
        )
    return compute_q, reduced  # type: ignore[possibly-undefined]


@register_meta([aten.linalg_qr.default, aten.linalg_qr.out])
@out_wrapper("Q", "R")
def linalg_qr_meta(A: Tensor, mode: str = "reduced") -> tuple[Tensor, Tensor]:
    checkIsMatrix(A, "linalg.qr")
    checkFloatingOrComplex(A, "linalg.qr")

    compute_q, reduced_mode = _parse_qr_mode(mode)

    m = A.shape[-2]
    n = A.shape[-1]
    k = min(m, n)

    if compute_q:
        Q_shape = list(A.shape)
        Q_shape[-1] = k if reduced_mode else m
        Q = A.new_empty(Q_shape)
        Q.as_strided_(Q_shape, make_contiguous_strides_for(Q_shape, row_major=False))
    else:
        Q = A.new_empty([0])

    # For readability
    R_shape = list(A.shape)
    R_shape[-2] = k if reduced_mode or not compute_q else m
    R = A.new_empty(R_shape)
    R.as_strided_(R_shape, make_contiguous_strides_for(R_shape, row_major=False))
    return Q, R


@register_meta([aten._linalg_slogdet.default, aten._linalg_slogdet.sign])
@out_wrapper("sign", "logabsdet", "LU", "pivots")
def _linalg_slogdet(A: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    squareCheckInputs(A, "linalg.slogdet")
    checkFloatingOrComplex(A, "linalg.slogdet", False)
    shape = A.shape
    sign = A.new_empty(shape[:-2])
    logabsdet = A.new_empty(shape[:-2], dtype=toRealValueType(A.dtype))
    LU = torch.empty_strided(
        size=shape,
        stride=make_contiguous_strides_for(shape, False),
        dtype=A.dtype,
        device=A.device,
    )
    pivots = A.new_empty(shape[:-1], dtype=torch.int32)
    return sign, logabsdet, LU, pivots


# From aten/src/ATen/native/BatchLinearAlgebra.cpp
# NOTE: matching defaults in aten/src/ATen/native/native_functions.yaml
@register_meta(aten._linalg_svd.default)
def _linalg_svd_meta(
    A: Tensor,
    full_matrices: bool = False,
    compute_uv: bool = True,
    driver: Optional[str] = None,
):
    checkIsMatrix(A, "linalg.svd")
    checkFloatingOrComplex(A, "linalg.svd")

    batch_dims = list(A.shape[:-2])
    m = A.shape[-2]
    n = A.shape[-1]
    k = min(m, n)

    if compute_uv:
        U_shape = batch_dims + [m, m if full_matrices else k]
        U = A.new_empty(U_shape)
        U.as_strided_(U_shape, make_contiguous_strides_for(U_shape, row_major=False))

        V_shape = batch_dims + [n if full_matrices else k, n]
        V = A.new_empty(V_shape)
        # NB: This checks for CUDA since there is no way to check for cuSolver.
        # Also, this might not work correctly on CPU when fake_device is not
        # available as device_hint just defaults to CUDA in that case. See
        # _linalg_svd meta in core.
        is_cuda = device_hint(A) == "cuda"
        V.as_strided_(V_shape, make_contiguous_strides_for(V_shape, row_major=is_cuda))
    else:
        # doesn't matter
        U = A.new_empty([0])
        V = A.new_empty([0])

    # S is always real, even when A is complex.
    S = A.new_empty(batch_dims + [k], dtype=toRealValueType(A.dtype))
    return U, S, V


def _linalg_broadcast_batch_dims(
    arg1: Tensor,
    arg2: Tensor,
) -> tuple[list[int], list[int]]:
    # broadcast the batch dimensions of arg1 and arg2.
    arg1_batch_sizes = arg1.shape[:-2]
    arg2_batch_sizes = arg2.shape[:-2]
    expand_batch_portion = _broadcast_shapes(arg1_batch_sizes, arg2_batch_sizes)

    arg1_expand_size = list(expand_batch_portion)
    arg1_expand_size += [arg1.size(-2), arg1.size(-1)]

    arg2_expand_size = list(expand_batch_portion)
    arg2_expand_size += [arg2.size(-2), arg2.size(-1)]
    return arg1_expand_size, arg2_expand_size


def _linalg_broadcast_batch_dims_name(
    arg1: Tensor,
    arg2: Tensor,
    name: Optional[str],
) -> tuple[Tensor, Tensor]:
    # If there's no name we assume we don't want to check the errors
    if name:
        linearSolveCheckInputs(arg1, arg2, name)

    arg1_expand_size, arg2_expand_size = _linalg_broadcast_batch_dims(arg1, arg2)

    arg1_broadcasted = (
        arg1 if arg1_expand_size == arg1.shape else arg1.expand(arg1_expand_size)
    )
    arg2_broadcasted = (
        arg2 if arg2_expand_size == arg2.shape else arg2.expand(arg2_expand_size)
    )
    return arg1_broadcasted, arg2_broadcasted


def linalg_solve_is_vector_rhs(input: Tensor, other: Tensor) -> bool:
    expected_batched_rhs_shape = input.shape[:-1]
    vector_case = other.ndim == 1 or (
        input.ndim - 1 == other.ndim and other.shape == expected_batched_rhs_shape
    )
    return vector_case


@register_meta(aten._linalg_solve_ex)
def _linalg_solve_ex(
    A: Tensor,
    B: Tensor,
    *,
    left: bool = True,
    check_errors: bool = False,
    result: Optional[Tensor] = None,
    LU: Optional[Tensor] = None,
    pivots: Optional[Tensor] = None,
    info: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    checkFloatingOrComplex(A, "linalg.solve")
    torch._check(
        A.dtype == B.dtype,
        lambda: (
            f"linalg.solve: Expected A and B to have the same dtype, but found A of type "
            f"{A.dtype} and B of type {B.dtype} instead"
        ),
    )
    vector_case = linalg_solve_is_vector_rhs(A, B)
    B_ = B.unsqueeze(-1) if vector_case else B
    checkInputsSolver(A, B_, left, "linalg.solve")
    B_broad_shape, _ = _linalg_broadcast_batch_dims(B_, A)
    torch._check(
        left or not vector_case,
        lambda: (
            "linalg.solve: Vector broadcasting of the left hand side is not supported for left=False. "
            "In this case linalg.solve is equivalent to B / A.squeeze(-1)"
        ),
    )
    result_shape = B_broad_shape[:-1] if vector_case else B_broad_shape
    result_ = torch.empty_strided(
        size=result_shape,
        stride=make_contiguous_strides_for(result_shape, not left),
        dtype=B.dtype,
        device=B.device,
    )
    shape = A.shape
    LU_ = torch.empty_strided(
        size=shape,
        stride=make_contiguous_strides_for(shape, False),
        dtype=A.dtype,
        device=A.device,
    )
    pivots_ = A.new_empty(shape[:-1], dtype=torch.int32)
    info_ = A.new_empty(shape[:-2], dtype=torch.int32)
    out = (result, LU, pivots, info)
    res = (result_, LU_, pivots_, info_)
    if all(x is not None for x in out):
        for r, o in zip(res, out):
            # resize and copy operations are done in-place
            _maybe_resize_out(o, r.shape)  # type: ignore[arg-type]
            # strides are not copied in out_wrapper
            o.as_strided_(r.shape, r.stride())  # type: ignore[union-attr]
            _safe_copy_out(copy_from=r, copy_to=o, exact_dtype=False)  # type: ignore[arg-type]
    return res


@register_meta([aten.linalg_solve_triangular.default, aten.linalg_solve_triangular.out])
def linalg_solve_triangular_meta(
    A: Tensor,
    B: Tensor,
    *,
    upper: bool,
    left: bool = True,
    unitriangular: bool = False,
    out: Optional[Tensor] = None,
) -> Tensor:
    if out is None:
        out = A.new_empty([0])
    assert isinstance(out, TensorLike)
    checkInputsSolver(A, B, left, "linalg.solve_triangular")
    B_, A_ = _linalg_broadcast_batch_dims_name(B, A, None)
    avoid_copy_A = A_.transpose(-2, -1).is_contiguous() and A_.is_conj()
    if avoid_copy_A:
        out = _maybe_resize_out(out, B_.shape)
    else:
        # reimplementation of resize_output with result F-contig
        if _resize_output_check(out, B_.shape):
            out.resize_(B_.transpose(-2, -1).shape)
            out.transpose_(-2, -1)
    return out  # type: ignore[return-value]


@register_meta(aten.triangular_solve)
@out_wrapper("X", "M", exact_dtype=True)
def triangular_solve_meta(
    self: Tensor,
    A: Tensor,
    upper: bool = True,
    transpose: bool = False,
    unitriangular: bool = False,
) -> tuple[Tensor, Tensor]:
    torch._check(
        self.ndim >= 2,
        lambda: (
            f"torch.triangular_solve: Expected b to have at least 2 dimensions, "
            f"but it has {self.ndim} dimensions instead"
        ),
    )
    torch._check(
        A.ndim >= 2,
        lambda: (
            f"torch.triangular_solve: Expected A to have at least 2 dimensions, "
            f"but it has {A.ndim} dimensions instead"
        ),
    )

    linearSolveCheckInputs(self, A, "triangular_solve")

    if A.layout == torch.strided:
        self_broadcast_size, A_broadcast_size = _linalg_broadcast_batch_dims(self, A)
        solution = torch.empty_strided(
            size=self_broadcast_size,
            stride=make_contiguous_strides_for(self_broadcast_size, row_major=False),
            dtype=self.dtype,
            device=self.device,
        )
        cloned_coefficient = torch.empty_strided(
            size=A_broadcast_size,
            stride=make_contiguous_strides_for(A_broadcast_size, row_major=False),
            dtype=A.dtype,
            device=A.device,
        )
    elif A.layout == torch.sparse_csr or A.layout == torch.sparse_bsr:
        solution = torch.empty_like(self)
        cloned_coefficient = self.new_empty([0])
    else:
        torch._check(False, lambda: "triangular_solve: Got an unexpected layout.")
    return solution, cloned_coefficient  # type: ignore[possibly-undefined]


# From aten/src/ATen/native/LinearAlgebra.cpp
@register_meta(aten._linalg_det.default)
def _linalg_det_meta(A):
    squareCheckInputs(A, "linalg.det")
    checkFloatingOrComplex(A, "linalg.det")

    det = A.new_empty(A.shape[:-2])

    LU = A.new_empty(A.shape)
    LU.as_strided_(A.shape, make_contiguous_strides_for(A.shape, row_major=False))

    pivots = A.new_empty(A.shape[:-1], dtype=torch.int32)
    return det, LU, pivots


@register_meta(aten.ormqr)
@out_wrapper()
def ormqr(
    input: Tensor,
    tau: Tensor,
    other: Tensor,
    left: bool = True,
    transpose: bool = False,
) -> Tensor:
    torch._check(
        input.ndim >= 2, lambda: "torch.ormqr: input must have at least 2 dimensions."
    )
    torch._check(
        other.ndim >= 2, lambda: "torch.ormqr: other must have at least 2 dimensions."
    )

    left_size_condition = -2 if left else -1
    torch._check(
        other.shape[left_size_condition] >= tau.shape[-1],
        lambda: f"torch.ormqr: other.shape[{left_size_condition}] must be greater than or equal to tau.shape[-1]",
    )
    torch._check(
        other.shape[left_size_condition] == input.shape[-2],
        lambda: f"torch.ormqr: other.shape[{left_size_condition}] must be equal to input.shape[-2]",
    )

    torch._check(
        tau.shape[-1] <= input.shape[-1],
        lambda: "torch.ormqr: tau.shape[-1] must be less than or equal to input.shape[-1]",
    )

    torch._check(
        input.ndim - tau.ndim == 1,
        lambda: (
            f"torch.ormqr: Expected tau to have one dimension less than input, "
            f"but got tau.ndim equal to {tau.ndim} and input.ndim is equal to {input.ndim}"
        ),
    )
    torch._check(
        input.ndim == other.ndim,
        lambda: (
            f"torch.ormqr: Expected other to have the same number of dimensions as input, "
            f"but got other.ndim equal to {other.ndim} and input.ndim is equal to {input.ndim}"
        ),
    )

    if input.ndim > 2:
        expected_batch_shape = input.shape[:-2]
        actual_batch_tau_shape = tau.shape[:-1]
        torch._check(
            actual_batch_tau_shape == expected_batch_shape,
            lambda: (
                f"torch.ormqr: Expected batch dimensions of tau to be "
                f"equal to input.shape[:-2], but got {actual_batch_tau_shape}"
            ),
        )

        actual_batch_other_shape = other.shape[:-2]
        torch._check(
            actual_batch_other_shape == expected_batch_shape,
            lambda: (
                f"torch.ormqr: Expected batch dimensions of other to be "
                f"equal to input.shape[:-2], but got {actual_batch_other_shape}"
            ),
        )

    torch._check(
        tau.dtype == input.dtype,
        lambda: (
            f"torch.ormqr: Expected input and tau to have the same dtype, "
            f"but input has dtype {input.dtype} and tau has dtype {tau.dtype}"
        ),
    )
    torch._check(
        other.dtype == input.dtype,
        lambda: (
            f"torch.ormqr: Expected input and other to have the same dtype, "
            f"but input has dtype {input.dtype} and other has dtype {other.dtype}"
        ),
    )

    checkSameDevice("torch.ormqr", tau, input, "tau")
    checkSameDevice("torch.ormqr", other, input, "other")

    return torch.empty_strided(
        size=other.shape,
        stride=make_contiguous_strides_for(other.shape, row_major=False),
        dtype=other.dtype,
        device=other.device,
    )


def _padding_check_valid_input(input, padding, *, dim):
    torch._check(
        len(padding) == 2 * dim,
        lambda: f"padding size is expected to be {2 * dim}, but got: {len(padding)}",
    )

    input_dim = input.ndim

    is_batch_mode = input_dim == (dim + 2)

    valid_batch_mode = is_batch_mode
    valid_non_batch_mode = not is_batch_mode

    if is_batch_mode:
        # allow batch size of 0-dim.
        for d in range(1, input_dim):
            valid_batch_mode = valid_batch_mode and input.size(d) != 0
    else:
        for d in range(0, input_dim):
            valid_non_batch_mode = valid_non_batch_mode and input.size(d) != 0

    # allow empty batch size but not other dimensions.
    torch._check(
        valid_batch_mode or valid_non_batch_mode,
        lambda: (
            f"Expected {dim + 1}D or {dim + 2}D (batch mode) tensor with possibly 0 batch size "
            f"and other non-zero dimensions for input, but got: {input.shape}"
        ),
    )


def _pad1d_common(input, padding, *, is_reflection):
    dim_plane = 0
    dim_w = 1
    nbatch = 1

    if input.ndim == 3:
        nbatch = input.size(0)
        dim_w += 1
        dim_plane += 1

    _padding_check_valid_input(input, padding, dim=1)

    pad_l, pad_r = padding

    nplane = input.size(dim_plane)
    input_w = input.size(dim_w)
    output_w = input_w + pad_l + pad_r

    if is_reflection:
        torch._check(
            pad_l < input_w and pad_r < input_w,
            lambda: (
                f"Argument #4: Padding size should be less than the corresponding input dimension, "
                f"but got: padding ({pad_l}, {pad_r}) at dimension {dim_w} of input {input.shape}"
            ),
        )

    torch._check(
        output_w >= 1,
        lambda: f"input (W: {input_w}) is too small. Calculated output W: {output_w}",
    )

    if input.ndim == 2:
        return input.new_empty((nplane, output_w))
    else:
        return input.new_empty((nbatch, nplane, output_w))


@register_meta(aten.reflection_pad1d)
@out_wrapper()
def meta_reflection_pad1d(input, padding):
    return _pad1d_common(input, padding, is_reflection=True)


@register_meta(aten.replication_pad1d)
@out_wrapper()
def meta_replication_pad1d(input, padding):
    torch._check(
        input.dtype != torch.bool,
        lambda: f""""replication_pad1d" not implemented for '{input.dtype.__str__()}'""",
    )
    return _pad1d_common(input, padding, is_reflection=False)


def _pad1d_backward_common(grad_output, input, padding, *, is_reflection):
    dim_w = 1
    if not is_reflection:
        torch._check(len(padding) == 2, lambda: "padding size is expected to be 2")

    if input.ndim == 3:
        dim_w += 1

    pad_l, pad_r = padding

    input_w = input.size(dim_w)
    output_w = input_w + pad_l + pad_r

    if is_reflection:
        torch._check(
            pad_l < input_w and pad_r < input_w,
            lambda: (
                f"Argument #4: Padding size should be less than the corresponding input dimension, "
                f"but got: padding ({pad_l}, {pad_r}) at dimension {dim_w} of input {input.shape}"
            ),
        )

    torch._check(
        output_w == grad_output.size(dim_w),
        lambda: f"grad_output width unexpected. Expected: {output_w}, Got: {grad_output.size(dim_w)}",
    )

    return input.new_empty(input.shape)


@register_meta(aten.reflection_pad1d_backward)
@out_wrapper("grad_input")
def meta_reflection_pad1d_backward(grad_output, input, padding):
    return _pad1d_backward_common(grad_output, input, padding, is_reflection=True)


@register_meta(aten.replication_pad1d_backward)
@out_wrapper("grad_input")
def meta_replication_pad1d_backward(grad_output, input, padding):
    return _pad1d_backward_common(grad_output, input, padding, is_reflection=False)


def _pad2d_common(input, padding, *, is_reflection):
    dim_w = 2
    dim_h = 1
    dim_slices = 0
    nbatch = 1

    _padding_check_valid_input(input, padding, dim=2)

    ndim = input.ndim
    if ndim == 4:
        nbatch = input.size(0)
        dim_w += 1
        dim_h += 1
        dim_slices += 1

    pad_l, pad_r, pad_t, pad_b = padding

    nplane = input.size(dim_slices)
    input_h = input.size(dim_h)
    input_w = input.size(dim_w)
    output_h = input_h + pad_t + pad_b
    output_w = input_w + pad_l + pad_r

    if is_reflection:
        torch._check(
            pad_l < input_w and pad_r < input_w,
            lambda: (
                f"Argument #4: Padding size should be less than the corresponding input dimension, "
                f"but got: padding ({pad_l}, {pad_r}) at dimension {dim_w} of input {input.shape}"
            ),
        )
        torch._check(
            pad_t < input_h and pad_b < input_h,
            lambda: (
                f"Argument #6: Padding size should be less than the corresponding input dimension, "
                f"but got: padding ({pad_t}, {pad_b}) at dimension {dim_h} of input {input.shape}"
            ),
        )

    torch._check(
        output_w >= 1 or output_h >= 1,
        lambda: (
            f"input (H: {input_h} W: {input_w}) is too small. "
            f"Calculated output H: {output_h} W: {output_w}"
        ),
    )

    if input.ndim == 3:
        return input.new_empty((nplane, output_h, output_w))
    else:
        return input.new_empty((nbatch, nplane, output_h, output_w))


@register_meta(aten.reflection_pad2d)
@out_wrapper()
def meta_reflection_pad2d(input, padding):
    return _pad2d_common(input, padding, is_reflection=True)


@register_meta(aten.replication_pad2d)
@out_wrapper()
def meta_replication_pad2d(input, padding):
    torch._check(
        input.dtype != torch.bool,
        lambda: f""""replication_pad2d" not implemented for '{input.dtype.__str__()}'""",
    )
    return _pad2d_common(input, padding, is_reflection=False)


@register_meta(
    [
        aten.reflection_pad2d_backward.default,
        aten.reflection_pad2d_backward.grad_input,
        aten.replication_pad2d_backward.default,
        aten.replication_pad2d_backward.grad_input,
    ]
)
@out_wrapper("grad_input")
def meta_pad2d_backward(grad_output, self, padding):
    dim_w = 2
    dim_h = 1
    dim_plane = 0

    self_shape = self.shape
    if self.dim() == 4:
        dim_w += 1
        dim_h += 1
        dim_plane += 1

    pad_l, pad_r, pad_t, pad_b = padding

    input_h = self_shape[dim_h]
    input_w = self_shape[dim_w]
    output_h = input_h + pad_t + pad_b
    output_w = input_w + pad_l + pad_r

    torch._check(
        output_w == grad_output.size(dim_w),
        lambda: f"grad_output width unexpected. Expected: {output_w}, Got: {grad_output.size(dim_w)}",
    )
    torch._check(
        output_h == grad_output.size(dim_h),
        lambda: f"grad_output height unexpected. Expected: {output_h}, Got: {grad_output.size(dim_h)}",
    )
    return self.new_empty(self.shape)


def _pad3d_common(input, padding, *, is_reflection):
    dim_w = 3
    dim_h = 2
    dim_d = 1
    dim_plane = 0

    _padding_check_valid_input(input, padding, dim=3)

    batch_mode = input.ndim == 5
    if batch_mode:
        nbatch = input.size(0)
        dim_w += 1
        dim_h += 1
        dim_d += 1
        dim_plane += 1

    pad_l, pad_r, pad_t, pad_b, pad_f, pad_bk = padding

    nplane = input.size(dim_plane)
    input_d = input.size(dim_d)
    input_h = input.size(dim_h)
    input_w = input.size(dim_w)
    output_d = input_d + pad_f + pad_bk
    output_h = input_h + pad_t + pad_b
    output_w = input_w + pad_l + pad_r

    if is_reflection:
        torch._check(
            pad_l < input_w and pad_r < input_w,
            lambda: (
                f"Argument #4: Padding size should be less than the corresponding input dimension, "
                f"but got: padding ({pad_l}, {pad_r}) at dimension {dim_w} of input {input.shape}"
            ),
        )
        torch._check(
            pad_t < input_h and pad_b < input_h,
            lambda: (
                f"Argument #6: Padding size should be less than the corresponding input dimension, "
                f"but got: padding ({pad_t}, {pad_b}) at dimension {dim_h} of input {input.shape}"
            ),
        )
        torch._check(
            pad_f < input_d and pad_bk < input_d,
            lambda: (
                f"Argument #8: Padding size should be less than the corresponding input dimension, "
                f"but got: padding ({pad_f}, {pad_bk}) at dimension {dim_d} of input {input.shape}"
            ),
        )

    torch._check(
        output_w >= 1 or output_h >= 1 or output_d >= 1,
        lambda: (
            f"input (D: {input_d} H: {input_h} W: {input_w}) is too small. "
            f"Calculated output D: {output_d} H: {output_h} W: {output_w}"
        ),
    )

    if batch_mode:
        return input.new_empty((nbatch, nplane, output_d, output_h, output_w))  # type: ignore[possibly-undefined]
    else:
        return input.new_empty((nplane, output_d, output_h, output_w))


@register_meta(aten.reflection_pad3d)
@out_wrapper()
def meta_reflection_pad3d(input, padding):
    return _pad3d_common(input, padding, is_reflection=True)


@register_meta(aten.replication_pad3d)
@out_wrapper()
def meta_replication_pad3d(input, padding):
    torch._check(
        input.dtype != torch.bool,
        lambda: f""""replication_pad3d" not implemented for '{input.dtype.__str__()}'""",
    )
    return _pad3d_common(input, padding, is_reflection=False)


@register_meta(
    [
        aten.reflection_pad3d_backward.default,
        aten.reflection_pad3d_backward.grad_input,
        aten.replication_pad3d_backward.default,
        aten.replication_pad3d_backward.grad_input,
    ]
)
@out_wrapper("grad_input")
def meta_pad3d_backward(grad_output, input, padding):
    torch._check(len(padding) == 6, lambda: "padding size is expected to be 6")
    assert input.ndim > 3
    assert grad_output.ndim == input.ndim

    dim_w = 3
    dim_h = 2
    dim_d = 1

    if input.ndim == 5:
        dim_w += 1
        dim_h += 1
        dim_d += 1

    pad_l, pad_r, pad_t, pad_b, pad_f, pad_bk = padding

    input_d = input.size(dim_d)
    input_h = input.size(dim_h)
    input_w = input.size(dim_w)
    output_d = input_d + pad_f + pad_bk
    output_h = input_h + pad_t + pad_b
    output_w = input_w + pad_l + pad_r

    torch._check(
        output_w == grad_output.size(dim_w),
        lambda: f"grad_output width unexpected. Expected: {output_w}, Got: {grad_output.size(dim_w)}",
    )
    torch._check(
        output_h == grad_output.size(dim_h),
        lambda: f"grad_output height unexpected. Expected: {output_h}, Got: {grad_output.size(dim_h)}",
    )
    torch._check(
        output_d == grad_output.size(dim_d),
        lambda: f"grad_output depth unexpected. Expected: {output_d}, Got: {grad_output.size(dim_d)}",
    )

    return input.new_empty(input.shape)


@register_meta(aten._pdist_forward)
@out_wrapper()
def meta__pdist_forward(self: Tensor, p: float = 2) -> Tensor:
    torch._check(
        self.is_contiguous(), lambda: "_pdist_forward requires contiguous input"
    )
    n = self.size(0)
    if n <= 1:
        return self.new_empty([0]).to(memory_format=torch.legacy_contiguous_format)  # type: ignore[call-overload]
    else:
        return self.new_empty((n * (n - 1) // 2,)).to(
            memory_format=torch.legacy_contiguous_format
        )  # type: ignore[call-overload]


@register_meta(aten._pdist_backward)
@out_wrapper()
def meta__pdist_backward(grad: Tensor, self: Tensor, p: float, pdist: Tensor) -> Tensor:
    torch._check(
        self.is_contiguous(), lambda: "_pdist_backward requires self to be contiguous"
    )
    torch._check(
        pdist.is_contiguous(), lambda: "_pdist_backward requires pdist to be contiguous"
    )
    return torch.empty_like(self, memory_format=torch.legacy_contiguous_format)


@register_meta([aten.baddbmm.default, aten.baddbmm.out])
@out_wrapper(exact_dtype=True)
def meta_baddbmm(self, batch1, batch2, *, beta=1, alpha=1):
    from torch.fx.experimental.symbolic_shapes import guard_or_true, sym_eq

    dim1 = batch1.size(0)
    dim2 = batch1.size(1)
    dim3 = batch2.size(2)
    if guard_or_true(torch.sym_not(sym_eq(self.shape, (dim1, dim2, dim3)))):
        self = self.expand((dim1, dim2, dim3))
    torch._check(batch1.dim() == 3, lambda: "batch1 must be a 3D tensor")
    torch._check(batch2.dim() == 3, lambda: "batch2 must be a 3D tensor")
    if not exp_config.skip_dtype_check_in_meta_registrations:
        torch._check(
            self.dtype == batch1.dtype == batch2.dtype,
            lambda: f"Input dtypes must be the same, got: input: {self.dtype}, batch1: {batch1.dtype}, batch2: {batch2.dtype}",
        )
    batch1_sizes = batch1.shape
    batch2_sizes = batch2.shape
    bs = batch1_sizes[0]
    contraction_size = batch1_sizes[2]
    torch._check(
        batch2_sizes[0] == bs and batch2_sizes[1] == contraction_size,
        lambda: (
            f"Expected size for first two dimensions of batch2 tensor to be: "
            f"[{bs}, {contraction_size}] but got: [{batch2_sizes[0]}, {batch2_sizes[1]}]."
        ),
    )
    return self.new_empty(self.size())


@register_meta([aten.bernoulli.default, aten.bernoulli.out])
@out_wrapper()
def meta_bernoulli(self, *, generator=None):
    # https://github.com/pytorch/pytorch/issues/88612
    return torch.empty_like(self, memory_format=torch.contiguous_format)


@register_meta(aten.bernoulli_.float)
def meta_bernoulli_(self, p=0.5, generator=None):
    return self


@register_meta(aten.bernoulli.p)
def meta_bernoulli_p(self, p=0.5, generator=None):
    # https://github.com/pytorch/pytorch/issues/88612
    return torch.empty_like(self, memory_format=torch.contiguous_format)


@register_meta([aten.poisson.default, aten.poisson.out])
@out_wrapper()
def meta_poisson(self, generator=None):
    return torch.empty_like(self)


@register_meta(aten._fused_moving_avg_obs_fq_helper.default)
def meta__fused_moving_avg_obs_fq_helper(
    self,
    observer_on,
    fake_quant_on,
    running_min,
    running_max,
    scale,
    zero_point,
    averaging_const,
    quant_min,
    quant_max,
    ch_axis,
    per_row_fake_quant=False,
    symmetric_quant=False,
):
    torch._check(
        ch_axis < self.dim(),
        lambda: "Error in fused_moving_avg_obs_fake_quant_cpu: ch_axis must be < self.dim()",
    )
    mask = torch.empty_like(self, dtype=torch.bool)
    return (torch.empty_like(self), mask)


@register_meta(aten.mm)
@out_wrapper(exact_dtype=True)
def meta_mm(a, b):
    torch._check(a.dim() == 2, lambda: "a must be 2D")
    torch._check(b.dim() == 2, lambda: "b must be 2D")
    N, M1 = a.shape
    M2, P = b.shape
    torch._check(
        M1 == M2,
        lambda: f"a and b must have same reduction dim, but got [{N}, {M1}] X [{M2}, {P}].",
    )
    return a.new_empty(N, P)


def _compute_reduction_shape(self, dims, keepdim):
    if keepdim:
        return tuple(self.shape[i] if i not in dims else 1 for i in range(self.ndim))

    return utils.compute_reduction_output_shape(self.shape, dims)


# FakeTensors (meta tensors with a device) will report device as meta
# when running meta kernels. Here, access the "fake device" of FakeTensor if it
# exists so meta kernels which have diverge per device will be more
# accurate when run with FakeTensors
def device_hint(tensor) -> "str":
    if isinstance(tensor, torch._subclasses.FakeTensor):
        return tensor.fake_device.type
    elif (
        hasattr(tensor, "device")
        and hasattr(tensor.device, "type")
        and tensor.device.type != "meta"
    ):
        return tensor.device.type
    else:
        return "cuda"  # default to cuda


def calc_conv_nd_return_shape(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    stride: Union[list[int], int],
    padding: Union[list[int], int],
    dilation: Union[list[int], int],
    is_transposed: bool,
    groups: int,
    output_padding: Optional[Union[list[int], int]] = None,
):
    def _formula(ln: int, p: int, d: int, k: int, s: int) -> int:
        """
        Formula to apply to calculate the length of some dimension of the output

        See: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

        Args:
            ln: length of the dimension
            p: padding in that dim
            d: dilation in that dim
            k: kernel size in that dim
            s: stride in that dim
        Returns:
            The output length
        """
        return (ln + 2 * p - d * (k - 1) - 1) // s + 1

    def _formula_transposed(ln: int, p: int, d: int, k: int, s: int, op: int) -> int:
        """
        Formula to apply to calculate the length of some dimension of the output
        if transposed convolution is used.
        See: https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html

        Args:
            ln: length of the dimension
            p: padding in that dim
            d: dilation in that dim
            k: kernel size in that dim
            s: stride in that dim
            op: output padding in that dim

        Returns:
            The output length
        """
        return (ln - 1) * s - 2 * p + d * (k - 1) + op + 1

    kernel_size = weight.shape[2:]
    dims = input_tensor.shape[2:]
    if is_transposed:
        out_channels = groups * weight.shape[1]
    else:
        out_channels = weight.shape[0]
        if weight.shape[1] * groups != input_tensor.shape[1]:
            raise RuntimeError("Invalid channel dimensions")

    ret_shape = [input_tensor.shape[0], out_channels]
    if isinstance(stride, IntLike):
        stride = [stride] * len(dims)
    elif len(stride) == 1:
        stride = [stride[0]] * len(dims)

    if isinstance(padding, IntLike):
        padding = [padding] * len(dims)
    elif len(padding) == 1:
        padding = [padding[0]] * len(dims)

    if isinstance(dilation, IntLike):
        dilation = [dilation] * len(dims)
    elif len(dilation) == 1:
        dilation = [dilation[0]] * len(dims)

    output_padding_list: Optional[list[int]] = None
    if output_padding:
        if isinstance(output_padding, IntLike):
            output_padding_list = [output_padding] * len(dims)
        elif len(output_padding) == 1:
            output_padding_list = [output_padding[0]] * len(dims)
        else:
            output_padding_list = output_padding

    for i in range(len(dims)):
        # If output_padding is present, we are dealing with a transposed convolution
        if output_padding_list:
            ret_shape.append(
                _formula_transposed(
                    dims[i],
                    padding[i],
                    dilation[i],
                    kernel_size[i],
                    stride[i],
                    output_padding_list[i],
                )
            )
        else:
            ret_shape.append(
                _formula(dims[i], padding[i], dilation[i], kernel_size[i], stride[i])
            )

    torch._check(
        any(x > 0 for x in ret_shape[2:]),
        lambda: f"Given input size per channel: {list(dims)}. "
        f"Calculated output size per channel: {ret_shape[2:]}. "
        f"Output size is too small",
    )

    return ret_shape


def is_channels_last(ten):
    return torch._prims_common.suggest_memory_format(ten) == torch.channels_last


@register_meta(aten.miopen_batch_norm.default)
def meta_miopen_batch_norm(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    running_mean: Optional[torch.Tensor],
    running_var: Optional[torch.Tensor],
    training: bool,
    exponential_average_factor: float,
    epsilon: float,
):
    # In batch norm the output is of the same shape as the input
    out_shape = input_tensor.shape

    # If tensor is provided for running_mean and running_var then use this. If these are not
    # provided then we return the shape of weight tensor. Similar to how this is handled in the decomposition
    save_mean_shape = running_mean.shape if running_mean is not None else weight.shape
    save_var_shape = running_var.shape if running_var is not None else weight.shape

    def pick_memory_format():
        if is_channels_last(input_tensor):
            return torch.channels_last
        if input_tensor.is_contiguous(memory_format=torch.contiguous_format):
            return torch.contiguous_format
        return torch.contiguous_format

    out = input_tensor.new_empty(out_shape).to(memory_format=pick_memory_format())

    if training:
        save_mean = input_tensor.new_empty(save_mean_shape)
        save_var = input_tensor.new_empty(save_var_shape)
    else:
        save_mean = input_tensor.new_empty((0,))
        save_var = input_tensor.new_empty((0,))

    return out, save_mean, save_var


@register_meta(aten.convolution.default)
def meta_conv(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: list[int],
    padding: list[int],
    dilation: list[int],
    is_transposed: bool,
    output_padding: list[int],
    groups: int,
):
    def pick_memory_format():
        if device_hint(input_tensor) == "cuda":
            if is_channels_last(input_tensor) or is_channels_last(weight):
                return torch.channels_last
        else:
            if is_channels_last(input_tensor):
                return torch.channels_last
        if input_tensor.is_contiguous(memory_format=torch.contiguous_format):
            return torch.contiguous_format
        elif input_tensor.is_contiguous(memory_format=torch.preserve_format):
            return torch.preserve_format

    shape_out = calc_conv_nd_return_shape(
        input_tensor,
        weight,
        stride,
        padding,
        dilation,
        is_transposed,
        groups,
        output_padding if is_transposed else None,
    )

    input_channels_dim = 1
    output_channels_dim = 1
    if input_tensor.size(input_channels_dim) == 0:
        shape_out[output_channels_dim] = 0

    out = input_tensor.new_empty(shape_out)
    out = out.to(memory_format=pick_memory_format())  # type: ignore[call-overload]
    return out


if torch._C._has_mkldnn:
    _meta_lib_dont_use_me_use_register_meta_for_mkldnn = torch.library.Library(
        "mkldnn", "IMPL", "Meta"
    )

    @register_meta(torch.ops.mkldnn._convolution_pointwise.default)
    def meta_mkldnn_convolution_default(
        input_tensor,
        weight,
        bias,
        padding,
        stride,
        dilation,
        groups,
        attr,
        scalars,
        algorithm,
    ):
        shape_out = calc_conv_nd_return_shape(
            input_tensor, weight, stride, padding, dilation, False, groups, []
        )
        out = input_tensor.new_empty(shape_out)
        out_memory_format = torch.channels_last
        if input_tensor.dim() == 5:
            out_memory_format = torch.channels_last_3d
        out = out.to(memory_format=out_memory_format)  # type: ignore[call-overload]
        return out

    @register_meta(torch.ops.mkldnn._linear_pointwise.default)
    def meta_linear_pointwise_default(
        input_tensor, weight, bias, attr, scalars, algorithm
    ):
        return input_tensor.new_empty((*input_tensor.shape[:-1], weight.shape[0]))

    if torch._C.has_mkl:
        _meta_lib_dont_use_me_use_register_meta_for_mkl = torch.library.Library(
            "mkl", "IMPL", "Meta"
        )

        @register_meta(torch.ops.mkl._mkl_linear)
        def meta_mkl_linear(input_tensor, packed_weight, orig_weight, bias, batch_size):
            return input_tensor.new_empty(
                (*input_tensor.shape[:-1], orig_weight.shape[0])
            )

    _meta_lib_dont_use_me_use_register_meta_for_onednn = torch.library.Library(
        "onednn", "IMPL", "Meta"
    )

    @register_meta(torch.ops.onednn.qconv2d_pointwise.default)
    @register_meta(torch.ops.onednn.qconv_pointwise.default)
    def meta_qconv_pointwise(
        x,
        x_scale,
        x_zp,
        w,  # prepacked_weight
        w_scale,
        w_zp,
        bias,
        stride,
        padding,
        dilation,
        groups,
        output_scale,
        output_zero_point,
        output_dtype,
        attr,
        scalars,
        algorithm,
    ):
        shape_out = calc_conv_nd_return_shape(
            x,
            w,
            stride,
            padding,
            dilation,
            False,
            groups,
            None,
        )
        if output_dtype is None:
            output_dtype = x.dtype
        assert output_dtype in [
            torch.float32,
            torch.bfloat16,
            torch.uint8,
            torch.int8,
            torch.float8_e4m3fn,
        ]
        out = x.new_empty(shape_out, dtype=output_dtype)
        assert len(shape_out) in [3, 4, 5], (
            "Expect output to be 3d/4d/5d for conv1d/2d/3d"
        )
        format = {
            3: torch.contiguous_format,
            4: torch.channels_last,
            5: torch.channels_last_3d,
        }[len(shape_out)]
        out = out.to(memory_format=format)
        return out

    @register_meta(torch.ops.onednn.qconv2d_pointwise.binary)
    def meta_qconv2d_pointwise_binary(
        x,
        x_scale,
        x_zp,
        w,
        w_scale,
        w_zp,
        accum,
        bias,
        stride,
        padding,
        dilation,
        groups,
        output_scale,
        output_zero_point,
        output_dtype,
        accum_scale,
        accum_zero_point,
        binary_op_name,
        alpha,
        unary_op_name,
        unary_op_args,
        unary_op_algorithm,
    ):
        assert binary_op_name == "sum"
        return accum

    @register_meta(torch.ops.onednn.qlinear_pointwise.default)
    @register_meta(torch.ops.onednn.qlinear_pointwise.tensor)
    def meta_qlinear_pointwise(
        x,
        x_scale,
        x_zp,
        w,
        w_scale,
        w_zp,
        bias,
        output_scale,
        output_zero_point,
        output_dtype,
        post_op_name,
        post_op_args,
        post_op_algorithm,
    ):
        output_shape = list(x.shape)
        # The weight has been transposed during the qlinear weight prepack process.
        output_shape[-1] = w.shape[1]
        assert output_dtype in [
            torch.float32,
            torch.bfloat16,
            torch.int8,
            torch.uint8,
            torch.float8_e4m3fn,
        ]
        out = x.new_empty(output_shape, dtype=output_dtype)
        return out

    @register_meta(torch.ops.onednn.qlinear_pointwise.binary)
    @register_meta(torch.ops.onednn.qlinear_pointwise.binary_tensor)
    def meta_qlinear_pointwise_binary(
        x,
        x_scale,
        x_zp,
        w,
        w_scale,
        w_zp,
        x_2,
        bias,
        output_scale,
        output_zero_point,
        output_dtype,
        x2_scale,
        x2_zp,
        binary_op_name,
        alpha,
        unary_op_name,
        unary_op_args,
        unary_op_algorithm,
    ):
        if binary_op_name == "sum":
            return x_2
        output_shape = list(x.shape)
        # The weight has been transposed during the qlinear weight prepack process.
        output_shape[-1] = w.shape[1]
        assert output_dtype in [
            torch.float32,
            torch.bfloat16,
            torch.uint8,
            torch.int8,
            torch.float8_e4m3fn,
        ]
        out = x.new_empty(output_shape, dtype=output_dtype)
        return out

    @register_meta(torch.ops.onednn.linear_dynamic_fp16.default)
    @register_meta(torch.ops.onednn.linear_relu_dynamic_fp16.default)
    def meta_linear_dynamic_fp16(
        x,
        w,
        bias,
    ):
        output_shape = list(x.shape)
        # The weight has been transposed during the qlinear weight prepack process.
        output_shape[-1] = w.shape[1]
        out = x.new_empty(output_shape)
        return out

    _meta_lib_dont_use_me_use_register_meta_for_quantized = torch.library.Library(
        "quantized", "IMPL", "Meta"
    )

    @register_meta(torch.ops.quantized.max_pool2d)
    def meta_quantized_max_pool2d(
        input,
        kernel_size,
        stride=(),
        padding=(0,),
        dilation=(1,),
        ceil_mode=False,
    ):
        (
            nInputPlane,
            outputHeight,
            outputWidth,
        ) = max_pool2d_checks_and_compute_shape(
            input, kernel_size, stride, padding, dilation, ceil_mode
        )
        nbatch = input.size(-4) if input.dim() == 4 else 1
        memory_format = torch.channels_last
        if input.dim() == 3:
            size = [nInputPlane, outputHeight, outputWidth]
        else:
            size = [nbatch, nInputPlane, outputHeight, outputWidth]
        return torch.empty(
            size,
            dtype=input.dtype,
            device=input.device,
            memory_format=memory_format,
        )

    @register_meta(torch.ops.quantized.int4mm_packed_weight_cpu)
    def meta_int4mm_packed_weight_cpu(x, w, q_group_size, q_scale_and_zeros):
        torch._check(x.dim() == 2, f"x must be a 2D tensor, got {x.dim()}D")
        torch._check(w.dim() == 2, f"w must be a 2D tensor, got {w.dim()}D")
        torch._check(
            x.dtype in [torch.float32, torch.float16, torch.bfloat16],
            f"expected x to be f32/f16/bf16, got {x.dtype}",
        )
        torch._check(w.dtype == torch.uint8, f"expected w to be uint8, got {w.dtype}")
        torch._check(
            q_group_size.dtype == torch.int64,
            f"q_group_size must be int64, got {q_group_size.dtype}",
        )
        torch._check(
            q_scale_and_zeros.dtype == x.dtype,
            f"q_scale_and_zeros must have the same dtype as x, got {q_scale_and_zeros.dtype}",
        )
        return x.new_empty(x.size(0), w.size(0), dtype=x.dtype)


# from check_dim_size() in aten/src/ATen/TensorUtils.cpp.
def check_dim_size(tensor, dim, dim_size, size):
    torch._check(
        tensor.dim() == dim and tensor.shape[dim_size] == size,
        lambda: f"Expected a tensor of dimension {dim} and tensor.size[{dim_size}] == {size}, "
        + f"but got : dimension {tensor.dim()} and tensor.size[{dim_size}] = {tensor.shape[dim_size]}",
    )


@register_meta(aten.avg_pool2d.default)
def meta_avg_pool2d(
    input,
    kernel_size,
    stride=(),
    padding=(0,),
    ceil_mode=False,
    count_include_pad=True,
    divisor_override=None,
):
    def unpack(name, val):
        torch._check(
            len(val) in [1, 2],
            lambda: f"avg_pool2d: {name} must either be a single int, or a tuple of two ints",
        )
        H = val[0]
        W = H if len(val) == 1 else val[1]
        return H, W

    kH, kW = unpack("kernel_size", kernel_size)
    torch._check(
        len(stride) in [0, 1, 2],
        lambda: "avg_pool2d: stride must either be omitted, a single int, or a tuple of two ints",
    )
    torch._check(
        input.dtype not in [torch.uint8, torch.uint16, torch.uint32, torch.uint64],
        lambda: f""""avg_pool2d" not implemented for '{input.dtype.__str__()}'""",
    )
    if len(stride) == 0:
        dH, dW = kH, kW
    elif len(stride) == 1:
        dH, dW = stride[0], stride[0]
    else:
        dH, dW = unpack("stride", stride)

    padH, padW = unpack("padding", padding)

    torch._check(
        divisor_override is None or divisor_override != 0,
        lambda: "divisor must be not zero",
    )

    nbatch = input.size(-4) if input.dim() == 4 else 1
    nInputPlane = input.size(-3)
    inputHeight = input.size(-2)
    inputWidth = input.size(-1)

    outputHeight = pooling_output_shape(inputHeight, kH, padH, dH, 1, ceil_mode)
    outputWidth = pooling_output_shape(inputWidth, kW, padW, dW, 1, ceil_mode)

    memory_format = utils.suggest_memory_format(input)
    pool2d_shape_check(
        input,
        kH,
        kW,
        dH,
        dW,
        padH,
        padW,
        1,
        1,
        nInputPlane,
        inputHeight,
        inputWidth,
        outputHeight,
        outputWidth,
        memory_format,
    )

    if input.dim() == 3:
        size = [nInputPlane, outputHeight, outputWidth]
    else:
        size = [nbatch, nInputPlane, outputHeight, outputWidth]
    return torch.empty(
        size,
        dtype=input.dtype,
        device=input.device,
        memory_format=memory_format,
    )


# from avg_pool2d_backward_shape_check() in aten/src/ATen/native/Pool.h.
def avg_pool2d_backward_shape_check(
    input,
    gradOutput,
    nbatch,
    kH,
    kW,
    dH,
    dW,
    padH,
    padW,
    nInputPlane,
    inputHeight,
    inputWidth,
    outputHeight,
    outputWidth,
    mem_format,
):
    pool2d_shape_check(
        input,
        kH,
        kW,
        dH,
        dW,
        padH,
        padW,
        1,
        1,
        nInputPlane,
        inputHeight,
        inputWidth,
        outputHeight,
        outputWidth,
        mem_format,
    )

    ndim = input.dim()
    nOutputPlane = nInputPlane

    check_dim_size(gradOutput, ndim, ndim - 3, nOutputPlane)
    check_dim_size(gradOutput, ndim, ndim - 2, outputHeight)
    check_dim_size(gradOutput, ndim, ndim - 1, outputWidth)


# Don't override the C++ registration.
@register_meta(aten.avg_pool2d_backward.default)
def meta_avg_pool2d_backward(
    gradOutput_,
    input,
    kernel_size,
    stride,
    padding,
    ceil_mode,
    count_include_pad,
    divisor_override,
):
    # From aten/src/ATen/native/AveragePool2d.cpp structured kernel meta func.
    torch._check(
        len(kernel_size) == 1 or len(kernel_size) == 2,
        lambda: "avg_pool2d: kernel_size must either be a single int, or a tuple of two ints",
    )
    kH = kernel_size[0]
    kW = kH if len(kernel_size) == 1 else kernel_size[1]
    torch._check(
        len(stride) == 0 or len(stride) == 1 or len(stride) == 2,
        lambda: "avg_pool2d: stride must either be omitted, a single int, or a tuple of two ints",
    )
    dH = kH if len(stride) == 0 else stride[0]
    dW = kW if len(stride) == 0 else dH if len(stride) == 1 else stride[1]
    torch._check(
        len(padding) == 1 or len(padding) == 2,
        lambda: "avg_pool2d: padding must either be a single int, or a tuple of two ints",
    )
    padH = padding[0]
    padW = padH if len(padding) == 1 else padding[1]

    torch._check(
        divisor_override is None or divisor_override != 0,
        lambda: "divisor must be not zero",
    )

    input_size = input.shape
    nbatch = input_size[-4] if input.dim() == 4 else 1
    nInputPlane = input_size[-3]
    inputHeight = input_size[-2]
    inputWidth = input_size[-1]

    outputHeight = pooling_output_shape(inputHeight, kH, padH, dH, 1, ceil_mode)
    outputWidth = pooling_output_shape(inputWidth, kW, padW, dW, 1, ceil_mode)

    mem_format = utils.suggest_memory_format(input)

    avg_pool2d_backward_shape_check(
        input,
        gradOutput_,
        nbatch,
        kH,
        kW,
        dH,
        dW,
        padH,
        padW,
        nInputPlane,
        inputHeight,
        inputWidth,
        outputHeight,
        outputWidth,
        mem_format,
    )

    return torch.empty(
        input_size,
        dtype=input.dtype,
        device=input.device,
        memory_format=mem_format,
    )


@register_meta(aten.avg_pool3d)
@out_wrapper()
def meta_avg_pool3d(
    input,
    kernel_size,
    stride=(),
    padding=(0,),
    ceil_mode=False,
    count_include_pad=True,
    divisor_override=None,
):
    torch._check(
        len(kernel_size) in (1, 3),
        lambda: "avg_pool3d: kernel_size must be a single int, or a tuple of three ints",
    )
    kT = kernel_size[0]
    kH = kT if len(kernel_size) == 1 else kernel_size[1]
    kW = kT if len(kernel_size) == 1 else kernel_size[2]

    torch._check(
        not stride or len(stride) in (1, 3),
        lambda: "avg_pool3d: stride must be omitted, a single int, or a tuple of three ints",
    )
    torch._check(
        input.dtype not in [torch.uint8, torch.uint16, torch.uint32, torch.uint64],
        lambda: f""""avg_pool3d" not implemented for '{input.dtype.__str__()}'""",
    )
    dT = kT if not stride else stride[0]
    dH = kH if not stride else (dT if len(stride) == 1 else stride[1])
    dW = kW if not stride else (dT if len(stride) == 1 else stride[2])

    torch._check(
        len(padding) in (1, 3),
        lambda: "avg_pool3d: padding must be a single int, or a tuple of three ints",
    )
    padT = padding[0]
    padH = padT if len(padding) == 1 else padding[1]
    padW = padT if len(padding) == 1 else padding[2]

    torch._check(
        input.ndim in (4, 5),
        lambda: "non-empty 4D or 5D (batch mode) tensor expected for input",
    )

    torch._check(
        not divisor_override or divisor_override != 0,
        lambda: "divisor must be not zero",
    )

    nbatch = input.size(0)
    nslices = input.size(-4)
    itime = input.size(-3)
    iheight = input.size(-2)
    iwidth = input.size(-1)

    otime = pooling_output_shape(itime, kT, padT, dT, 1, ceil_mode)
    oheight = pooling_output_shape(iheight, kH, padH, dH, 1, ceil_mode)
    owidth = pooling_output_shape(iwidth, kW, padW, dW, 1, ceil_mode)

    pool3d_shape_check(
        input,
        nslices,
        kT,
        kH,
        kW,
        dT,
        dH,
        dW,
        padT,
        padH,
        padW,
        1,
        1,
        1,
        itime,
        iheight,
        iwidth,
        otime,
        oheight,
        owidth,
        "avg_pool3d()",
        check_input_size=True,
    )

    if input.ndim == 4:
        return input.new_empty((nslices, otime, oheight, owidth))
    else:
        return input.new_empty((nbatch, nslices, otime, oheight, owidth))


@register_meta(aten.avg_pool3d_backward)
@out_wrapper("grad_input")
def meta_avg_pool3d_backward(
    grad_output,
    input,
    kernel_size,
    stride,
    padding,
    ceil_mode,
    count_include_pad,
    divisor_override,
):
    torch._check(
        len(kernel_size) in (1, 3),
        lambda: "avg_pool3d: kernel_size must be a single int, or a tuple of three ints",
    )
    kT = kernel_size[0]
    kH = kT if len(kernel_size) == 1 else kernel_size[1]
    kW = kT if len(kernel_size) == 1 else kernel_size[2]

    torch._check(
        not stride or len(stride) in (1, 3),
        lambda: "avg_pool3d: stride must be omitted, a single int, or a tuple of three ints",
    )
    dT = kT if not stride else stride[0]
    dH = kH if not stride else (dT if len(stride) == 1 else stride[1])
    dW = kW if not stride else (dT if len(stride) == 1 else stride[2])

    torch._check(
        len(padding) in (1, 3),
        lambda: "avg_pool3d: padding must be a single int, or a tuple of three ints",
    )
    padT = padding[0]
    padH = padT if len(padding) == 1 else padding[1]
    padW = padT if len(padding) == 1 else padding[2]

    torch._check(
        input.ndim in (4, 5),
        lambda: "non-empty 4D or 5D (batch mode) tensor expected for input",
    )

    torch._check(
        not divisor_override or divisor_override != 0,
        lambda: "divisor must be not zero",
    )

    nslices = input.size(-4)
    itime = input.size(-3)
    iheight = input.size(-2)
    iwidth = input.size(-1)

    otime_for_shape_check = pooling_output_shape(itime, kT, padT, dT, 1, ceil_mode)
    oheight_for_shape_check = pooling_output_shape(iheight, kH, padH, dH, 1, ceil_mode)
    owidth_for_shape_check = pooling_output_shape(iwidth, kW, padW, dW, 1, ceil_mode)

    avg_pool3d_backward_shape_check(
        input,
        grad_output,
        nslices,
        kT,
        kH,
        kW,
        dT,
        dH,
        dW,
        padT,
        padH,
        padW,
        itime,
        iheight,
        iwidth,
        otime_for_shape_check,
        oheight_for_shape_check,
        owidth_for_shape_check,
        "avg_pool3d_backward()",
    )

    return input.new_empty(input.shape)


@register_meta(aten._adaptive_avg_pool2d.default)
def meta_adaptive_avg_pool2d(self, output_size):
    torch._check(
        self.ndim == 3 or self.ndim == 4,
        lambda: f"Expected 3D or 4D tensor, but got {self.shape}",
    )
    output_shape = self.shape[:-2] + tuple(output_size)
    memory_format = utils.suggest_memory_format(self)
    # need to set memory_format to preserve the memory format of the input
    # channel last input should have channel last output
    return torch.empty(
        output_shape,
        dtype=self.dtype,
        device=self.device,
        memory_format=memory_format,
    )


@register_meta(aten._adaptive_avg_pool3d.default)
def meta_adaptive_avg_pool3d(self, output_size):
    torch._check(
        self.ndim == 4 or self.ndim == 5,
        lambda: f"Expected 4D or 5D tensor, but got {self.shape}",
    )
    return self.new_empty(self.shape[:-3] + tuple(output_size))


@register_meta(aten._adaptive_avg_pool2d_backward.default)
def meta__adaptive_avg_pool2d_backward(grad_out, self):
    ndim = grad_out.ndim
    for i in range(1, ndim):
        torch._check(
            grad_out.size(i) > 0,
            lambda: f"adaptive_avg_pool2d_backward(): Expected grad_output to have non-zero \
                      size for non-batch dimensions, {grad_out.shape} with dimension {i} being empty",
        )
    torch._check(
        ndim == 3 or ndim == 4,
        lambda: f"adaptive_avg_pool2d_backward(): Expected 3D or 4D tensor, but got {self.shape}",
    )
    torch._check(
        self.dtype == grad_out.dtype,
        lambda: f"expected dtype {self.dtype} for `grad_output` but got dtype {grad_out.dtype}",
    )
    memory_format = torch.contiguous_format
    if is_channels_last(self):
        memory_format = torch.channels_last
    return self.new_empty(self.shape).to(memory_format=memory_format)


@register_meta(aten._adaptive_avg_pool3d_backward)
@out_wrapper("grad_input")
def meta__adaptive_avg_pool3d_backward(grad_output, self):
    _adaptive_pool_empty_output_check(grad_output, "adaptive_avg_pool3d_backward")
    return torch.empty_like(self, memory_format=torch.legacy_contiguous_format)


def _adaptive_pool_empty_output_check(grad_output: Tensor, arg_name: str):
    ndim = grad_output.ndim
    for i in range(1, ndim):
        torch._check(
            grad_output.size(i) > 0,
            lambda: (
                f"{arg_name}(): Expected grad_output to have non-zero size for non-batch dimensions, "
                f"but grad_output has sizes {grad_output.shape} with dimension {i} being empty"
            ),
        )


@register_meta(aten.adaptive_max_pool2d)
@out_wrapper("out", "indices")
def meta_adaptive_max_pool2d(input, output_size):
    ndim = input.ndim
    torch._check(
        ndim in (3, 4),
        lambda: f"adaptive_max_pool2d(): Expected 3D or 4D tensor, but got: {input.shape}",
    )
    for i in range(1, ndim):
        torch._check(
            input.size(i) > 0,
            lambda: (
                f"adaptive_max_pool2d(): Expected input to have non-zero size for non-batch dimensions, "
                f"but input has sizes {input.shape} with dimension {i} being empty"
            ),
        )

    torch._check(
        len(output_size) == 2,
        lambda: "adaptive_max_pool2d(): internal error: output_size.size() must be 2",
    )

    dimH = 1
    sizeB = 1
    sizeD = 0

    if input.ndim == 4:
        sizeB = input.size(0)
        dimH += 1

    sizeD = input.size(dimH - 1)
    osizeH, osizeW = output_size

    if input.ndim == 3:
        out_shape = (sizeD, osizeH, osizeW)
        out = input.new_empty(out_shape)
        indices = input.new_empty(out_shape, dtype=torch.int64)
        return out, indices
    else:
        out_shape = (sizeB, sizeD, osizeH, osizeW)  # type: ignore[assignment]
        memory_format = utils.suggest_memory_format(input)
        out = input.new_empty(out_shape).to(memory_format=memory_format)
        indices = input.new_empty(out_shape, dtype=torch.int64).to(
            memory_format=memory_format
        )
        return out, indices


@register_meta(aten.adaptive_max_pool2d_backward)
@out_wrapper("grad_input")
def meta_adaptive_max_pool2d_backward(grad_output, input, indices):
    ndim = grad_output.ndim
    torch._check(
        ndim in (3, 4),
        lambda: f"adaptive_max_pooling2d_backward(): Expected 3D or 4D grad_output, but got: {grad_output.shape}",
    )

    _adaptive_pool_empty_output_check(grad_output, "adaptive_max_pool2d_backward")

    torch._check(
        input.dtype == grad_output.dtype,
        lambda: f"expected dtype {input.dtype} for `grad_output` but got dtype {grad_output.dtype}",
    )

    memory_format = utils.suggest_memory_format(input)
    return input.new_empty(input.shape).to(memory_format=memory_format)


@register_meta(aten.adaptive_max_pool3d)
@out_wrapper("out", "indices")
def meta_adaptive_max_pool3d(input, output_size):
    ndim = input.ndim
    torch._check(
        ndim in (4, 5),
        lambda: f"adaptive_max_pool3d(): Expected 4D or 5D tensor, but got: {input.shape}",
    )
    for i in range(1, ndim):
        torch._check(
            input.size(i) > 0,
            lambda: (
                f"adaptive_max_pool3d(): Expected input to have non-zero size for non-batch dimensions, "
                f"but input has sizes {input.shape} with dimension {i} being empty"
            ),
        )

    torch._check(
        len(output_size) == 3,
        lambda: "adaptive_max_pool3d(): internal error: output_size.size() must be 3",
    )

    dimD = 0
    sizeB = 1
    sizeD = 0

    if ndim == 5:
        sizeB = input.size(0)
        dimD += 1

    sizeD = input.size(dimD)
    osizeT, osizeH, osizeW = output_size

    if ndim == 4:
        out_shape = (sizeD, osizeT, osizeH, osizeW)
    else:
        out_shape = (sizeB, sizeD, osizeT, osizeH, osizeW)  # type: ignore[assignment]

    out = input.new_empty(out_shape)
    indices = input.new_empty(out_shape, dtype=torch.int64)

    return out, indices


@register_meta(aten.adaptive_max_pool3d_backward)
@out_wrapper("grad_input")
def meta_adaptive_max_pool3d_backward(grad_output, input, indices):
    _adaptive_pool_empty_output_check(grad_output, "adaptive_max_pool3d_backward")
    return input.new_empty(input.shape)


@register_meta(aten.repeat_interleave.Tensor)
def meta_repeat_interleave_Tensor(repeats, output_size=None):
    if output_size is None:
        raise RuntimeError("cannot repeat_interleave a meta tensor without output_size")
    return repeats.new_empty(output_size)


@register_meta([aten.complex.default, aten.complex.out])
@out_wrapper()
def meta_complex(real, imag):
    assert real.dtype.is_floating_point
    assert imag.dtype.is_floating_point
    out_shape = _broadcast_shapes(real.shape, imag.shape)
    return real.new_empty(out_shape, dtype=corresponding_complex_dtype(real.dtype))


@register_meta([aten.nonzero_static.default, aten.nonzero_static.out])
@out_wrapper()
def nonzero_static(self, *, size, fill_value: int = -1):
    return self.new_empty((size, self.dim()), dtype=torch.long)


@register_meta([torch.ops.aten.nonzero.default, torch.ops.aten.nonzero.out])
@out_wrapper()
def nonzero(self):
    torch._check_not_implemented(
        exp_config.meta_nonzero_assume_all_nonzero,
        lambda: "The register_meta function for torch.nonzero() raises unimplemented by default, "
        "as a correct data-independent implementation does not exist. This implementation "
        "returns a fake value, assuming all elements of the tensor are non-zero. "
        "To enable this registration, please set "
        "'torch.fx.experimental._config.meta_nonzero_assume_all_nonzero' to True.",
    )
    return torch.empty_strided(
        (self.numel(), self.dim()),
        (1, self.numel()),
        dtype=torch.long,
        device=self.device,
    )


@register_meta([aten.index.Tensor, aten._unsafe_index.Tensor])
def meta_index_Tensor(self, indices):
    torch._check(bool(indices), lambda: "at least one index must be provided")
    # aten::index is the internal advanced indexing implementation
    # checkIndexTensorTypes and expandTensors
    result: list[Optional[Tensor]] = []
    for i, index in enumerate(indices):
        if index is not None:
            torch._check(
                index.dtype in [torch.long, torch.int, torch.int8, torch.bool],
                lambda: "tensors used as indices must be long, int, byte or bool tensors",
            )
            if index.dtype in [torch.int8, torch.bool]:
                nonzero = index.nonzero()
                k = len(result)
                torch._check_index(
                    k + index.ndim <= self.ndim,
                    lambda: f"too many indices for tensor of dimension {self.ndim}",
                )
                for j in range(index.ndim):
                    torch._check_index(
                        index.shape[j] == self.shape[k + j],
                        lambda: f"The shape of the mask {index.shape} at index {i} "
                        f"does not match the shape of the indexed tensor {self.shape} at index {k + j}",
                    )
                    result.append(nonzero.select(1, j))
            else:
                result.append(index)
        else:
            result.append(index)
    indices = result
    torch._check(
        len(indices) <= self.ndim,
        lambda: f"too many indices for tensor of dimension {self.ndim} (got {len(indices)})",
    )
    # expand_outplace
    import torch._refs as refs  # avoid import cycle in mypy

    indices = list(refs._maybe_broadcast(*indices))
    # add missing null tensors
    while len(indices) < self.ndim:
        indices.append(None)

    # hasContiguousSubspace
    #   true if all non-null tensors are adjacent
    # See:
    # https://numpy.org/doc/stable/user/basics.indexing.html#combining-advanced-and-basic-indexing
    # https://stackoverflow.com/questions/53841497/why-does-numpy-mixed-basic-advanced-indexing-depend-on-slice-adjacency
    state = 0
    has_contiguous_subspace = False
    for index in indices:
        if state == 0:
            if index is not None:
                state = 1
        elif state == 1:
            if index is None:
                state = 2
        else:
            if index is not None:
                break
    else:
        has_contiguous_subspace = True

    # transposeToFront
    # This is the logic that causes the newly inserted dimensions to show up
    # at the beginning of the tensor, if they're not contiguous
    if not has_contiguous_subspace:
        dims = []
        transposed_indices = []
        for i, index in enumerate(indices):
            if index is not None:
                dims.append(i)
                transposed_indices.append(index)
        for i, index in enumerate(indices):
            if index is None:
                dims.append(i)
                transposed_indices.append(index)
        self = self.permute(dims)
        indices = transposed_indices

    # AdvancedIndex::AdvancedIndex
    # Now we can assume the indices have contiguous subspace
    # This is simplified from AdvancedIndex which goes to more effort
    # to put the input and indices in a form so that TensorIterator can
    # take them.  If we write a ref for this, probably that logic should
    # get implemented
    before_shape: list[int] = []
    after_shape: list[int] = []
    replacement_shape: list[int] = []
    for dim, index in enumerate(indices):
        if index is None:
            if replacement_shape:
                after_shape.append(self.shape[dim])
            else:
                before_shape.append(self.shape[dim])
        else:
            replacement_shape = list(index.shape)

    def _restride_src(self):
        """
        This follows restride_src in TensorAdvancedIndexing.cpp
        """
        shape = before_shape + replacement_shape + after_shape
        strides = list(self.stride())
        strides[len(before_shape) : len(self.shape) - len(after_shape)] = [0] * len(
            replacement_shape
        )
        return self.as_strided(shape, strides)

    out = self.new_empty(before_shape + replacement_shape + after_shape)
    from torch.fx.experimental.symbolic_shapes import guard_or_false

    if guard_or_false(self.numel() == 0):
        # No need to worry about the output strides if self is empty.
        return out

    # Try to follow eager to decide the output stride based on self.
    # Note that perm here is the reverse of the 'perm_' decided by
    # TensorIteratorBase::reorder_dimensions
    restrided_self = _restride_src(self)
    perm = utils.compute_elementwise_output_logical_to_physical_perm(restrided_self)

    # Follow TensorIteratorBase::allocate_or_resize_outputs
    if list(perm) != list(range(len(perm))):
        perm_shape = utils.apply_perm(out.shape, perm)
        new_stride = utils.make_contiguous_strides_for(perm_shape)
        new_stride = utils.apply_perm(new_stride, utils.invert_perm(perm))
        out = out.as_strided(out.size(), new_stride)
    return out


@register_meta([aten.convolution_backward.default])
def meta_convolution_backward(
    grad_output_,
    input_,
    weight_,
    bias_sizes_opt,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
    output_mask,
):
    # High level logic taken from slow_conv3d_backward_cpu which should
    # be representative of all convolution_backward impls
    backend_grad_input = None
    backend_grad_weight = None
    backend_grad_bias = None

    if output_mask[0]:
        backend_grad_input = grad_output_.new_empty(input_.size())
    if output_mask[1]:
        backend_grad_weight = grad_output_.new_empty(weight_.size())
    if output_mask[2]:
        backend_grad_bias = grad_output_.new_empty(bias_sizes_opt)

    return (backend_grad_input, backend_grad_weight, backend_grad_bias)


@register_meta([aten.addbmm.default, aten.addbmm.out])
@out_wrapper(exact_dtype=True)
def meta_addbmm(self, batch1, batch2, *, beta=1, alpha=1):
    dim1 = batch1.size(1)
    dim2 = batch2.size(2)
    self = self.expand((dim1, dim2))
    torch._check(batch1.dim() == 3, lambda: "batch1 must be a 3D tensor")
    torch._check(batch2.dim() == 3, lambda: "batch2 must be a 3D tensor")
    torch._check(
        batch1.size(0) == batch2.size(0),
        lambda: f"batch1 and batch2 must have same number of batches, got {batch1.size(0)} and {batch2.size(0)}",
    )
    torch._check(
        batch1.size(2) == batch2.size(1),
        lambda: (
            f"Incompatible matrix sizes for bmm ({batch1.size(1)}x{batch1.size(2)} "
            f"and {batch2.size(1)}x{batch2.size(2)})"
        ),
    )
    torch._check(
        self.size(0) == dim1 and self.size(1) == dim2,
        lambda: "self tensor does not match matmul output shape",
    )
    return self.new_empty(self.size())


@register_meta([aten.randint_like.Tensor])
def meta_randint_like(self, high, **kwargs):
    return self.new_empty(self.size())


@register_meta([aten._fused_adam_.default, aten._fused_adamw_.default])
def meta__fused_adam_(
    self,
    grads,
    exp_avgs,
    exp_avg_sqs,
    max_exp_avg_sqs,
    state_steps,
    *,
    lr,
    beta1,
    beta2,
    weight_decay,
    eps,
    amsgrad,
    maximize,
    grad_scale=None,
    found_inf=None,
):
    for l in [self, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps]:
        torch._check(
            isinstance(l, list),
            lambda: f"exponent must be a tensor list but got {type(l)}",
        )


@register_meta([aten._fused_adam.default])
def meta__fused_adam(
    self,
    grads,
    exp_avgs,
    exp_avg_sqs,
    max_exp_avg_sqs,
    state_steps,
    *,
    lr,
    beta1,
    beta2,
    weight_decay,
    eps,
    amsgrad,
    maximize,
    grad_scale=None,
    found_inf=None,
):
    for l in [self, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps]:
        torch._check(
            isinstance(l, list),
            lambda: f"exponent must be a tensor list but got {type(l)}",
        )

    def empty_like_list(tensor_list):
        return [torch.empty_like(t) for t in tensor_list]

    return (
        empty_like_list(self),
        empty_like_list(grads),
        empty_like_list(exp_avgs),
        empty_like_list(exp_avg_sqs),
        empty_like_list(max_exp_avg_sqs),
    )


@register_meta([aten._int_mm])
@out_wrapper()
def meta__int_mm(a, b):
    torch._check(a.dim() == 2, lambda: "a must be a 2D tensor")
    torch._check(b.dim() == 2, lambda: "b must be a 2D tensor")
    torch._check(
        a.dtype is torch.int8,
        lambda: f"expected self to be int8, got {a.dtype}",
    )
    torch._check(
        b.dtype is torch.int8,
        lambda: f"expected mat2 to be int8, got {b.dtype}",
    )
    torch._check(
        a.size(1) == b.size(0),
        lambda: (
            f"Incompatible matrix sizes for _int_mm ({a.size(0)}x{a.size(1)} "
            f"and {b.size(0)}x{b.size(1)})"
        ),
    )
    return a.new_empty((a.size(0), b.size(1)), dtype=torch.int32)


@register_meta([aten._convert_weight_to_int4pack])
def meta__convert_weight_to_int4pack(w, inner_k_tiles):
    torch._check(w.dim() == 2, lambda: "w must be a 2D tensor")
    torch._check(
        w.dtype is torch.uint8,
        lambda: f"expected w to be uint8, got {w.dtype}",
    )
    n = w.size(0)
    k = w.size(1) * 2  # w is [n][k / 2] uint8
    return w.new_empty(
        (
            n // 8,
            k // (inner_k_tiles * 16),
            32,
            inner_k_tiles // 2,
        ),
        dtype=torch.int32,
    )


@register_meta([aten._convert_weight_to_int4pack_for_cpu])
def meta__convert_weight_to_int4pack_for_cpu(w, inner_k_tiles):
    torch._check(w.dim() == 2, lambda: "w must be a 2D tensor")
    torch._check(
        w.dtype is torch.int32,
        lambda: f"expected w to be int32, got {w.dtype}",
    )
    n = w.size(0)
    k = w.size(1)  # w is [n][k] int32
    return w.new_empty(
        (n, k // 2),
        dtype=torch.uint8,
    )


@register_meta([aten._weight_int4pack_mm])
def meta__weight_int4pack_mm(x, w, q_group_size, q_scale_and_zeros):
    torch._check(x.dim() == 2, lambda: "x must be a 2D tensor")
    torch._check(w.dim() == 4, lambda: "w must be a 4D tensor")
    torch._check(
        x.dtype in [torch.float32, torch.float16, torch.bfloat16],
        lambda: f"expected x to be f32/f16/bf16, got {x.dtype}",
    )
    torch._check(
        w.dtype is torch.int32,
        lambda: f"expected w to be int32, got {w.dtype}",
    )
    return x.new_empty(x.size(0), w.size(0) * 8, dtype=x.dtype)


@register_meta([aten._weight_int4pack_mm_for_cpu])
def meta__weight_int4pack_mm_for_cpu(x, w, q_group_size, q_scale_and_zeros):
    torch._check(x.dim() == 2, lambda: "x must be a 2D tensor")
    torch._check(w.dim() == 2, lambda: "w must be a 2D tensor")
    torch._check(
        x.dtype in [torch.float32, torch.float16, torch.bfloat16],
        lambda: f"expected x to be f32/f16/bf16, got {x.dtype}",
    )
    torch._check(
        w.dtype is torch.uint8,
        lambda: f"expected w to be uint8, got {w.dtype}",
    )
    return x.new_empty(x.size(0), w.size(0), dtype=x.dtype)


@register_meta([aten._weight_int4pack_mm_with_scales_and_zeros])
def _weight_int4pack_mm_with_scales_and_zeros(x, w, q_group_size, qScale, qZeros):
    torch._check(x.dim() == 2, lambda: "x must be a 2D tensor")
    torch._check(w.dim() == 2, lambda: "w must be a 2D tensor")
    torch._check(
        x.dtype in [torch.float32, torch.float16, torch.bfloat16],
        lambda: f"expected x to be f32/f16/bf16, got {x.dtype}",
    )
    torch._check(
        w.dtype is torch.int32,
        lambda: f"expected w to be int32, got {w.dtype}",
    )
    return x.new_empty(x.size(0), w.size(0), dtype=x.dtype)


def kai_roundup(a: int, b: int) -> int:
    return ((a + b - 1) // b) * b


def get_kai_packed_weight_size(n_bits, N, K, groupsize):
    if n_bits == 4:
        if groupsize == K:  # channelwise
            # dotprod params only [1x8x32_neon_dotprod]
            kai_nr = 8
            kai_kr = 16
            kai_sr = 2
            kai_num_bytes_sum_rhs = 4  # sizeof(int32_t)
            kai_num_bytes_multiplier_rhs = 4  # sizeof(float)
            kai_num_bytes_bias = 4  # sizeof(float)

            def kai_k_roundedup(k, kr, sr):
                # Since we pack a float and int32 value at the end of the row,
                # we must make sure that k is a multiple of 4 for alignment
                kr_sr_roundedup4 = kai_roundup(kr * sr, 4)
                return kai_roundup(k, kr_sr_roundedup4)

            def kai_get_rhs_packed_stride_rhs_pack_nxk_qsi4cxp_qsu4cxs1s0(
                k, nr, kr, sr
            ):
                k_internal = kai_k_roundedup(k, kr, sr)

                assert (k_internal % 2) == 0, "k_internal must be even"

                return nr * (
                    (k_internal // 2)
                    + kai_num_bytes_multiplier_rhs
                    + kai_num_bytes_sum_rhs
                    + kai_num_bytes_bias
                )

            def kai_get_rhs_packed_size_rhs_pack_nxk_qsi4cxp_qsu4cxs1s0(
                n, k, nr, kr, sr
            ):
                num_rows = kai_roundup(n, nr) // nr

                return (
                    num_rows
                    * kai_get_rhs_packed_stride_rhs_pack_nxk_qsi4cxp_qsu4cxs1s0(
                        k, nr, kr, sr
                    )
                )

            return kai_get_rhs_packed_size_rhs_pack_nxk_qsi4cxp_qsu4cxs1s0(
                N, K, kai_nr, kai_kr, kai_sr
            )
        elif groupsize % 32 == 0 and K % groupsize == 0:  # groupwise
            kai_nr = 8
            kai_kr = 16
            kai_sr = 2
            kai_num_bytes_sum_rhs = 4
            kai_num_bytes_bias = 4
            kai_nr_multiple_of = 4
            kai_bl_multiple_of = 32

            def kai_get_rhs_packed_size_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0(
                n, k, nr, kr, sr, bl
            ):
                assert (bl % kr) == 0
                assert (nr % kai_nr_multiple_of) == 0
                assert (bl % kai_bl_multiple_of) == 0

                num_rows = kai_roundup(n, nr) // nr

                return (
                    num_rows
                    * kai_get_rhs_packed_stride_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0(
                        k, nr, kr, sr, bl
                    )
                )

            def kai_get_rhs_packed_stride_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0(
                k, nr, kr, sr, bl
            ):
                assert (bl % kr) == 0
                assert (nr % kai_nr_multiple_of) == 0
                assert (bl % kai_bl_multiple_of) == 0

                # kr and sr are unused in the calculation
                num_bytes_multiplier_rhs = kai_get_bf16_datatype_size_in_bytes()
                num_blocks_per_row = kai_num_blocks_per_row(k, bl)
                num_bytes_per_block = kai_num_bytes_per_block(
                    bl, num_bytes_multiplier_rhs
                )

                return nr * (
                    (num_bytes_per_block * num_blocks_per_row)
                    + kai_num_bytes_sum_rhs
                    + kai_num_bytes_bias
                )

            # This function returns size of these datatypes stored as enum. We modify it to just return bf16 datatype
            # https://gitlab.arm.com/kleidi/kleidiai/-/blob/main/kai/kai_common.h?ref_type=heads#L55
            def kai_get_bf16_datatype_size_in_bytes():
                return 2  # 2 bytes

            def kai_num_blocks_per_row(k, bl):
                assert (bl % kai_bl_multiple_of) == 0
                return kai_roundup(k, bl) // bl

            def kai_num_bytes_per_block(bl, num_bytes_multiplier_rhs):
                assert (bl % kai_bl_multiple_of) == 0
                return (bl // 2) + num_bytes_multiplier_rhs

            return kai_get_rhs_packed_size_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0(
                N, K, kai_nr, kai_kr, kai_sr, groupsize
            )


@register_meta([aten._dyn_quant_pack_4bit_weight])
def meta__dyn_quant_pack_4bit_weight(
    weights, scales_zeros, bias: Optional[Tensor], block_size, in_features, out_features
):
    torch._check(
        weights.dtype is torch.uint8,
        lambda: f"expected w to be uint8, got {weights.dtype}",
    )
    if torch.backends.kleidiai.is_available() and (
        (block_size == in_features and scales_zeros.dtype == torch.float)
        or (
            block_size < in_features
            and block_size % 32 == 0
            and in_features % block_size == 0
            and scales_zeros.dtype == torch.bfloat16
        )
    ):
        packed_weight_size = get_kai_packed_weight_size(
            4, out_features, in_features, block_size
        )
        return weights.new_empty(int(packed_weight_size), dtype=torch.uint8)
    packed_weight_size = weights.numel() + scales_zeros.numel()
    return weights.new_empty(packed_weight_size, dtype=torch.float)


@register_meta([aten._dyn_quant_matmul_4bit])
def meta__dyn_quant_matmul_4bit(
    inp,
    packed_weights,
    block_size,
    in_features,
    out_features,
):
    torch._check(inp.dim() == 2, lambda: "input must be a 2D tensor")
    torch._check(
        inp.dtype in [torch.float32],
        lambda: f"expected input to be f32, got {inp.dtype}",
    )
    M = inp.size(0)
    return inp.new_empty(M, out_features, dtype=inp.dtype)


@register_meta([aten._weight_int8pack_mm])
def meta__weight_int8pack_mm(x, w, q_scales):
    torch._check(x.dim() == 2, lambda: "x must be a 2D tensor")
    torch._check(
        x.dtype in [torch.float32, torch.float16, torch.bfloat16],
        lambda: f"expected x to be f32/f16/bf16, got {x.dtype}",
    )
    torch._check(w.dim() == 2, lambda: "w must be a 2D tensor")
    torch._check(
        w.dtype is torch.int8,
        lambda: f"expected w to be int8, got {w.dtype}",
    )
    return x.new_empty(x.size(0), w.size(0), dtype=x.dtype)


@register_meta(aten._cdist_forward.default)
def meta_cdist_forward(x1, x2, p, compute_mode):
    torch._check(
        x1.dim() >= 2,
        lambda: f"cdist only supports at least 2D tensors, X1 got: {x1.dim()}D",
    )
    torch._check(
        x2.dim() >= 2,
        lambda: f"cdist only supports at least 2D tensors, X2 got: {x2.dim()}D",
    )
    torch._check(
        x1.size(-1) == x2.size(-1),
        lambda: f"X1 and X2 must have the same number of columns. X1: {x1.size(-1)} X2: {x2.size(-1)}",
    )
    torch._check(
        utils.is_float_dtype(x1.dtype),
        lambda: "cdist only supports floating-point dtypes, X1 got: {x1.dtype}",
    )
    torch._check(
        utils.is_float_dtype(x2.dtype),
        lambda: "cdist only supports floating-point dtypes, X2 got: {x2.dtype}",
    )
    torch._check(p >= 0, lambda: "cdist only supports non-negative p values")
    torch._check(
        compute_mode in (None, 1, 2),
        lambda: f"possible modes: None, 1, 2, but was: {compute_mode}",
    )
    r1 = x1.size(-2)
    r2 = x2.size(-2)
    batch_tensor1 = x1.shape[:-2]
    batch_tensor2 = x2.shape[:-2]
    output_shape = list(torch.broadcast_shapes(batch_tensor1, batch_tensor2))
    output_shape.extend([r1, r2])
    return x1.new_empty(output_shape)


@register_meta(aten._cdist_backward)
@out_wrapper()
def meta_cdist_backward(grad, x1, x2, p, cdist):
    c1 = x1.shape[-1]
    r1 = x1.shape[-2]
    r2 = x2.shape[-2]
    batch_tensor1 = x1.shape[:-2]
    batch_tensor2 = x2.shape[:-2]
    expand_batch_portion = list(torch.broadcast_shapes(batch_tensor1, batch_tensor2))
    tensor1_expand_size = expand_batch_portion.copy()
    tensor1_expand_size.extend([r1, c1])
    batch_product = math.prod(expand_batch_portion)
    if r1 == 0 or r2 == 0 or c1 == 0 or batch_product == 0:
        return torch.zeros_like(x1)
    if tensor1_expand_size != list(x1.shape):
        x1 = x1.expand(tensor1_expand_size)
    return torch.empty_like(x1, memory_format=torch.contiguous_format)


# NB: This meta function accepts non-meta arguments!  When this behavior
# was originally introduced this was accidental, but it is now load bearing
# as people are using this so that they can conveniently test code involving
# embeddings (feeding CPU tensor inputs with meta device EmbeddingBag module)
@register_meta(aten._embedding_bag.default)
def meta_embedding_bag(
    weight,
    indices,
    offsets,
    scale_grad_by_freq=False,
    mode=0,
    sparse=False,
    per_sample_weights=None,
    include_last_offset=False,
    padding_idx=-1,
):
    torch._check(
        indices.dtype in (torch.long, torch.int),
        lambda: f"expected indices to be long or int, got {indices.dtype}",
    )
    torch._check(
        offsets.dtype in (torch.long, torch.int),
        lambda: f"expected offsets to be long or int, got {offsets.dtype}",
    )
    torch._check(
        utils.is_float_dtype(weight.dtype),
        lambda: f"expected weight to be floating point type, got {weight.dtype}",
    )

    num_bags = offsets.size(0)
    if include_last_offset:
        torch._check(
            num_bags >= 1,
            lambda: "include_last_offset: numBags should be at least 1",
        )
        num_bags -= 1

    output = weight.new_empty(num_bags, weight.size(1))

    if per_sample_weights is not None:
        torch._check(
            mode == MODE_SUM,
            lambda: "embedding_bag: per_sample_weights only supported with mode='sum'",
        )
        torch._check(
            per_sample_weights.ndim == 1,
            lambda: f"expected per_sample_weights to be 1D tensor, got {per_sample_weights.ndim}D",
        )
        torch._check(
            per_sample_weights.numel() == indices.numel(),
            lambda: (
                f"expected per_sample_weights.numel() ({per_sample_weights.numel()} "
                f"to be the same as indices.numel() ({indices.numel()})"
            ),
        )

    def is_fast_path_index_select_scale(src, scale, output, padding_idx):
        return (
            is_fast_path_index_select(src, output, padding_idx) and scale.stride(0) == 1
        )

    def is_fast_path_index_select(src, output, padding_idx):
        return (
            (src.dtype == torch.float or src.dtype == torch.half)
            and src.stride(1) == 1
            and output.stride(1) == 1
            and padding_idx < 0
        )

    def is_fast_path(src, scale, output, padding_idx):
        if scale is not None:
            return is_fast_path_index_select_scale(src, scale, output, padding_idx)
        else:
            return is_fast_path_index_select(src, output, padding_idx)

    if device_hint(offsets) != "cpu":
        offset2bag = indices.new_empty(indices.size(0))
        bag_size = indices.new_empty(offsets.size())
        if mode == MODE_MAX:
            max_indices = indices.new_empty(num_bags, weight.size(1))
        else:
            max_indices = indices.new_empty(0)
    else:
        fast_path_sum = is_fast_path(weight, per_sample_weights, output, padding_idx)
        if mode in (MODE_MEAN, MODE_MAX) or not fast_path_sum:
            offset2bag = offsets.new_empty(indices.size(0))
        else:
            offset2bag = offsets.new_empty(0)
        bag_size = offsets.new_empty(num_bags)
        # This part of the logic comes from make_max_indices_out in EmbeddingBag.cpp
        numBags = offsets.shape[0]
        if mode == MODE_MAX:
            if include_last_offset:
                torch._check(
                    numBags >= 1,
                    lambda: "include_last_offset: numBags should be at least 1",
                )
                numBags -= 1
            max_indices = offsets.new_empty(numBags, weight.shape[1])
        else:
            max_indices = offsets.new_empty(bag_size.size())
    return output, offset2bag, bag_size, max_indices


@register_meta(aten._embedding_bag_forward_only.default)
def meta_embedding_bag_forward_only(weight, indices, offsets, *args):
    output, offset2bag, bag_size, max_indices = meta_embedding_bag(
        weight, indices, offsets, *args
    )
    if device_hint(offsets) == "cpu":
        bag_size = offsets.new_empty(offsets.size())
    return output, offset2bag, bag_size, max_indices


def _get_reduction_dtype(input, dtype, promote_int_to_long=True):
    # if specified, dtype takes precedence
    if dtype:
        return dtype

    if input.dtype.is_floating_point or input.dtype.is_complex:
        return input.dtype
    elif promote_int_to_long:
        return torch.long

    return input.dtype


@register_meta([aten.nansum.default, aten.nansum.out])
@out_wrapper()
def meta_nansum(input, dims=None, keepdim=False, *, dtype=None):
    output_dtype = _get_reduction_dtype(input, dtype, promote_int_to_long=True)
    dims = utils.reduction_dims(input.shape, dims)
    output_shape = _compute_reduction_shape(input, dims, keepdim)
    return input.new_empty(output_shape, dtype=output_dtype)


@register_meta([aten.median.default, aten.nanmedian.default])
def meta_median(input):
    output_shape = utils.compute_reduction_output_shape(
        input.shape, tuple(range(input.dim()))
    )
    return input.new_empty(output_shape)


@register_meta(
    [
        aten.median.dim,
        aten.median.dim_values,
        aten.nanmedian.dim,
        aten.nanmedian.dim_values,
        aten.mode.default,
        aten.mode.values,
    ]
)
@out_wrapper("values", "indices")
def meta_median_mode_dim(input, dim=-1, keepdim=False):
    if device_hint(input) == "cuda":
        utils.alert_not_deterministic("median CUDA with indices output")
    dim = utils.reduction_dims(input.shape, (dim,))
    output_shape = _compute_reduction_shape(input, dim, keepdim)
    return (
        input.new_empty(output_shape),
        input.new_empty(output_shape, dtype=torch.long),
    )


@register_meta(aten.logical_not_.default)
def meta_logical_not_(self):
    return self


@register_meta(aten.repeat.default)
def meta_repeat(self, repeats):
    torch._check(
        len(repeats) >= self.dim(),
        lambda: "Number of dimensions of repeat dims can not be smaller than number of dimensions of tensor",
    )
    for i, rep in enumerate(repeats):
        torch._check(
            rep >= 0,
            lambda: f"Repeats cannot be negative, found {rep} at index {i}",
        )
    # Add new leading dimensions to the tensor if the
    # number of target dimensions is larger than the
    # number of source dimensions.
    num_new_dimensions = len(repeats) - self.dim()
    padded_size = (1,) * num_new_dimensions + tuple(self.shape)
    target_size = [padded_size[i] * repeats[i] for i in range(len(repeats))]
    return self.new_empty(target_size)


@register_meta(aten.zero_.default)
def meta_zero_(self):
    return self


@register_meta(
    [
        aten.mul_.Scalar,
        aten.div_.Scalar,
        aten.mul_.Tensor,
        aten.div_.Tensor,
        aten.logical_and_.default,
        aten.logical_or_.default,
        aten.logical_xor_.default,
    ],
)
def meta_binop_inplace(self, other):
    if isinstance(other, torch.Tensor):
        check_inplace_broadcast(self.shape, other.shape)
    return self


@register_meta(
    [
        aten.add_.Scalar,
        aten.sub_.Scalar,
        aten.add_.Tensor,
        aten.sub_.Tensor,
    ],
)
def meta_binop_inplace_alpha(self, other, alpha=1):
    """
    Some checks for inplace ops.
    Checks for promotion rules for some dtypes.
    int.add/sub_(float) and bool.add/sub_(others) are rejected.
    Promoting in these in-place operations would require reallocating
    and copying over elements, hence not allowed.
    Checks for alpha param.
    """

    def is_integeric(arg):
        if isinstance(arg, TensorLike):
            return utils.is_integer_dtype(arg.dtype)
        else:
            return isinstance(arg, IntLike)

    def is_floatic(arg):
        if isinstance(arg, TensorLike):
            return utils.is_float_dtype(arg.dtype)
        else:
            return isinstance(arg, FloatLike)

    def is_booleanic(arg):
        if isinstance(arg, TensorLike):
            return utils.is_boolean_dtype(arg.dtype)
        else:
            return isinstance(arg, BoolLike)

    # Do not allow int+float->int in-place
    if is_integeric(self) and is_floatic(other):
        raise RuntimeError(
            "Promotion of int.add/sub_(float) in in-place ops are not possible due to element size change."
        )

    # Do not allow bool+other->bool in-place
    if is_booleanic(self) and not is_booleanic(other):
        raise RuntimeError(
            "Promotion of book.add/sub_(others) in in-place ops are not possible due to element size change."
        )

    if isinstance(other, torch.Tensor):
        check_inplace_broadcast(self.shape, other.shape)
    return self


@register_meta([aten.round.default, aten.round.decimals])
def meta_round(self, **kwargs):
    return elementwise_meta(
        self, type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    )


def shift_dtype_check(fn_name, self, val):
    torch._check(
        utils.is_integer_dtype(self.dtype),
        lambda: f"{fn_name}: Expected input tensor to have an integral dtype. Got {self.dtype}",
    )
    if isinstance(val, torch.Tensor):
        torch._check(
            utils.is_integer_dtype(val.dtype),
            lambda: f"{fn_name}: Expected shift value to have an integral dtype. Got {val.dtype}",
        )
    else:
        torch._check(
            isinstance(val, IntLike),
            lambda: f"{fn_name}: Expected shift value to be an int. Got {val}",
        )


@register_meta([aten.__rshift__.Tensor, aten.__rshift__.Scalar])
def meta_rshifts(self, other):
    shift_dtype_check("rshift", self, other)
    return elementwise_meta(
        self, other, type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    )


@register_meta([aten.__lshift__.Tensor, aten.__lshift__.Scalar])
def meta_lshifts(self, other):
    shift_dtype_check("lshift", self, other)
    return elementwise_meta(
        self, other, type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    )


@register_meta(aten.zero.default)
def meta_zero(self):
    return self.new_empty(self.shape)


@register_meta([aten.fill_.Tensor, aten.fill_.Scalar])
def meta_fill_(self, val):
    return self


@register_meta([aten.fill.Tensor, aten.fill.Scalar])
def meta_fill(self, val):
    return torch.empty_like(self)


@register_meta(aten.relu_.default)
def meta_relu_(self):
    return self


@register_meta(aten._add_relu.Tensor)
@out_wrapper()
def meta__add_relu(self, other, alpha=1) -> Tensor:
    return elementwise_meta(
        self, other, type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    )


@register_meta([aten.rrelu_with_noise])
@out_wrapper()
def meta_rrelu_with_noise(
    self, noise, lower=0.125, upper=0.3333333333333333, training=False, generator=None
):
    return torch.empty_like(self)


@register_meta([aten.rrelu_with_noise_functional])
def meta_rrelu_with_noise_functional(
    self, noise, lower=0.125, upper=0.3333333333333333, training=False, generator=None
):
    return torch.empty_like(self), torch.empty_like(noise)


@register_meta([aten.rrelu_with_noise_])
def meta_rrelu_with_noise_(
    self, lower=0.125, upper=0.3333333333333333, training=False, generator=None
):
    return self


@register_meta([aten.index_put.default, aten._unsafe_index_put.default])
def meta_index_put(self, indices, values, accumulate=False):
    return torch.empty_like(self)


@register_meta(aten.masked_fill_.Scalar)
def meta_masked_fill_(self, mask, value):
    check_inplace_broadcast(self.shape, mask.shape)
    return self


@register_meta(aten._masked_scale.default)
def meta__masked_scale(self, mask, scale):
    masked_scale = self.new_empty(self.size()).to(
        memory_format=utils.suggest_memory_format(self)
    )
    return masked_scale


@register_meta(aten.masked_scatter_)
def meta_masked_scatter_(self, mask, source):
    torch._check(
        mask.dtype in (torch.bool, torch.uint8), lambda: "Mask must be bool or uint8"
    )
    torch._check(
        self.dtype == source.dtype,
        lambda: "masked_scatter: expected self and source to have same "
        f"dtypes but got {self.dtype} and {source.dtype}",
    )
    return self


@register_meta(aten.masked_scatter)
@out_wrapper()
def meta_masked_scatter(self, mask, source):
    self, mask = _maybe_broadcast(self, mask)
    output = torch.empty_like(self, memory_format=torch.contiguous_format)
    return meta_masked_scatter_(output, mask, source)


@register_meta(aten.masked_scatter_backward)
def meta_masked_scatter_backward(self, mask, sizes):
    return self.new_empty(sizes)


@register_meta(aten.index_put_.default)
def meta_index_put_(self, indices, values, accumulate=False):
    return self


def common_meta_baddbmm_bmm(batch1, batch2, is_bmm, self_baddbmm=None, out_dtype=None):
    torch._check(batch1.dim() == 3, lambda: "batch1 must be a 3D tensor")
    torch._check(batch2.dim() == 3, lambda: "batch2 must be a 3D tensor")

    batch1_sizes = batch1.size()
    batch2_sizes = batch2.size()

    bs = batch1_sizes[0]
    contraction_size = batch1_sizes[2]
    res_rows = batch1_sizes[1]
    res_cols = batch2_sizes[2]
    output_size = (bs, res_rows, res_cols)

    torch._check(
        batch2_sizes[0] == bs and batch2_sizes[1] == contraction_size,
        lambda: f"Expected size for first two dimensions of batch2 tensor to be: [{bs}"
        f", {contraction_size}] but got: [{batch2_sizes[0]}, {batch2_sizes[1]}].",
    )
    if out_dtype:
        supported_out_dtype = (
            batch1.dtype == torch.float16 or batch1.dtype == torch.bfloat16
        ) and out_dtype == torch.float32
        torch._check(
            out_dtype == batch1.dtype or supported_out_dtype,
            lambda: "out_dtype only supported for torch.float32 output with float16/bfloat16 inputs or same as input dtypes",
        )
        output = batch2.new_empty(output_size).to(out_dtype)
    else:
        # TODO: handle out
        output = batch2.new_empty(output_size)

    if not is_bmm and self_baddbmm is not None:
        torch._check(self_baddbmm.dim() == 3, lambda: "self must be a 3D tensor")
        torch._check(
            self_baddbmm.size() == output_size,
            lambda: f"Expected an input tensor shape with shape {output_size} but got shape: {self_baddbmm.size()}",
        )

    return output


@register_meta(aten.bmm.default)
def meta_bmm(self, mat2):
    return common_meta_baddbmm_bmm(self, mat2, True)


@register_meta(aten.bmm.dtype)
def meta_bmm_dtype(self, mat2, out_dtype):
    return common_meta_baddbmm_bmm(self, mat2, True, out_dtype=out_dtype)


def div_rtn(x, y):
    q = x // y
    r = x % y
    # WARNING: explicit bool conversion here is necessary;
    # would be fixed by SymBool
    if r != 0 and (bool(r < 0) != bool(y < 0)):
        q -= 1
    return q


def pooling_output_shape_pad_lr(
    inputSize,
    kernelSize,
    pad_l,
    pad_r,
    stride,
    dilation,
    ceil_mode,
):
    outputSize = (
        div_rtn(
            inputSize
            + pad_l
            + pad_r
            - dilation * (kernelSize - 1)
            - 1
            + (stride - 1 if ceil_mode else 0),
            stride,
        )
        + 1
    )
    if ceil_mode:
        if (outputSize - 1) * stride >= inputSize + pad_l:
            outputSize -= 1
    return outputSize


def pooling_output_shape(inputSize, kernelSize, pad, stride, dilation, ceil_mode):
    torch._check(stride != 0, lambda: "stride should not be zero")
    torch._check(pad >= 0, lambda: f"pad must be non-negative, but got pad: {pad}")
    torch._check(
        pad <= ((kernelSize - 1) * dilation + 1) // 2,
        lambda: (
            f"pad should be at most half of effective kernel size, but got pad={pad}, "
            f"kernel_size={kernelSize} and dilation={dilation}"
        ),
    )
    return pooling_output_shape_pad_lr(
        inputSize, kernelSize, pad, pad, stride, dilation, ceil_mode
    )


def pool2d_shape_check(
    input,
    kH,
    kW,
    dH,
    dW,
    padH,
    padW,
    dilationH,
    dilationW,
    nInputPlane,
    inputHeight,
    inputWidth,
    outputHeight,
    outputWidth,
    memory_format,
):
    ndim = input.dim()
    nOutputPlane = nInputPlane

    torch._check(
        kW > 0 and kH > 0,
        lambda: "kernel size should be greater than zero, but got kH: {kH}, kW: {kW}",
    )
    torch._check(
        dW > 0 and dH > 0,
        lambda: "stride should be greater than zero, but got dH: {dH}, dW: {dW}",
    )
    torch._check(
        dilationH > 0 and dilationW > 0,
        lambda: "dilation should be greater than zero, but got dilationH: {dilationH}, dilationW: {dilationW}",
    )

    valid_dims = input.size(1) != 0 and input.size(2) != 0

    if memory_format == torch.channels_last:
        torch._check(
            ndim == 4 and valid_dims and input.size(3) != 0,
            lambda: "Expected 4D (batch mode) tensor expected for input with channels_last layout"
            " with optional 0 dim batch size for input, but got: {input.size()}",
        )
    else:
        torch._check(
            (ndim == 3 and input.size(0) != 0 and valid_dims)
            or (ndim == 4 and valid_dims and input.size(3) != 0),
            lambda: f"Expected 3D or 4D (batch mode) tensor with optional 0 dim batch size for input, but got: {input.size()}",
        )

    torch._check(
        kW // 2 >= padW and kH // 2 >= padH,
        lambda: "pad should be smaller than or equal to half of kernel size, but got "
        f"padW = {padW}, padH = {padH}, kW = {kW}, kH = {kH}",
    )

    torch._check(
        outputWidth >= 1 and outputHeight >= 1,
        lambda: f"Given input size: ({nInputPlane}x{inputHeight}x{inputWidth}). "
        f"Calculated output size: ({nOutputPlane}x{outputHeight}x{outputWidth}). "
        "Output size is too small",
    )


def pool3d_shape_check(
    input: Tensor,
    nslices: int,
    kT: int,
    kH: int,
    kW: int,
    dT: int,
    dH: int,
    dW: int,
    pT: int,
    pH: int,
    pW: int,
    dilationT: int,
    dilationH: int,
    dilationW: int,
    itime: int,
    iheight: int,
    iwidth: int,
    otime: int,
    oheight: int,
    owidth: int,
    fn_name: str,
    check_input_size: bool = False,
):
    ndim = input.ndim

    torch._check(
        kT > 0 and kW > 0 and kH > 0,
        lambda: (
            f"kernel size should be greater than zero, but got "
            f"kT: {kT}, kH: {kH}, kW: {kW}"
        ),
    )
    torch._check(
        dT > 0 and dW > 0 and dH > 0,
        lambda: (
            f"stride should be greater than zero, but got dT: {dT}, dH: {dH}, dW: {dW}"
        ),
    )
    torch._check(
        dilationT > 0 and dilationW > 0 and dilationH > 0,
        lambda: (
            f"dilation should be greater than zero, but got "
            f"dilationT: {dilationT}, dilationH: {dilationH}, dilationW: {dilationW}"
        ),
    )

    torch._check(
        ndim in (4, 5),
        lambda: f"{fn_name}: Expected 4D or 5D tensor for input, but got: {input.shape}",
    )

    for i in range(ndim):
        if ndim == 5 and i == 0:
            # size of batch-dim can be 0.
            continue
        torch._check(
            input.size(i) > 0,
            lambda: (
                f"{fn_name}: Expected input's non-batch dimensions to have positive length,"
                f" but input has a shape of {input.shape}"
                f" and non-batch dimension {input.size(i)} has length zero!"
            ),
        )

    if check_input_size:  # AveragePool3d
        torch._check(
            itime >= kT and iheight >= kH and iwidth >= kW,
            lambda: (
                f"input image (T: {itime} H: {iheight} W: {iwidth}) smaller than "
                f"kernel size (kT: {kT} kH: {kH} kW: {kW})"
            ),
        )

    torch._check(
        kT / 2 >= pT and kW / 2 >= pW and kH / 2 >= pH,
        lambda: (
            f"pad should be smaller than or equal to half of kernel size, but got "
            f"kT: {kT} kW: {kW} kH: {kH} padT: {pT} padW: {pW} padH: {pH}"
        ),
    )

    torch._check(
        otime >= 1 and owidth >= 1 and oheight >= 1,
        lambda: (
            f"Given input size: ({nslices}x{itime}x{iheight}x{iwidth}). "
            f"Calculated output size: ({nslices}x{otime}x{oheight}x{owidth}). "
            f"Output size is too small"
        ),
    )


def max_pool3d_backward_shape_check(
    input,
    grad_output,
    indices,
    nslices,
    kT,
    kH,
    kW,
    dT,
    dH,
    dW,
    pT,
    pH,
    pW,
    dilationT,
    dilationH,
    dilationW,
    itime,
    iheight,
    iwidth,
    otime,
    oheight,
    owidth,
    fn_name,
):
    ndim = input.ndim

    pool3d_shape_check(
        input,
        nslices,
        kT,
        kH,
        kW,
        dT,
        dH,
        dW,
        pT,
        pH,
        pW,
        dilationT,
        dilationH,
        dilationW,
        itime,
        iheight,
        iwidth,
        otime,
        oheight,
        owidth,
        fn_name,
    )

    check_dim_size(grad_output, ndim, ndim - 4, nslices)
    check_dim_size(grad_output, ndim, ndim - 3, otime)
    check_dim_size(grad_output, ndim, ndim - 2, oheight)
    check_dim_size(grad_output, ndim, ndim - 1, owidth)

    check_dim_size(indices, ndim, ndim - 4, nslices)
    check_dim_size(indices, ndim, ndim - 3, otime)
    check_dim_size(indices, ndim, ndim - 2, oheight)
    check_dim_size(indices, ndim, ndim - 1, owidth)


def avg_pool3d_backward_shape_check(
    input: Tensor,
    grad_output: Tensor,
    nslices: int,
    kT: int,
    kH: int,
    kW: int,
    dT: int,
    dH: int,
    dW: int,
    pT: int,
    pH: int,
    pW: int,
    itime: int,
    iheight: int,
    iwidth: int,
    otime: int,
    oheight: int,
    owidth: int,
    fn_name: str,
):
    ndim = input.ndim

    pool3d_shape_check(
        input,
        nslices,
        kT,
        kH,
        kW,
        dT,
        dH,
        dW,
        pT,
        pH,
        pW,
        1,
        1,
        1,
        itime,
        iheight,
        iwidth,
        otime,
        oheight,
        owidth,
        fn_name,
        True,
    )

    check_dim_size(grad_output, ndim, ndim - 4, nslices)
    check_dim_size(grad_output, ndim, ndim - 3, otime)
    check_dim_size(grad_output, ndim, ndim - 2, oheight)
    check_dim_size(grad_output, ndim, ndim - 1, owidth)


def max_pool2d_checks_and_compute_shape(
    input,
    kernel_size,
    stride,
    padding,
    dilation,
    ceil_mode,
):
    # Reference: aten/src/ATen/native/DilatedMaxPool2d.cpp
    def unpack(name, val):
        torch._check(
            len(val) in [1, 2],
            lambda: f"max_pool2d: {name} must either be a single int, or a tuple of two ints",
        )
        H = val[0]
        W = H if len(val) == 1 else val[1]
        return H, W

    kH, kW = unpack("kernel_size", kernel_size)

    torch._check(
        len(stride) in [0, 1, 2],
        lambda: "max_pool2d: stride must either be omitted, a single int, or a tuple of two ints",
    )
    if len(stride) == 0:
        dH, dW = kH, kW
    else:
        dH, dW = unpack("stride", stride)

    padH, padW = unpack("padding", padding)
    dilationH, dilationW = unpack("dilation", dilation)
    nInputPlane = input.size(-3)
    inputHeight = input.size(-2)
    inputWidth = input.size(-1)

    memory_format = utils.suggest_memory_format(input)
    if memory_format == torch.channels_last:
        torch._check(
            input.dim() == 4,
            lambda: "non-empty 4D (batch mode) tensor expected for input with channels_last layout",
        )
    elif memory_format == torch.contiguous_format:
        torch._check(
            input.dim() in [3, 4],
            lambda: "non-empty 3D or 4D (batch mode) tensor expected for input",
        )
    else:
        torch._check(
            False,
            lambda: "Unsupported memory format. Supports only ChannelsLast, Contiguous",
        )

    outputHeight = pooling_output_shape(inputHeight, kH, padH, dH, dilationH, ceil_mode)
    outputWidth = pooling_output_shape(inputWidth, kW, padW, dW, dilationW, ceil_mode)

    pool2d_shape_check(
        input,
        kH,
        kW,
        dH,
        dW,
        padH,
        padW,
        dilationH,
        dilationW,
        nInputPlane,
        inputHeight,
        inputWidth,
        outputHeight,
        outputWidth,
        memory_format,
    )

    return nInputPlane, outputHeight, outputWidth


@register_meta(aten.max_pool2d_with_indices_backward.default)
def meta_max_pool2d_with_indices_backward(
    grad_output,
    self,
    kernel_size,
    stride,
    padding,
    dilation,
    ceil_mode,
    indices,
):
    (
        nInputPlane,
        outputHeight,
        outputWidth,
    ) = max_pool2d_checks_and_compute_shape(
        self, kernel_size, stride, padding, dilation, ceil_mode
    )

    torch._check(
        self.dtype == grad_output.dtype,
        lambda: f"Expected dtype {self.dtype} for `gradOutput` but got dtype {grad_output.dtype}",
    )

    nOutputPlane = nInputPlane
    ndim = self.ndim

    def _check_dim_size(t):
        check_dim_size(t, ndim, ndim - 3, nOutputPlane)
        check_dim_size(t, ndim, ndim - 2, outputHeight)
        check_dim_size(t, ndim, ndim - 1, outputWidth)

    _check_dim_size(grad_output)
    _check_dim_size(indices)

    memory_format = utils.suggest_memory_format(self)
    return torch.empty(
        self.shape,
        dtype=self.dtype,
        device=self.device,
        memory_format=memory_format,
    )


@register_meta(aten.max_pool2d_with_indices.default)
def meta_max_pool2d_with_indices(
    input,
    kernel_size,
    stride=(),
    padding=(0,),
    dilation=(1,),
    ceil_mode=False,
):
    (
        nInputPlane,
        outputHeight,
        outputWidth,
    ) = max_pool2d_checks_and_compute_shape(
        input, kernel_size, stride, padding, dilation, ceil_mode
    )

    nbatch = input.size(-4) if input.dim() == 4 else 1
    memory_format = utils.suggest_memory_format(input)
    if input.dim() == 3:
        size = [nInputPlane, outputHeight, outputWidth]
    else:
        size = [nbatch, nInputPlane, outputHeight, outputWidth]
    return (
        torch.empty(
            size,
            dtype=input.dtype,
            device=input.device,
            memory_format=memory_format,
        ),
        torch.empty(
            size,
            dtype=torch.int64,
            device=input.device,
            memory_format=memory_format,
        ),
    )


@register_meta(aten.fractional_max_pool2d.default)
def meta_fractional_max_pool2d(self, kernel_size, output_size, random_samples):
    torch._check(
        self.ndim in (3, 4),
        lambda: f"fractional_max_pool2d: Expected 3D or 4D tensor, but got: {self.ndim}",
    )
    ndim = self.ndim

    for d in range(ndim - 3, ndim):
        torch._check(
            self.size(d) > 0,
            f"fractional_max_pool2d: Expected input to have non-zero "
            f" size for non-batch dimensions, but got {self.size()} with dimension {d} empty",
        )

    # the check and message are out of sync, but this matches the structured meta
    torch._check(
        len(kernel_size) == 2,
        lambda: "fractional_max_pool2d: kernel_size must"
        "either be a single int or tuple of Ints",
    )
    torch._check(
        len(output_size) == 2,
        lambda: "fractional_max_pool2d: output_size must "
        "either be a single int or tuple of Ints",
    )

    input_channels = self.size(-3)
    input_height = self.size(-2)
    input_width = self.size(-1)
    if ndim == 4:
        input_batch = self.size(0)
    else:
        input_batch = 1

    torch._check(
        self.dtype == random_samples.dtype,
        lambda: "Expect _random_samples to have the same dtype as input",
    )
    torch._check(
        random_samples.ndim == 3,
        lambda: f"Expect _random samples to have 3 dimensions got, {random_samples.ndim}",
    )

    n = random_samples.size(0)
    c = random_samples.size(1)
    d = random_samples.size(2)
    torch._check(
        n >= input_batch,
        "Expect _random_samples.size(0) no less then input batch size.",
    )
    torch._check(
        c == input_channels,
        lambda: "Expect _random_samples.size(1) equals to input channel size.",
    )
    torch._check(d == 2, lambda: f"Expect _random_samples.size(2) equals to 2 got {d}.")

    torch._check(
        output_size[0] + kernel_size[0] - 1 <= input_height,
        lambda: f"fractional_max_pool2d: kernel height {kernel_size[0]} is too large relative to input height {input_height}",
    )
    torch._check(
        output_size[1] + kernel_size[1] - 1 <= input_width,
        lambda: f"fractional_max_pool2d: kernel width {kernel_size[1]} is too large relative to input width {input_width}",
    )

    if self.dim() == 4:
        size = [input_batch, input_channels, output_size[0], output_size[1]]
    else:
        size = [input_channels, output_size[0], output_size[1]]

    return (
        torch.empty(
            size,
            dtype=self.dtype,
            device=self.device,
        ),
        torch.empty(
            size,
            dtype=torch.int64,
            device=self.device,
        ),
    )


@register_meta(aten.max_pool3d_with_indices)
@out_wrapper("out", "indices")
def meta_max_pool3d_with_indices(
    input,
    kernel_size,
    stride=(),
    padding=(0,),
    dilation=(1,),
    ceil_mode=False,
):
    torch._check(
        len(kernel_size) in (1, 3),
        lambda: "max_pool3d: kernel_size must either be a single int, or a tuple of three ints",
    )
    kT = kernel_size[0]
    kH = kT if len(kernel_size) == 1 else kernel_size[1]
    kW = kT if len(kernel_size) == 1 else kernel_size[2]

    torch._check(
        not stride or len(stride) in (1, 3),
        lambda: "max_pool3d: stride must either be omitted, a single int, or a tuple of three ints",
    )
    dT = kT if not stride else stride[0]
    dH = kH if not stride else (dT if len(stride) == 1 else stride[1])
    dW = kW if not stride else (dT if len(stride) == 1 else stride[2])

    torch._check(
        len(padding) in (1, 3),
        lambda: "max_pool3d: padding must either be a single int, or a tuple of three ints",
    )
    pT = padding[0]
    pH = pT if len(padding) == 1 else padding[1]
    pW = pT if len(padding) == 1 else padding[2]

    torch._check(
        len(dilation) in (1, 3),
        lambda: "max_pool3d: dilation must be either a single int, or a tuple of three ints",
    )
    dilationT = dilation[0]
    dilationH = dilationT if len(dilation) == 1 else dilation[1]
    dilationW = dilationT if len(dilation) == 1 else dilation[2]

    torch._check(
        input.ndim in (4, 5),
        lambda: "non-empty 4D or 5D (batch mode) tensor expected for input",
    )

    nbatch = input.size(-5) if input.ndim == 5 else 1
    nslices = input.size(-4)
    itime = input.size(-3)
    iheight = input.size(-2)
    iwidth = input.size(-1)

    otime = pooling_output_shape(itime, kT, pT, dT, dilationT, ceil_mode)
    oheight = pooling_output_shape(iheight, kH, pH, dH, dilationH, ceil_mode)
    owidth = pooling_output_shape(iwidth, kW, pW, dW, dilationW, ceil_mode)

    pool3d_shape_check(
        input,
        nslices,
        kT,
        kH,
        kW,
        dT,
        dH,
        dW,
        pT,
        pH,
        pW,
        dilationT,
        dilationH,
        dilationW,
        itime,
        iheight,
        iwidth,
        otime,
        oheight,
        owidth,
        "max_pool3d_with_indices()",
    )

    channels_last = (
        input.ndim == 5 and utils.suggest_memory_format(input) == torch.channels_last_3d
    )
    if input.ndim == 4:
        input_channels_last_check = input.unsqueeze(0)
        channels_last = (
            not input_channels_last_check.is_contiguous()
        ) and input_channels_last_check.is_contiguous(
            memory_format=torch.channels_last_3d
        )
        out_shape = (nslices, otime, oheight, owidth)
    else:
        out_shape = (nbatch, nslices, otime, oheight, owidth)  # type: ignore[assignment]

    out = input.new_empty(out_shape)
    indices = input.new_empty(out_shape, dtype=torch.int64)

    if channels_last:
        out = out.to(memory_format=torch.channels_last_3d)
        indices = indices.to(memory_format=torch.channels_last_3d)

    return out, indices


@register_meta(aten.max_pool3d_with_indices_backward)
@out_wrapper("grad_input")
def meta_max_pool3d_with_indices_backward(
    grad_output,
    input,
    kernel_size,
    stride,
    padding,
    dilation,
    ceil_mode,
    indices,
):
    torch._check(
        len(kernel_size) in (1, 3),
        lambda: "max_pool3d: kernel_size must either be a single int, or a tuple of three ints",
    )
    kT = kernel_size[0]
    kH = kT if len(kernel_size) == 1 else kernel_size[1]
    kW = kT if len(kernel_size) == 1 else kernel_size[2]

    torch._check(
        not stride or len(stride) in (1, 3),
        lambda: "max_pool3d: stride must either be omitted, a single int, or a tuple of three ints",
    )
    dT = kT if not stride else stride[0]
    dH = kH if not stride else (dT if len(stride) == 1 else stride[1])
    dW = kW if not stride else (dT if len(stride) == 1 else stride[2])

    torch._check(
        len(padding) in (1, 3),
        lambda: "max_pool3d: padding must either be a single int, or a tuple of three ints",
    )
    pT = padding[0]
    pH = pT if len(padding) == 1 else padding[1]
    pW = pT if len(padding) == 1 else padding[2]

    torch._check(
        len(dilation) in (1, 3),
        lambda: "max_pool3d: dilation must be either a single int, or a tuple of three ints",
    )
    dilationT = dilation[0]
    dilationH = dilationT if len(dilation) == 1 else dilation[1]
    dilationW = dilationT if len(dilation) == 1 else dilation[2]

    torch._check(
        input.ndim in (4, 5),
        lambda: "non-empty 4D or 5D (batch mode) tensor expected for input",
    )

    nslices = input.size(-4)
    itime = input.size(-3)
    iheight = input.size(-2)
    iwidth = input.size(-1)

    otime = grad_output.size(-3)
    oheight = grad_output.size(-2)
    owidth = grad_output.size(-1)

    max_pool3d_backward_shape_check(
        input,
        grad_output,
        indices,
        nslices,
        kT,
        kH,
        kW,
        dT,
        dH,
        dW,
        pT,
        pH,
        pW,
        dilationT,
        dilationH,
        dilationW,
        itime,
        iheight,
        iwidth,
        otime,
        oheight,
        owidth,
        "max_pool3d_with_indices_backward()",
    )

    channels_last = (
        input.ndim == 5 and utils.suggest_memory_format(input) == torch.channels_last_3d
    )
    if input.ndim == 4:
        input_channels_last_check = input.unsqueeze(0)
        channels_last = (
            not input_channels_last_check.is_contiguous()
        ) and input_channels_last_check.is_contiguous(
            memory_format=torch.channels_last_3d
        )

    grad_input = input.new_empty(input.shape)

    if channels_last:
        grad_input = grad_input.to(memory_format=torch.channels_last_3d)

    return grad_input


def check_grid_sampler_common(input: Tensor, grid: Tensor):
    torch._check(
        input.device == grid.device,
        lambda: (
            f"grid_sampler(): expected input and grid to be on same device, but input "
            f"is on {input.device} and grid is on {grid.device}"
        ),
    )
    torch._check(
        input.layout == torch.strided and grid.layout == torch.strided,
        lambda: (
            f"grid_sampler(): expected input and grid to have torch.strided layout, but "
            f"input has {input.layout} and grid has {grid.layout}"
        ),
    )
    torch._check(
        input.shape[0] == grid.shape[0],
        lambda: (
            f"grid_sampler(): expected grid and input to have same batch size, but got "
            f"input with sizes {input.shape} and grid with sizes {grid.shape}"
        ),
    )
    torch._check(
        grid.shape[-1] == input.ndim - 2,
        lambda: (
            f"grid_sampler(): expected grid to have size {input.ndim - 2} in last "
            f"dimension, but got grid with sizes {grid.shape}"
        ),
    )

    for i in range(2, input.ndim):
        torch._check(
            input.shape[i] > 0,
            lambda: (
                f"grid_sampler(): expected input to have non-empty spatial dimensions, "
                f"but input has sizes {input.shape} with dimension {i} being empty"
            ),
        )


class GridSamplerInterpolation(Enum):
    BILINEAR = 0
    NEAREST = 1
    BICUBIC = 2


def check_grid_sampler_3d(input: Tensor, grid: Tensor, interpolation_mode: int):
    torch._check(
        input.ndim == 5 and input.ndim == grid.ndim,
        lambda: (
            f"grid_sampler(): expected 5D input and grid with same number of "
            f"dimensions, but got input with sizes {input.shape}"
            f" and grid with sizes {grid.shape}"
        ),
    )
    torch._check(
        not (
            input.ndim == 5
            and interpolation_mode == GridSamplerInterpolation.BICUBIC.value
        ),
        lambda: "grid_sampler(): bicubic interpolation only supports 4D input",
    )


@register_meta(aten.grid_sampler_2d_backward.default)
def grid_sampler_2d_backward_meta(
    grad_output,
    input,
    grid,
    interpolation_mode,
    padding_mode,
    align_corners,
    output_mask,
):
    input_requires_grad = output_mask[0]
    if input_requires_grad:
        grad_input = torch.zeros_like(input, memory_format=torch.contiguous_format)
    else:
        grad_input = None
    grad_grid = torch.empty_like(grid, memory_format=torch.contiguous_format)
    return (grad_input, grad_grid)


@register_meta(aten.grid_sampler_3d)
@out_wrapper()
def grid_sampler_3d(
    input,
    grid,
    interpolation_mode,
    padding_mode,
    align_corners,
):
    check_grid_sampler_common(input, grid)
    check_grid_sampler_3d(input, grid, interpolation_mode)
    N = input.shape[0]
    C = input.shape[1]
    out_D = grid.shape[1]
    out_H = grid.shape[2]
    out_W = grid.shape[3]
    return input.new_empty((N, C, out_D, out_H, out_W))


@register_meta(aten.grid_sampler_3d_backward)
@out_wrapper("grad_input", "grad_grid")
def grid_sampler_3d_backward(
    grad_output,
    input,
    grid,
    interpolation_mode,
    padding_mode,
    align_corners,
    output_mask,
):
    check_grid_sampler_common(input, grid)
    check_grid_sampler_3d(input, grid, interpolation_mode)
    input_requires_grad = output_mask[0]
    if input_requires_grad:
        grad_input = torch.zeros_like(
            input, memory_format=torch.legacy_contiguous_format
        )
    else:
        grad_input = None
    grad_grid = torch.empty_like(grid, memory_format=torch.legacy_contiguous_format)
    return grad_input, grad_grid


@register_meta([aten.full.default])
def full(size, fill_value, *args, **kwargs):
    dtype = kwargs.get("dtype", None)
    if not dtype:
        dtype = utils.get_dtype(fill_value)
    kwargs["dtype"] = dtype
    return torch.empty(size, *args, **kwargs)


# zeros_like is special cased to work for sparse
@register_meta(aten.zeros_like.default)
def zeros_like(
    self,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
    memory_format=None,
):
    if layout == torch.sparse_coo:
        torch._check(
            memory_format is None,
            lambda: "memory format option is only supported by strided tensors",
        )

        res = torch.empty(
            0,
            dtype=self.dtype if dtype is None else dtype,
            layout=layout,
            device=self.device if device is None else device,
            pin_memory=pin_memory,
        )

        if self.is_sparse:
            res.sparse_resize_and_clear_(
                self.size(), self.sparse_dim(), self.dense_dim()
            )
        else:
            res.sparse_resize_and_clear_(self.size(), self.dim(), 0)

        res._coalesced_(True)
        return res
    res = aten.empty_like.default(
        self,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
        memory_format=memory_format,
    )
    # device can be not "meta"
    res.fill_(0)
    return res


@register_meta([aten.ones.default, aten.ones.out])
@out_wrapper()
def meta_ones(
    size,
    *,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
    requires_grad=False,
):
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.get_default_device()
    if layout is None:
        layout = torch.strided
    return torch.empty(
        size, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory
    )


@register_meta([aten.zeros.default, aten.zeros.out])
@out_wrapper()
def meta_zeros(
    size,
    *,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
    requires_grad=False,
):
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.get_default_device()
    if layout is None:
        layout = torch.strided
    return torch.empty(
        size, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory
    )


@register_meta(aten.select.int)
def meta_select(self, dim, index):
    from torch.fx.experimental.symbolic_shapes import guard_size_oblivious

    ndim = self.dim()
    torch._check_index(
        ndim != 0,
        lambda: "select() cannot be applied to a 0-dim tensor.",
    )

    dim = dim if dim >= 0 else dim + ndim
    size = self.size(dim)

    torch._check_index(
        not (
            guard_size_oblivious(-index > size) or guard_size_oblivious(index >= size)
        ),
        lambda: f"select(): index {index} out of range for tensor of size "
        f"{self.size()} at dimension {dim}",
    )

    index = index if index >= 0 else index + size

    new_size = list(self.size())
    new_stride = list(self.stride())

    new_storage_offset = self.storage_offset() + index * new_stride[dim]
    del new_size[dim]
    del new_stride[dim]

    return self.as_strided(new_size, new_stride, new_storage_offset)


@register_meta(aten.select_scatter.default)
def meta_select_scatter(self, src, dim, index):
    return utils.clone_preserve_strides(self)


@register_meta(aten.slice_scatter.default)
def meta_slice_scatter(self, src, dim=0, start=None, end=None, step=1):
    return utils.clone_preserve_strides(self)


# TODO: Deduplicate this with canonicalize_dim
def maybe_wrap_dim(dim: int, dim_post_expr: int, wrap_scalar: bool = True):
    if dim_post_expr <= 0:
        assert wrap_scalar
        dim_post_expr = 1
    min = -dim_post_expr
    max = dim_post_expr - 1
    assert not (dim < min or dim > max), f"dim {dim} out of bounds ({min}, {max})"
    if dim < 0:
        dim += dim_post_expr
    return dim


def ensure_nonempty_size(t, dim):
    return 1 if t.dim() == 0 else t.shape[dim]


# From aten/src/ATen/native/ScatterGatherChecks.h
def gather_shape_check(self, dim, index):
    self_dims = max(self.dim(), 1)
    index_dims = max(index.dim(), 1)
    torch._check(
        self_dims == index_dims,
        lambda: "Index tensor must have the same number of dimensions as input tensor",
    )
    for i in range(self_dims):
        if i != dim:
            torch._check(
                ensure_nonempty_size(index, i) <= ensure_nonempty_size(self, i),
                lambda: f"Size does not match at dimension {i} expected index {index.shape}"
                + f" to be no larger than self {self.shape} apart from dimension {dim}",
            )


@register_meta(aten.gather.default)
def meta_gather(self, dim, index, sparse_grad=False):
    from torch.fx.experimental.symbolic_shapes import guard_or_false

    wrapped_dim = maybe_wrap_dim(dim, self.dim())
    is_index_empty = guard_or_false(index.numel() == 0)
    if not is_index_empty:
        torch._check(
            index.dtype == torch.long or index.dtype == torch.int,
            lambda: f"gather(): Expected dtype int32/int64 for index, but got {index.dtype}",
        )
        gather_shape_check(self, wrapped_dim, index)
    return self.new_empty(index.shape)


# From aten/src/ATen/native/TensorAdvancedIndexing.cpp
def get_operator_enum(reduce_, use_new_options=False):
    if use_new_options:
        if reduce_ == "sum":
            return "REDUCE_ADD"
        elif reduce_ == "prod":
            return "REDUCE_MULTIPLY"
        elif reduce_ == "mean":
            return "REDUCE_MEAN"
        elif reduce_ == "amax":
            return "REDUCE_MAXIMUM"
        elif reduce_ == "amin":
            return "REDUCE_MINIMUM"
        torch._check(
            False,
            lambda: "reduce argument must be either sum, prod, mean, amax or amin.",
        )
        return
    else:
        if reduce_ == "add":
            return "REDUCE_ADD"
        elif reduce_ == "multiply":
            return "REDUCE_MULTIPLY"
        torch._check(False, lambda: "reduce argument must be either add or multiply.")
        return


# From aten/src/ATen/native/ScatterGatherChecks.h
def scatter_gather_dtype_check(method_name, self, index, src_opt=None):
    from torch.fx.experimental.symbolic_shapes import guard_or_true

    if guard_or_true(index.numel() != 0):
        torch._check(
            index.dtype == torch.long or index.dtype == torch.int,
            lambda: f"{method_name}(): Expected dtype int32/int64 for index",
        )

    if src_opt is not None:
        torch._check(
            self.dtype == src_opt.dtype,
            lambda: f"{method_name}(): Expected self.dtype to be equal to src.dtype",
        )


def ensure_nonempty_dim(dim):
    return max(dim, 1)


# From aten/src/ATen/native/ScatterGatherChecks.h
def scatter_shape_check(self, dim, index, src_opt=None):
    from torch.fx.experimental.symbolic_shapes import guard_or_false

    if guard_or_false(index.numel() == 0):
        return
    torch._check(
        ensure_nonempty_dim(self.dim()) == ensure_nonempty_dim(index.dim()),
        lambda: "Index tensor must have the same number of dimensions as self tensor",
    )

    is_wrong_shape = False
    self_dims = ensure_nonempty_dim(self.dim())

    # Check: index.size(d) <= self.size(d) for all d != dim
    for d in range(self_dims):
        index_d_size = ensure_nonempty_size(index, d)
        if d == dim:
            continue
        if index_d_size > ensure_nonempty_size(self, d):
            is_wrong_shape = True
            break

    # Check: index.size(d) <= src.size(d) for all d if src is Tensor
    if not is_wrong_shape and src_opt is not None:
        for d in range(self_dims):
            index_d_size = ensure_nonempty_size(index, d)
            if index_d_size > ensure_nonempty_size(src_opt, d):
                is_wrong_shape = True
                break

    if src_opt is not None:
        torch._check(
            ensure_nonempty_dim(self.dim()) == ensure_nonempty_dim(index.dim()),
            lambda: "Index tensor must have the same number of dimensions as self tensor",
        )
        torch._check(
            not is_wrong_shape,
            lambda: f"Expected index {index.shape} to be no larger than self {self.shape}"
            + f" apart from dimension {dim} and to be no larger than src {src_opt.shape}",
        )
    else:
        torch._check(
            not is_wrong_shape,
            lambda: f"Expected index {index.shape} to be no larger than self {self.shape}"
            + f" apart from dimension {dim}",
        )


# From aten/src/ATen/native/TensorAdvancedIndexing.cpp
def scatter_meta_impl(self, dim, index, src=None, reduce_=None, use_new_options=False):
    wrapped_dim = maybe_wrap_dim(dim, self.dim())
    scatter_gather_dtype_check("scatter", self, index, src)
    scatter_shape_check(self, wrapped_dim, index, src)
    if reduce_ is not None:
        # Check if we have a valid reduce operator.
        get_operator_enum(reduce_, use_new_options)


@register_meta(aten.scatter_add.default)
def meta_scatter_add(self, dim, index, src):
    scatter_meta_impl(self, dim, index, src, "add")
    return self.new_empty(self.shape)


@register_meta(aten.scatter_add_)
def meta_scatter_add_(self, dim, index, src):
    scatter_meta_impl(self, dim, index, src, "add")
    return self


@register_meta(
    [
        aten.scatter.src,
        aten.scatter.value,
        aten.scatter.reduce,
        aten.scatter.value_reduce,
    ]
)
@out_wrapper()
def meta_scatter(self, dim, index, src_or_value, reduce=None):
    src = src_or_value if isinstance(src_or_value, torch.Tensor) else None
    scatter_meta_impl(self, dim, index, src, reduce)
    return self.new_empty(self.shape)


@register_meta(
    [
        aten.scatter_.src,
        aten.scatter_.value,
        aten.scatter_.reduce,
        aten.scatter_.value_reduce,
    ]
)
def meta_scatter_(self, dim, index, src_or_value, reduce=None):
    src = src_or_value if isinstance(src_or_value, torch.Tensor) else None
    scatter_meta_impl(self, dim, index, src, reduce)
    return self


@register_meta([aten._scaled_dot_product_flash_attention])
def meta__scaled_dot_product_flash_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    return_debug_mask: bool = False,
    scale: Optional[float] = None,
):
    batch_size = query.size(0)
    num_heads = query.size(1)
    max_seqlen_batch_q = query.size(2)
    head_dim = query.size(3)
    max_seqlen_batch_k = key.size(2)

    query_t = query.transpose(1, 2)
    attention = torch.empty_like(query_t).transpose(1, 2)
    logsumexp = torch.empty(
        (batch_size, num_heads, max_seqlen_batch_q),
        dtype=torch.float,
        device=query.device,
    )

    if return_debug_mask:
        blocksize_c = 128 if head_dim > 64 else 256
        max_seqlen_k = math.ceil(max_seqlen_batch_q / blocksize_c)
        if max_seqlen_batch_k <= 128:
            max_seqlen_k = 128
        elif max_seqlen_batch_k <= 256:
            max_seqlen_k = 256
        debug_mask = torch.empty(
            (batch_size, num_heads, max_seqlen_batch_q, max_seqlen_k),
            dtype=query.dtype,
            device=query.device,
        )
    else:
        debug_mask = torch.empty(0, dtype=query.dtype, device=query.device)

    # Note [Seed and Offset]: device for seed and offset below depends on whether we are
    # capturing or not, but at the time of tracing we don't know if we
    # are going to use cudagraphs or not, so we return meta tensors here
    # it's possible we'll need to have some special handling in inductor for sdpa
    # See [Note] BC breaking change to flash seed/offset
    if torch.version.hip and torch.cuda.is_available():
        # Maintain old path on AMD
        seed = torch.empty((), dtype=torch.long, device="meta")
        offset = torch.empty((), dtype=torch.long, device="meta")
    else:
        seed = torch.empty((2), dtype=torch.uint64, device="meta")
        offset = torch.empty((), dtype=torch.uint64, device="meta")

    return (
        attention,
        logsumexp,
        None,
        None,
        max_seqlen_batch_q,
        max_seqlen_batch_k,
        seed,
        offset,
        debug_mask,
    )


def alloc_with_matching_layout(
    query: Tensor,
    res_shape: tuple[int, ...],
):
    if tuple(query.shape) == res_shape:
        query_t = query.transpose(1, 2)
        res = torch.empty_like(query_t).transpose(1, 2)
    else:
        dim_order = sorted(
            [0, 1, 2, 3], key=lambda idx: query.stride()[idx], reverse=True
        )
        permuted_shape = [res_shape[idx] for idx in dim_order]
        final_permute = [dim_order.index(i) for i in range(len(dim_order))]
        res = torch.empty(
            permuted_shape, dtype=query.dtype, device=query.device
        ).permute(final_permute)

    return res


@register_meta([aten._scaled_dot_product_cudnn_attention])
def meta__scaled_dot_product_cudnn_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_bias: Optional[Tensor],
    compute_log_sumexp: bool,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    return_debug_mask: bool = False,
    scale: Optional[float] = None,
):
    B = query.size(0)
    H = query.size(1)
    S_Q = query.size(2)
    S_KV = key.size(2)
    D_V = value.size(-1)

    res_shape = (B, H, S_Q, D_V)
    res = alloc_with_matching_layout(query, res_shape)

    logsum_exp = torch.empty(
        (B, H, S_Q),
        dtype=torch.float,
        device=query.device,
    )

    # See Note [Seed and Offset]
    seed = torch.empty((), dtype=torch.long, device="meta")
    offset = torch.empty((), dtype=torch.long, device="meta")

    return (
        res,
        logsum_exp,
        None,
        None,
        S_Q,
        S_KV,
        seed,
        offset,
        None,
    )


@register_meta([aten._scaled_dot_product_fused_attention_overrideable])
def meta__scaled_dot_product_fused_attention_overrideable(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_bias: Optional[Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    return_debug_mask: bool = False,
    scale: Optional[float] = None,
):
    B = query.size(0)
    H_Q = query.size(1)
    S_Q = query.size(2)
    S_KV = key.size(2)
    D_V = value.size(-1)

    res_shape = (B, H_Q, S_Q, D_V)
    res = alloc_with_matching_layout(query, res_shape)

    logsum_exp = torch.empty(
        (B, H_Q, S_Q),
        dtype=torch.float,
        device=query.device,
    )

    # See Note [Seed and Offset]
    seed = torch.empty((), dtype=torch.long, device="meta")
    offset = torch.empty((), dtype=torch.long, device="meta")

    return (
        res,
        logsum_exp,
        None,
        None,
        S_Q,
        S_KV,
        seed,
        offset,
        None,
    )


@register_meta(
    [
        aten._scaled_dot_product_flash_attention_backward,
    ]
)
def meta__scaled_dot_product_flash_backward(
    grad_out: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    out: Tensor,
    logsumexp: Tensor,
    cum_seq_q: Tensor,
    cum_seq_k: Tensor,
    max_q: int,
    max_k: int,
    dropout_p: float,
    is_causal: bool,
    philox_seed: Tensor,
    philox_offset: Tensor,
    scale: Optional[float] = None,
):
    grad_q = torch.empty_like(query.transpose(1, 2)).transpose(1, 2)
    grad_k = torch.empty_like(key.transpose(1, 2)).transpose(1, 2)
    grad_v = torch.empty_like(value.transpose(1, 2)).transpose(1, 2)
    return grad_q, grad_k, grad_v


@register_meta(
    [
        aten._scaled_dot_product_flash_attention_for_cpu,
    ]
)
def meta__scaled_dot_product_flash_attention_for_cpu(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    attn_mask: Optional[Tensor] = None,
    scale: Optional[float] = None,
):
    batch_size = query.size(0)
    num_heads = query.size(1)
    max_seqlen_batch_q = query.size(2)

    attention = torch.empty_like(query)
    logsumexp = torch.empty(
        (
            batch_size,
            max_seqlen_batch_q,
            num_heads,
        ),
        dtype=torch.float,
        device=query.device,
    ).transpose(1, 2)
    return (
        attention,
        logsumexp,
    )


@register_meta(
    [
        aten._scaled_dot_product_flash_attention_for_cpu_backward,
    ]
)
def meta__scaled_dot_product_flash_attention_for_cpu_backward(
    grad_out: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    out: Tensor,
    logsumexp: Tensor,
    dropout_p: float,
    is_causal: bool,
    attn_mask: Optional[Tensor] = None,
    scale: Optional[float] = None,
):
    # cpus's grad layout is different from cuda's,
    # i.e. (batch_size, seq_len, num_heads, head_dim)
    grad_q = torch.empty_permuted(
        query.size(),
        (0, 2, 1, 3),
        dtype=query.dtype,
        device=query.device,
    )
    grad_k = torch.empty_permuted(
        key.size(),
        (0, 2, 1, 3),
        dtype=key.dtype,
        device=key.device,
    )
    grad_v = torch.empty_permuted(
        value.size(),
        (0, 2, 1, 3),
        dtype=value.dtype,
        device=value.device,
    )

    return grad_q, grad_k, grad_v


@register_meta([aten._scaled_dot_product_efficient_attention])
def meta__scaled_dot_product_efficient_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_bias: Optional[Tensor],
    compute_log_sumexp: bool,
    dropout_p=0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
):
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    B = query.size(0)
    M = query.size(1)
    num_heads = query.size(-2)
    Kv = value.size(-1)

    res = torch.empty(B, M, num_heads, Kv, dtype=query.dtype, device=query.device)

    if torch.version.hip and torch.cuda.is_available():
        """Please see: https://github.com/pytorch/pytorch/issues/146848
        longsumexp last dim should be seq length
        """
        logsumexp_dim = M if compute_log_sumexp else 0
    else:
        logsumexp_dim = math.ceil(M / 32) * 32 if compute_log_sumexp else 0

    logsum_exp = torch.empty(
        (B, num_heads, logsumexp_dim),
        dtype=torch.float,
        device=query.device,
    )

    res = res.transpose(1, 2)

    # See Note [Seed and Offset]:
    seed = torch.empty((), dtype=torch.long, device="meta")
    offset = torch.empty((), dtype=torch.long, device="meta")

    return res, logsum_exp, seed, offset


@register_meta(
    [
        aten._scaled_dot_product_efficient_attention_backward,
    ]
)
def meta__scaled_dot_product_efficient_backward(
    grad_out: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_bias: Optional[Tensor],
    out: Tensor,
    logsumexp: Tensor,
    philox_seed: Tensor,
    philox_offset: Tensor,
    dropout_p: float,
    grad_input_mask: list[bool],
    is_causal: bool = False,
    scale: Optional[float] = None,
):
    batch_size = query.size(0)
    num_heads = query.size(1)
    max_q = query.size(2)
    head_dim = query.size(3)
    head_dim_v = value.size(3)

    max_k = key.size(2)

    grad_q = torch.empty_permuted(
        (batch_size, num_heads, max_q, head_dim),
        (0, 2, 1, 3),
        dtype=query.dtype,
        device=query.device,
    )
    grad_k = torch.empty_permuted(
        (batch_size, num_heads, max_k, head_dim),
        (0, 2, 1, 3),
        dtype=key.dtype,
        device=key.device,
    )
    grad_v = torch.empty_permuted(
        (batch_size, num_heads, max_k, head_dim_v),
        (0, 2, 1, 3),
        dtype=value.dtype,
        device=value.device,
    )
    grad_bias = None
    if attn_bias is not None and grad_input_mask[3]:
        lastDim = attn_bias.size(-1)
        lastDimAligned = lastDim if lastDim % 16 == 0 else lastDim + 16 - lastDim % 16
        new_sizes = list(attn_bias.size())
        new_sizes[-1] = lastDimAligned
        grad_bias = torch.empty(
            new_sizes, dtype=attn_bias.dtype, device=attn_bias.device
        )
        grad_bias = grad_bias[..., :lastDim]

    return grad_q, grad_k, grad_v, grad_bias


@register_meta(
    [
        aten._scaled_dot_product_cudnn_attention_backward,
    ]
)
def meta__scaled_dot_product_cudnn_backward(
    grad_out: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    out: Tensor,
    logsumexp: Tensor,
    philox_seed: Tensor,
    philox_offset: Tensor,
    attn_bias: Tensor,
    cum_seq_q: Tensor,
    cum_seq_k: Tensor,
    max_q: int,
    max_k: int,
    dropout_p: float,
    is_causal: bool,
    scale: Optional[float] = None,
):
    grad_q = torch.empty_like(query)
    grad_k = torch.empty_like(key)
    grad_v = torch.empty_like(value)
    return grad_q, grad_k, grad_v


@register_meta(
    [
        aten._flash_attention_forward,
    ]
)
def meta__flash_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    cum_seq_q: Optional[Tensor],
    cum_seq_k: Optional[Tensor],
    max_q: int,
    max_k: int,
    dropout_p: float,
    is_causal: bool,
    return_debug_mask: bool,
    scale: Optional[float] = None,
    window_size_left: Optional[int] = None,
    window_size_right: Optional[int] = None,
    seqused_k: Optional[Tensor] = None,
    alibi_slopes: Optional[Tensor] = None,
):
    # NB: there are two underlying paths:
    # 1. normal dense path; expect 4D inputs of shape (batch_size, seqlen, num_heads, head_dim)
    # 2. varseqlen path; expect 3D inputs of shape (total, num_heads, head_dim) where total
    #    includes all batch item sequences. cum_seq_q / cum_seq_k contain offsets into total
    batch_size = query.size(0) if cum_seq_q is None else cum_seq_q.numel() - 1
    max_seqlen_batch_q = query.size(1) if cum_seq_q is None else max_q
    max_seqlen_batch_k = key.size(1) if cum_seq_k is None else max_k
    num_heads = query.size(-2)
    head_dim = query.size(-1)

    # Cuda Path
    attention = torch.empty_like(query)
    if cum_seq_q is None:
        logsumexp = torch.empty(
            (batch_size, num_heads, max_seqlen_batch_q),
            dtype=torch.float,
            device=query.device,
        )
    else:
        total_q = query.size(0)
        logsumexp = torch.empty(
            (num_heads, total_q), dtype=torch.float, device=query.device
        )

    if return_debug_mask:
        blocksize_c = 128 if head_dim > 64 else 256
        max_seqlen_k = math.ceil(max_seqlen_batch_q / blocksize_c)
        if max_seqlen_batch_k <= 128:
            max_seqlen_k = 128
        elif max_seqlen_batch_k <= 256:
            max_seqlen_k = 256
        debug_mask = torch.empty(
            (batch_size, num_heads, max_seqlen_batch_q, max_seqlen_k),
            dtype=query.dtype,
            device=query.device,
        )
    else:
        debug_mask = torch.empty(0, dtype=query.dtype, device=query.device)

    # See Note [Seed and Offset]
    # See [Note] BC breaking change to flash seed/offset
    seed, offset = None, None
    if torch.version.hip and torch.cuda.is_available():
        # Maintain old path on AMD
        seed = torch.empty((), dtype=torch.long, device="meta")
        offset = torch.empty((), dtype=torch.long, device="meta")
    else:
        seed = torch.empty((2), dtype=torch.uint64, device="meta")
        offset = torch.empty((), dtype=torch.uint64, device="meta")
    return (
        attention,
        logsumexp,
        seed,
        offset,
        debug_mask,
    )


@register_meta(
    [
        aten._flash_attention_backward,
    ]
)
def meta__flash_attention_backward(
    grad_out: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    out: Tensor,
    logsumexp: Tensor,
    cum_seq_q: Tensor,
    cum_seq_k: Tensor,
    max_q: int,
    max_k: int,
    dropout_p: float,
    is_causal: bool,
    philox_seed: Tensor,
    philox_offset: Tensor,
    scale: Optional[float] = None,
    window_size_left: Optional[int] = None,
    window_size_right: Optional[int] = None,
):
    grad_query = torch.empty_like(query)
    grad_key = torch.empty_like(key)
    grad_value = torch.empty_like(value)

    return grad_query, grad_key, grad_value


@register_meta(
    [
        aten._efficient_attention_forward,
    ]
)
def meta__efficient_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    bias: Optional[Tensor],
    cu_seqlens_q: Optional[Tensor],
    cu_seqlens_k: Optional[Tensor],
    max_seqlen_q: Optional[int],
    max_seqlen_k: Optional[int],
    dropout_p: float,
    custom_mask_type: int,
    compute_log_sumexp: bool = False,
    scale: Optional[float] = None,
    causal_diagonal: Optional[Tensor] = None,
    seqlen_k: Optional[Tensor] = None,
    window_size: Optional[int] = None,
):
    B = query.size(0)
    M = query.size(1)
    N = key.size(1)
    num_heads = query.size(-2)
    Kv = value.size(-1)

    res = torch.empty(B, M, num_heads, Kv, dtype=query.dtype, device=query.device)

    logsumexp_batch_dim = cu_seqlens_q.size(0) - 1 if (cu_seqlens_q is not None) else B
    actual_max_seqlen_q = M
    if cu_seqlens_q is not None:
        assert max_seqlen_q is not None
        actual_max_seqlen_q = max_seqlen_q
    actual_max_seqlen_k = max_seqlen_k if max_seqlen_k is not None else N
    logsumexp_dim = (
        math.ceil(actual_max_seqlen_q / 32) * 32 if compute_log_sumexp else 0
    )
    logsum_exp = torch.empty(
        (logsumexp_batch_dim, num_heads, logsumexp_dim),
        dtype=torch.float,
        device=query.device,
    )

    # See Note [Seed and Offset]:
    seed = torch.empty((), dtype=torch.long, device="meta")
    offset = torch.empty((), dtype=torch.long, device="meta")

    return res, logsum_exp, seed, offset, actual_max_seqlen_q, actual_max_seqlen_k


@register_meta(
    [
        aten._efficient_attention_backward,
    ]
)
def meta__efficient_attention_backward(
    grad_out: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    bias: Optional[Tensor],
    cu_seqlens_q: Optional[Tensor],
    cu_seqlens_k: Optional[Tensor],
    max_seqlen_q: torch.SymInt,
    max_seqlen_k: torch.SymInt,
    logsumexp: Tensor,
    dropout_p: float,
    philox_seed: Tensor,
    philox_offset: Tensor,
    custom_mask_type: int,
    bias_requires_grad: bool,
    scale: Optional[float] = None,
    num_splits_key: Optional[int] = None,
    shared_storage_dqdkdv: bool = False,
):
    if shared_storage_dqdkdv:
        torch._check(
            query.shape[1] == key.shape[1],
            lambda: "seqlen must match for `shared_storage_dqdkdv",
        )
        torch._check(
            query.shape[3] == key.shape[3],
            lambda: "embedding dim must match for `shared_storage_dqdkdv",
        )
        chunk = torch.empty(
            (*query.shape[0:-2], 3, query.shape[-2], query.shape[-1]),
            dtype=query.dtype,
            device=query.device,
        )
        grad_query = chunk.select(-3, 0)
        grad_key = chunk.select(-3, 1)
        grad_value = chunk.select(-3, 2)
    else:
        grad_query = torch.empty_like(query)
        grad_key = torch.empty_like(key)
        grad_value = torch.empty_like(value)

    if bias is not None:
        lastDim = bias.size(-1)
        lastDimAligned = lastDim if lastDim % 16 == 0 else lastDim + 16 - lastDim % 16
        new_sizes = list(bias.size())
        new_sizes[-1] = lastDimAligned
        grad_bias = torch.empty(new_sizes, dtype=bias.dtype, device=bias.device)
        grad_bias = grad_bias[..., :lastDim]
    else:
        grad_bias = torch.empty((), device=query.device)

    return grad_query, grad_key, grad_value, grad_bias


@register_meta([aten._scaled_mm.default])
def meta_scaled_mm(
    self: torch.Tensor,
    mat2: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    scale_result: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    use_fast_accum: bool = False,
):
    def is_fp8_or_fp4_type(dtype):
        return dtype in (
            torch.float8_e4m3fn,
            torch.float8_e5m2,
            torch.float8_e4m3fnuz,
            torch.float8_e5m2fnuz,
            torch.float4_e2m1fn_x2,
        )

    torch._check(
        self.dim() == 2 and mat2.dim() == 2,
        lambda: f"Inputs must be 2D but got self.dim()={self.dim()} and mat2.dim()={mat2.dim()}",
    )
    torch._check(
        is_fp8_or_fp4_type(self.dtype) and is_fp8_or_fp4_type(mat2.dtype),
        lambda: f"Expected both inputs to be fp8 or fp4 types but got self.dtype={self.dtype} and mat2.dtype={mat2.dtype}",
    )

    if device_hint(self) == "cuda":

        def is_row_major(stride):
            return stride[0] > stride[1] and stride[1] == 1

        def is_col_major(stride):
            return stride[0] == 1 and stride[1] > 1

        def has_zero_dim(tensor_2d):
            return tensor_2d.size(0) == 0 or tensor_2d.size(1) == 0

        torch._check(
            is_row_major(self.stride()) or has_zero_dim(self),
            lambda: f"self must be row_major, got stride {self.stride()}",
        )
        torch._check(
            is_col_major(mat2.stride()) or has_zero_dim(mat2),
            lambda: f"mat2 must be col_major, got stride {mat2.stride()}",
        )
        torch._check(
            self.size(1) % 16 == 0,
            lambda: f"Expected self.size(1) to be divisible by 16, but got self.size(1)={self.size(1)}",
        )
        torch._check(
            mat2.size(0) % 16 == 0 and mat2.size(1) % 16 == 0,
            lambda: f"Expected both dimensions of mat2 to be divisible by 16 but got {mat2.shape}",
        )

        # determine scaling type and check input dimensions (refer to Blas.cpp op)

        m, _k = self.shape
        n = mat2.size(1)

        is_blockwise_scaling = (
            scale_a.dtype == torch.float8_e8m0fnu
            and scale_b.dtype == torch.float8_e8m0fnu
        ) or (
            scale_a.dtype == torch.float8_e4m3fn
            and scale_b.dtype == torch.float8_e4m3fn
        )

        if scale_a.numel() == 1 and scale_b.numel() == 1:
            # tensorwise scaling
            torch._check(
                scale_a.dtype == torch.float32 and scale_b.dtype == torch.float32,
                lambda: "For tensorwise scaling, both scale_a and scale_b must be float (fp32) tensors.",
            )
        elif is_blockwise_scaling:
            # blockwise scaling

            if scale_a.dtype == torch.float8_e4m3fn:
                # NVIDIA's nvfp4 recipe:
                # * block size is 16 elements packed (32 unpacked)
                # * _k needs to be translated to the unpacked version
                block_size_k = 16
                _k = _k * 2
            else:
                block_size_k = 32

            block_size_mn = 128

            def ceil_div(a, b):
                return (a + b - 1) // b

            num_k_blocks = ceil_div(_k, block_size_k)
            padded_num_k_blocks = ceil_div(num_k_blocks, 4) * 4

            expected_a_size = (
                block_size_mn * ceil_div(m, block_size_mn) * padded_num_k_blocks
            )
            expected_b_size = (
                block_size_mn * ceil_div(n, block_size_mn) * padded_num_k_blocks
            )

            if (
                scale_a.numel() == expected_a_size
                and scale_b.numel() == expected_b_size
            ):
                torch._check(
                    scale_a.is_contiguous(),
                    lambda: "scale_a must be contiguous",
                )
                torch._check(
                    scale_b.is_contiguous(),
                    lambda: "scale_b must be contiguous",
                )
            else:
                torch._check(
                    False,
                    lambda: (
                        "Invalid blockwise scaling configuration. "
                        f"For blockwise scaling, scale_a should have {expected_a_size} elements, got {scale_a.numel()}, "
                        f"scale_b should have {expected_b_size} elements, got {scale_b.numel()}."
                    ),
                )
        else:
            torch._check(
                scale_a.dtype == torch.float32 and scale_b.dtype == torch.float32,
                lambda: "For rowwise scaling, both scale_a and scale_b must be float (fp32) tensors.",
            )
            # for rowwise scaling, enforce 2D input tensors
            torch._check(
                scale_a.dim() == 2 and scale_b.dim() == 2,
                lambda: f"For non-tensorwise scaling, scale tensors must be 2D, but got {scale_a.dim()=} and {scale_b.dim()=}",
            )

            if (
                scale_a.size(0) == m
                and scale_a.size(1) == 1
                and scale_b.size(0) == 1
                and scale_b.size(1) == n
            ):
                # rowwise scaling
                torch._check(
                    scale_a.is_contiguous() and scale_b.is_contiguous(),
                    lambda: "Both scale_a and scale_b must be contiguous for rowwise scaling.",
                )
            else:
                # does not match any valid scaling type
                torch._check(
                    False,
                    lambda: (
                        "Invalid scaling configuration. "
                        "For tensorwise scaling, both scales should be scalar. "
                        f"For rowwise scaling, scale_a should be ({m}, 1), scale_b should be (1, {n}). "
                        f"Got scale_a.size()=({scale_a.size(0)}, {scale_a.size(1)}) "
                        f"and scale_b.size()=({scale_b.size(0)}, {scale_b.size(1)})"
                    ),
                )

    _out_dtype = out_dtype if out_dtype is not None else self.dtype
    return torch.empty(self.size(0), mat2.size(1), dtype=_out_dtype, device=self.device)


@register_meta([aten.scatter_reduce.two, aten.scatter_reduce.two_out])
@out_wrapper()
def meta_scatter_reduce_two(self, dim, index, src, reduce, include_self=True):
    scatter_meta_impl(self, dim, index, src, reduce, use_new_options=True)
    return self.new_empty(self.shape)


@register_meta(aten.scatter_reduce_.two)
def meta_scatter_reduce__two(self, dim, index, src, reduce, include_self=True):
    scatter_meta_impl(self, dim, index, src, reduce, use_new_options=True)
    return self


@register_meta([aten.multinomial.default, aten.multinomial.out])
@out_wrapper()
def meta_multinomial(input, num_samples, replacement=False, *, generator=None):
    torch._check(
        0 < input.dim() <= 2,
        lambda: f"The probability distributions dimensions must be 1 or 2, but got {input.dim()}",
    )
    if input.dim() == 1:
        return torch.empty(num_samples, dtype=torch.long, device=input.device)
    return torch.empty(
        input.size(0), num_samples, dtype=torch.long, device=input.device
    )


def multiply_integers(vs):
    r = 1
    for v in vs:
        r *= v
    return r


def upsample_common_check(input_size, output_size, num_spatial_dims):
    torch._check(
        len(output_size) == num_spatial_dims,
        lambda: f"It is expected output_size equals to {num_spatial_dims}, but got size {len(output_size)}",
    )
    expected_input_dims = num_spatial_dims + 2  # N, C, ...
    torch._check(
        len(input_size) == expected_input_dims,
        lambda: f"It is expected input_size equals to {expected_input_dims}, but got size {len(input_size)}",
    )

    torch._check(
        all(s > 0 for s in input_size[2:]) and all(s > 0 for s in output_size),
        lambda: f"Input and output sizes should be greater than 0, but got "
        f"input size {input_size} and output size {output_size}",
    )

    nbatch, channels = input_size[:2]
    return (nbatch, channels, *output_size)


@register_meta(
    [aten.upsample_nearest1d.default, aten._upsample_nearest_exact1d.default]
)
def upsample_nearest1d(input, output_size, scales=None):
    torch._check(
        input.numel() != 0 or multiply_integers(input.size()[1:]),
        lambda: f"Non-empty 3D data tensor expected but got a tensor with sizes {input.size()}",
    )
    full_output_size = upsample_common_check(
        input.size(), output_size, num_spatial_dims=1
    )
    return input.new_empty(full_output_size).to(
        memory_format=utils.suggest_memory_format(input)
    )


@register_meta(
    [aten.upsample_nearest2d.default, aten._upsample_nearest_exact2d.default]
)
def upsample_nearest2d(input, output_size, scales_h=None, scales_w=None):
    torch._check(
        input.numel() != 0 or multiply_integers(input.size()[1:]),
        lambda: f"Non-empty 4D data tensor expected but got a tensor with sizes {input.size()}",
    )
    full_output_size = upsample_common_check(
        input.size(), output_size, num_spatial_dims=2
    )
    output = input.new_empty(full_output_size)

    # convert output to correct memory format, if necessary
    memory_format = utils.suggest_memory_format(input)

    # following "heuristic: only use channels_last path when it's faster than the contiguous path"
    _, n_channels, _, _ = input.shape
    if input.device.type == "cuda" and n_channels < 4:
        memory_format = torch.contiguous_format

    output = output.contiguous(memory_format=memory_format)

    return output


@register_meta(
    [
        aten.upsample_nearest2d_backward.default,
        aten._upsample_nearest_exact2d_backward.default,
    ]
)
def upsample_nearest2d_backward(
    grad_output: Tensor,
    output_size: Sequence[Union[int, torch.SymInt]],
    input_size: Sequence[Union[int, torch.SymInt]],
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
):
    full_output_size = upsample_common_check(
        input_size, output_size, num_spatial_dims=2
    )
    torch._check(
        grad_output.ndim == 4,
        lambda: f"Expected grad_output to be a tensor of dimension 4 but got: dimension {grad_output.ndim}",
    )
    for i in range(4):
        torch._check(
            grad_output.size(i) == full_output_size[i],
            lambda: (
                f"Expected grad_output to have the same shape as output;"
                f" output.size({i}) = {full_output_size[i]}"
                f" but got grad_output.size({i}) = {grad_output.size(i)}"
            ),
        )

    return grad_output.new_empty(input_size).to(
        memory_format=utils.suggest_memory_format(grad_output)
    )  # type: ignore[call-overload]


@register_meta(
    [aten.upsample_nearest3d.default, aten._upsample_nearest_exact3d.default]
)
def upsample_nearest3d(input, output_size, scales_d=None, scales_h=None, scales_w=None):
    torch._check(
        input.numel() != 0 or multiply_integers(input.size()[1:]),
        lambda: f"Non-empty 5D data tensor expected but got a tensor with sizes {input.size()}",
    )
    full_output_size = upsample_common_check(
        input.size(), output_size, num_spatial_dims=3
    )
    return input.new_empty(full_output_size).to(
        memory_format=utils.suggest_memory_format(input)
    )


@register_meta(
    [
        aten.sort.default,
        aten.sort.stable,
        aten.sort.values,
        aten.sort.values_stable,
    ]
)
def meta_sort(self, stable=None, dim=-1, descending=False, values=None, indices=None):
    v, i = torch.empty_like(self), torch.empty_like(self, dtype=torch.int64)
    if values is not None and indices is not None:
        assert isinstance(values, TensorLike)
        assert isinstance(indices, TensorLike)
        # Makes sure values and indices have the same strides. For cases where
        # these have different shapes, like (5, 10, 5) and (0) in msort.
        out_shape = v.shape
        out_stride = v.stride()
        values = _maybe_resize_out(values, out_shape)
        indices = _maybe_resize_out(indices, out_shape)
        values.as_strided_(out_shape, out_stride)
        indices.as_strided_(out_shape, out_stride)
        _safe_copy_out(copy_from=v, copy_to=values)  # type: ignore[arg-type]
        _safe_copy_out(copy_from=i, copy_to=indices)  # type: ignore[arg-type]
        return values, indices
    return v, i


def rnn_cell_checkSizes(
    input_gates,
    hidden_gates,
    input_bias,
    hidden_bias,
    factor,
    prev_hidden,
):
    torch._check(input_gates.ndim == 2, lambda: f"{input_gates.ndim} != 2")
    torch._check(
        input_gates.shape == hidden_gates.shape,
        lambda: f"{input_gates.shape} != {hidden_gates.shape}",
    )
    gates_size = input_gates.size(1)
    if input_bias is not None:
        torch._check(input_bias.ndim == 1, lambda: f"{input_bias.ndim} != 1")
        torch._check(
            input_bias.numel() == gates_size,
            lambda: f"{input_bias.numel()} != {gates_size}",
        )
        torch._check(
            input_bias.shape == hidden_bias.shape,
            lambda: f"{input_bias.shape} != {hidden_bias.shape}",
        )
    torch._check(prev_hidden.ndim == 2, lambda: f"{prev_hidden.ndim} != 2")
    expected_prev_hidden_numel = input_gates.size(0) * gates_size // factor
    torch._check(
        prev_hidden.numel() == expected_prev_hidden_numel,
        lambda: f"{prev_hidden.numel()} != {input_gates.size(0)} * {gates_size} // {factor} (aka {expected_prev_hidden_numel})",
    )
    torch._check(
        all(
            x.device == input_gates.device
            for x in [hidden_gates, input_bias, hidden_bias, prev_hidden]
        ),
        lambda: "expected all inputs to be same device",
    )


@register_meta(aten._thnn_fused_lstm_cell.default)
def _thnn_fused_lstm_cell_meta(
    input_gates,
    hidden_gates,
    cx,
    input_bias=None,
    hidden_bias=None,
):
    rnn_cell_checkSizes(input_gates, hidden_gates, input_bias, hidden_bias, 4, cx)
    workspace = torch.empty_like(input_gates, memory_format=torch.contiguous_format)
    hy = torch.empty_like(cx, memory_format=torch.contiguous_format)
    cy = torch.empty_like(cx, memory_format=torch.contiguous_format)
    return (hy, cy, workspace)


@register_meta(aten._cudnn_rnn.default)
def _cudnn_rnn(
    input,
    weight,
    weight_stride0,
    weight_buf,
    hx,
    cx,
    mode,
    hidden_size,
    proj_size,
    num_layers,
    batch_first,
    dropout,
    train,
    bidirectional,
    batch_sizes,
    dropout_state,
):
    is_input_packed = len(batch_sizes) != 0
    if is_input_packed:
        seq_length = len(batch_sizes)
        mini_batch = batch_sizes[0]
        batch_sizes_sum = input.shape[0]
    else:
        seq_length = input.shape[1] if batch_first else input.shape[0]
        mini_batch = input.shape[0] if batch_first else input.shape[1]
        batch_sizes_sum = -1

    num_directions = 2 if bidirectional else 1
    out_size = proj_size if proj_size != 0 else hidden_size
    if is_input_packed:
        out_shape = [batch_sizes_sum, out_size * num_directions]
    else:
        out_shape = (
            [mini_batch, seq_length, out_size * num_directions]
            if batch_first
            else [seq_length, mini_batch, out_size * num_directions]
        )
    output = input.new_empty(out_shape)

    cell_shape = [num_layers * num_directions, mini_batch, hidden_size]
    if cx is None:
        cy = torch.empty(0, device=input.device)
    else:
        cy = cx.new_empty(cell_shape)

    hy = hx.new_empty([num_layers * num_directions, mini_batch, out_size])

    # TODO: Query cudnnGetRNNTrainingReserveSize (expose to python)
    reserve_shape = 0 if train else 0
    reserve = input.new_empty(reserve_shape, dtype=torch.uint8)

    return output, hy, cy, reserve, weight_buf


@register_meta(aten.mkldnn_rnn_layer.default)
def mkldnn_rnn_layer(
    input,
    w0,
    w1,
    w2,
    w3,
    hx_,
    cx_,
    reverse,
    batch_sizes,
    mode,
    hidden_size,
    num_layers,
    has_biases,
    bidirectional,
    batch_first,
    train,
):
    seq_length = input.shape[1] if batch_first else input.shape[0]
    mini_batch = input.shape[0] if batch_first else input.shape[1]
    output_chanels = hidden_size
    out_shape = (
        [mini_batch, seq_length, output_chanels]
        if batch_first
        else [seq_length, mini_batch, output_chanels]
    )
    output = input.new_empty(out_shape)
    if hx_ is None:
        hy = torch.empty(0, device=input.device)
    else:
        hy = hx_.new_empty(hx_.shape)
    if cx_ is None:
        cy = torch.empty(0, device=input.device)
    else:
        cy = cx_.new_empty(cx_.shape)
    workspace = torch.empty(0, device=input.device, dtype=torch.uint8)
    return output, hy, cy, workspace


def zero_numel_check_dims(self, dim, fn_name):
    if self.ndim == 0:
        torch._check_index(
            dim == 0 or dim == -1,
            lambda: f"{fn_name}: Expected reduction dim -1 or 0 for scalar but got {dim}",
        )
    else:
        torch._check_index(
            self.size(dim) != 0,
            lambda: f"{fn_name}: Expected reduction dim {dim} to have non-zero size.",
        )


# From aten/src/ATen/native/ReduceOps.cpp
def check_argmax_argmin(name, self, dim):
    if dim is not None:
        dim = maybe_wrap_dim(dim, self.dim())
        zero_numel_check_dims(self, dim, name)
    else:
        torch._check(
            self.numel() != 0,
            lambda: f"{name}: Expected reduction dim to be specified for input.numel() == 0.",
        )


@register_meta([aten.argmax.default, aten.argmin.default])
def argmax_argmin_meta(self, dim=None, keepdim=False):
    check_argmax_argmin("argmax", self, dim)
    dims = utils.reduction_dims(self.shape, (dim,) if dim is not None else None)
    shape = _compute_reduction_shape(self, dims, keepdim)
    return self.new_empty(shape, dtype=torch.int64)


@register_meta(aten.scalar_tensor.default)
def scalar_tensor(s, dtype=None, layout=None, device=None, pin_memory=None):
    # NB: It's always wrong to try to create a scalar tensor with the jagged layout.
    # Rather than fix this everywhere, just use the strided layout and let NJT handle
    # scalar tensor broadcasting.
    if layout == torch.jagged:
        layout = torch.strided
    return torch.empty(
        (), dtype=dtype, layout=layout, device=device, pin_memory=pin_memory
    )


@register_meta(aten.topk.default)
def topk_meta(self, k, dim=-1, largest=True, sorted=True):
    # From aten/src/ATen/native/Sorting.cpp
    dim = maybe_wrap_dim(dim, self.dim(), wrap_scalar=True)
    sliceSize = 1 if self.dim() == 0 else self.size(dim)
    torch._check_is_size(k)
    torch._check(k <= sliceSize, lambda: "k not in range for dimension")

    topKSize = list(self.shape)
    if len(topKSize) > 0:
        topKSize[dim] = k
    return self.new_empty(topKSize), self.new_empty(topKSize, dtype=torch.int64)


@register_meta(aten._segment_reduce_backward)
@out_wrapper()
def meta__segment_reduce_backward(
    grad, output, data, reduce, lengths=None, offsets=None, axis=0, initial=None
):
    assert lengths is not None or offsets is not None, (
        "segment_reduce(): Either lengths or offsets must be defined"
    )
    data_contig = data.contiguous()
    grad_contig = grad.contiguous()
    return torch.empty_like(
        data_contig,
        dtype=grad_contig.dtype,
        device=grad_contig.device,
        layout=grad_contig.layout,
    )


@register_meta([aten.kthvalue.default, aten.kthvalue.values])
@out_wrapper("values", "indices")
def kthvalue_meta(self, k, dim=-1, keepdim=False):
    from torch.fx.experimental.symbolic_shapes import sym_and

    dim = maybe_wrap_dim(dim, self.dim(), wrap_scalar=True)
    dimSize = self.size(dim) if self.dim() > 0 else 1
    torch._check(
        sym_and(k >= 1, k <= dimSize),
        lambda: f"kthvalue(): selected number k out of range for dimension {dim}",
    )

    shape = list(self.shape[:dim] + self.shape[dim + 1 :])
    if keepdim and self.dim() > 0:
        shape.insert(dim, 1)
    return self.new_empty(shape), self.new_empty(shape, dtype=torch.int64)


legacy_contiguous_memory_format = torch.contiguous_format


# From aten/src/ATen/native/cuda/RNN.cu
def checkLSTMBackwardSizes(grad_hy, grad_cy, cx, cy, workspace):
    defined_grad = grad_hy if grad_hy is not None else grad_cy
    torch._check(defined_grad.dim() == 2, lambda: "")
    exp_size = defined_grad.size()
    if grad_hy is not None:
        torch._check(grad_hy.size() == exp_size, lambda: "")
    if grad_cy is not None:
        torch._check(grad_cy.size() == exp_size, lambda: "")
    torch._check(cx.size() == exp_size, lambda: "")
    torch._check(cy.size() == exp_size, lambda: "")
    torch._check(workspace.dim() == 2, lambda: "")
    torch._check(workspace.numel() == exp_size[0] * exp_size[1] * 4, lambda: "")


# From aten/src/ATen/native/cuda/RNN.cu
@register_meta(aten._thnn_fused_lstm_cell_backward_impl.default)
def _thnn_fused_lstm_cell_backward_impl(grad_hy, grad_cy, cx, cy, workspace, has_bias):
    if grad_hy is None and grad_cy is None:
        return None, None, None
    checkLSTMBackwardSizes(grad_hy, grad_cy, cx, cy, workspace)
    grad_gates = torch.empty_like(
        workspace, memory_format=legacy_contiguous_memory_format
    )
    grad_cx = torch.empty_like(cx, memory_format=legacy_contiguous_memory_format)
    grad_bias = grad_gates.sum(0, keepdim=False) if has_bias else None
    return grad_gates, grad_cx, grad_bias


# From aten/src/ATen/native/mps/operations/Linear.mm
@register_meta(aten.linear_backward.default)
def linear_backward(input_, grad_output_, weight_, output_mask):
    grad_input = None
    grad_weight = None
    grad_bias = None
    if output_mask[0]:
        grad_input = grad_output_.new_empty(input_.size())
    if output_mask[1] or output_mask[2]:
        grad_weight = grad_output_.new_empty((grad_output_.size(-1), input_.size(-1)))
        grad_bias = grad_output_.new_empty(grad_output_.size(-1))
    return (grad_input, grad_weight, grad_bias)


@register_meta(aten.pixel_shuffle.default)
def meta_pixel_shuffle(self, upscale_factor):
    assert (
        len(self.shape) > 2 and self.shape[-3] % (upscale_factor * upscale_factor) == 0
    ), (
        f"Invalid input shape for pixel_shuffle: {self.shape} with upscale_factor = {upscale_factor}"
    )

    def is_channels_last(ten):
        return torch._prims_common.suggest_memory_format(ten) == torch.channels_last

    def pick_memory_format():
        if is_channels_last(self):
            if device_hint(self) == "cuda":
                return torch.contiguous_format
            else:
                return torch.channels_last
        elif self.is_contiguous(memory_format=torch.contiguous_format):
            return torch.contiguous_format
        elif self.is_contiguous(memory_format=torch.preserve_format):
            return torch.preserve_format

    C = self.shape[-3] // (upscale_factor * upscale_factor)
    Hr = self.shape[-2] * upscale_factor
    Wr = self.shape[-1] * upscale_factor
    out_shape = (*self.shape[:-3], C, Hr, Wr)

    out = self.new_empty(out_shape)
    out = out.to(memory_format=pick_memory_format())  # type: ignore[call-overload]
    return out


@register_meta(aten.mkldnn_rnn_layer_backward.default)
def mkldnn_rnn_layer_backward(
    input,
    weight0,
    weight1,
    weight2,
    weight3,
    hx_,
    cx_tmp,
    output,
    hy_,
    cy_,
    grad_output_r_opt,
    grad_hy_r_opt,
    grad_cy_r_opt,
    reverse,
    mode,
    hidden_size,
    num_layers,
    has_biases,
    train,
    bidirectional,
    batch_sizes,
    batch_first,
    workspace,
):
    diff_x = input.new_empty(input.shape)
    diff_hx = hx_.new_empty(hx_.shape)
    diff_cx = cx_tmp.new_empty(cx_tmp.shape)
    diff_w1 = weight0.new_empty(weight0.shape)
    diff_w2 = weight1.new_empty(weight1.shape)
    diff_b = weight2.new_empty(weight2.shape)
    return diff_x, diff_w1, diff_w2, diff_b, diff_b, diff_hx, diff_cx


@register_meta([aten.bucketize.Tensor, aten.bucketize.Tensor_out])
@out_wrapper()
def meta_bucketize(self, boundaries, *, out_int32=False, right=False):
    return torch.empty_like(
        self,
        dtype=torch.int32 if out_int32 else torch.int64,
        memory_format=torch.contiguous_format,
    )


@register_meta([aten.histc])
@out_wrapper()
def meta_histc(input, bins=100, min=0, max=0):
    fn_name = "histc()"
    if device_hint(input) == "cpu":
        torch._check(
            input.is_floating_point(),
            lambda: f"\"histogram_cpu\" not implemented for '{input.dtype}'",
        )
    if device_hint(input) == "cuda" and input.is_floating_point():
        utils.alert_not_deterministic("_histc_cuda with floating point input")
    torch._check(
        isinstance(bins, IntLike),
        lambda: f"{fn_name}: argument 'bins' must be int, not {type(bins)}",
    )
    torch._check(bins > 0, lambda: f"{fn_name}: bins must be > 0, but got {bins}")
    torch._check(
        isinstance(min, Number),
        lambda: f"{fn_name}: argument 'min' must be Number, not {type(min)}",
    )
    torch._check(
        isinstance(max, Number),
        lambda: f"{fn_name}: argument 'max' must be Number, not {type(max)}",
    )
    torch._check(max >= min, lambda: "{fn_name}: max must be larger than min")
    return torch.empty(bins, device=input.device, dtype=input.dtype)


@register_meta(
    [aten._upsample_bilinear2d_aa.default, aten._upsample_bicubic2d_aa.default]
)
def meta_upsample_bimode2d_aa(
    input,
    output_size,
    align_corners,
    scales_h=None,
    scales_w=None,
):
    full_output_size = upsample_common_check(
        input.size(), output_size, num_spatial_dims=2
    )
    torch._check(
        input.numel() != 0 or all(size > 0 for size in input.size()[1:]),
        lambda: f"Non-empty 4D data tensor expected but got a tensor with sizes {input.size()}",
    )
    return input.new_empty(full_output_size).to(
        memory_format=utils.suggest_memory_format(input)
    )


@register_meta([aten._upsample_bilinear2d_aa_backward.default])
def meta_upsample_bimode2d_aa_backward(
    grad_output,
    output_size,
    input_size,
    align_corners,
    scales_h=None,
    scales_w=None,
):
    full_output_size = upsample_common_check(
        input_size, output_size, num_spatial_dims=2
    )
    torch._check(
        grad_output.ndim == 4,
        lambda: f"Expected grad_output to be a tensor of dimension 4 but got: dimension {grad_output.ndim}",
    )
    for i in range(4):
        torch._check(
            grad_output.shape[i] == full_output_size[i],
            lambda: f"""
Expected grad_output to have the same shape as output; output.size({i}) = {full_output_size[i]}
but got grad_output_size({i}) = {grad_output.size(i)}""",
        )
    return grad_output.new_empty(input_size).to(
        memory_format=utils.suggest_memory_format(grad_output)
    )


# From aten/src/ATen/native/cuda/AmpKernels.cu
@register_meta(aten._amp_foreach_non_finite_check_and_unscale_.default)
def _amp_foreach_non_finite_check_and_unscale_(self, found_inf, inv_scale):
    torch._check(
        found_inf.numel() == 1, lambda: "found_inf must be a 1-element tensor."
    )
    torch._check(
        inv_scale.numel() == 1, lambda: "inv_scale must be a 1-element tensor."
    )
    torch._check(
        found_inf.dtype.is_floating_point,
        lambda: "found_inf must be a float tensor.",
    )
    torch._check(
        inv_scale.dtype.is_floating_point,
        lambda: "inv_scale must be a float tensor.",
    )


# From aten/src/ATen/native/UnaryOps.cpp
@register_meta([aten.nan_to_num.default, aten.nan_to_num.out])
@out_wrapper()
def nan_to_num(self, nan=None, posinf=None, neginf=None):
    result_size = list(self.size())
    return self.new_empty(result_size)


@register_meta(torch.ops.aten.transpose_)
def transpose_(self, dim0, dim1):
    assert self.layout not in {
        torch.sparse_csr,
        torch.sparse_csc,
        torch.sparse_bsr,
        torch.sparse_bsc,
    }, (
        f"torch.transpose_: in-place transposition is not supported for {self.layout} layout"
    )

    ndims = self.ndim

    dim0 = maybe_wrap_dim(dim0, ndims)
    dim1 = maybe_wrap_dim(dim1, ndims)

    if dim0 == dim1:
        return self

    size = list(self.size())
    stride = list(self.stride())

    stride[dim0], stride[dim1] = stride[dim1], stride[dim0]
    size[dim0], size[dim1] = size[dim1], size[dim0]

    self.as_strided_(size, stride)
    return self


@register_meta(torch.ops.aten.t_)
def t_(self):
    ndims = self.ndim

    if self.is_sparse:
        sparse_dim = self.sparse_dim()
        dense_dim = self.dense_dim()
        assert sparse_dim <= 2 and dense_dim == 0, (
            f"t_ expects a tensor with <= 2 sparse and 0 dense dimensions, "
            f"but got {sparse_dim} sparse and {dense_dim} dense dimensions"
        )
    else:
        assert self.dim() <= 2, (
            f"t_ expects a tensor with <= 2 dimensions, but self is {ndims}D"
        )

    return transpose_(self, 0, 0 if ndims < 2 else 1)


@register_meta(aten.searchsorted)
@out_wrapper()
def meta_searchsorted(
    sorted_sequence,
    self,
    *,
    out_int32=False,
    right=False,
    side=None,
    sorter=None,
):
    # If the sorted_sequence is not one-dimensional, its shape must match that of values
    # in all but the last dimension.
    torch._check(
        len(sorted_sequence.shape) <= 1
        or sorted_sequence.shape[:-1] == self.shape[:-1],
        lambda: (
            "torch.searchsorted(): boundaries tensor should be 1 dimension or the "
            "first N-1 dimensions of boundaries tensor and input value tensor must "
            f"match, but we got boundaries tensor {list(sorted_sequence.shape)} and "
            f"input value tensor {list(self.shape)}"
        ),
    )

    # If a sorter array is provided, its dimensions must exactly match sorted_sequence.
    torch._check(
        sorter is None or sorted_sequence.shape == sorter.shape,
        lambda: (
            "torch.searchsorted(): boundary and sorter must have the same size, but "
            f"got boundary tensor {list(sorted_sequence.shape)} and got sorter tensor "
            f"{list(sorter.shape) if sorter is not None else []}"
        ),
    )

    # Per the docs, if side == "left" and right is True, we error.
    torch._check(
        side != "left" or not right,
        "torch.searchsorted(): side and right can't be set to opposites, got side of "
        "left while right was True",
    )

    dtype = torch.int32 if out_int32 else torch.int64
    if isinstance(self, torch.Tensor):
        return torch.empty_like(
            self, dtype=dtype, memory_format=torch.contiguous_format
        )
    else:  # Scalar
        return torch.empty((), dtype=dtype, device=sorted_sequence.device)


def _check_for_unsupported_isin_dtype(dtype):
    torch._check(
        dtype not in (torch.bool, torch.complex128, torch.complex64),
        lambda: f"Unsupported input type encountered for isin(): {dtype}",
    )


@register_meta(aten.embedding_dense_backward)
def meta_embedding_dense_backward(
    grad_output,
    indices,
    num_weights,
    padding_idx,
    scale_grad_by_freq,
):
    grad_weight = grad_output.new_empty((num_weights, grad_output.size(-1)))
    return grad_weight


@register_meta(aten._embedding_bag_backward)
def meta_embedding_bag_backward(
    grad,
    indices,
    offsets,
    offset2bag,
    bag_size,
    maximum_indices,
    num_weights,
    scale_grad_by_freq,
    mode,
    sparse,
    per_sample_weights,
    padding_idx=-1,
):
    if sparse:
        return aten._embedding_bag_sparse_backward(
            grad,
            indices,
            offsets,
            offset2bag,
            bag_size,
            num_weights,
            scale_grad_by_freq,
            mode,
            per_sample_weights,
            padding_idx,
        )
    else:
        return meta_embedding_bag_dense_backward(
            grad,
            indices,
            offset2bag,
            bag_size,
            maximum_indices,
            num_weights,
            scale_grad_by_freq,
            mode,
            per_sample_weights,
            padding_idx,
        )


@register_meta(aten._embedding_bag_dense_backward)
def meta_embedding_bag_dense_backward(
    grad,
    indices,
    offset2bag,
    bag_size,
    maximum_indices,
    num_weights,
    scale_grad_by_freq,
    mode,
    per_sample_weights,
    padding_idx=-1,
):
    torch._check(
        grad.dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64],
        lambda: f"Unsupported input type encountered: {grad.dtype}",
    )
    if mode == MODE_MAX:
        torch._check(maximum_indices is not None)
    index_grad_weight = grad.new_empty((num_weights, grad.size(1)))
    return index_grad_weight


@register_meta(aten._embedding_bag_per_sample_weights_backward)
def meta_embedding_bag_per_sample_weights_backward(
    grad,
    weight,
    indices,
    offsets,
    offset2bag,
    mode,
    padding_idx=-1,
):
    embedding_features = grad.size(1)
    torch._check(
        mode == MODE_SUM,
        "embedding_bag_backward: per_sample_weights only supported for mode='sum'",
    )
    torch._check(grad.dim() == 2)
    torch._check(indices.dim() == 1)
    num_samples = indices.size(0)
    torch._check(weight.dim() == 2)
    torch._check(weight.size(1) == embedding_features)
    output = grad.new_empty((num_samples,))
    return output


@register_meta(aten.isin)
@out_wrapper()
def meta_isin(elements, test_elements, *, assume_unique=False, invert=False):
    torch._check(
        isinstance(elements, Tensor) or isinstance(test_elements, Tensor),
        lambda: "At least one of elements and test_elements must be a Tensor.",
    )
    if not isinstance(elements, Tensor):
        elements = torch.tensor(elements, device=test_elements.device)

    if not isinstance(test_elements, Tensor):
        test_elements = torch.tensor(test_elements, device=elements.device)

    _check_for_unsupported_isin_dtype(elements.dtype)
    _check_for_unsupported_isin_dtype(test_elements.dtype)
    return torch.empty_like(elements, dtype=torch.bool)


@register_meta(aten.polygamma)
@out_wrapper()
def meta_polygamma(n: int, self: Tensor) -> Tensor:
    torch._check(n >= 0, lambda: "polygamma(n, x) does not support negative n.")
    _, result_dtype = elementwise_dtypes(
        self,
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
    )
    return torch.empty_like(self, dtype=result_dtype)


@register_meta(aten._local_scalar_dense)
def meta_local_scalar_dense(self: Tensor):
    raise RuntimeError("Tensor.item() cannot be called on meta tensors")


@register_meta(aten.silu)
@out_wrapper(exact_dtype=True)
def silu(self: Tensor) -> Tensor:
    return torch.empty_like(self)


@register_meta(aten.sigmoid)
@out_wrapper()
def sigmoid(self: Tensor) -> Tensor:
    _, result_dtype = elementwise_dtypes(
        self,
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
    )
    return torch.empty_like(self, dtype=result_dtype)


def _create_grouped_mm_output_tensor(mat1, mat2, offs, out_dtype):
    mat1_is_2d = mat1.dim() == 2
    mat2_is_2d = mat2.dim() == 2

    if mat1_is_2d:
        if mat2_is_2d:
            out_size = [offs.size(0), mat1.size(0), mat2.size(1)]
        else:
            torch._check(
                offs.size(0) == mat2.size(0), lambda: "matrix batch sizes have to match"
            )
            out_size = [mat1.size(0), mat2.size(-1)]
    else:
        if mat2_is_2d:
            torch._check(
                offs.size(0) == mat1.size(0), lambda: "matrix batch sizes have to match"
            )
            out_size = [mat1.size(1), mat2.size(1)]
        else:
            # regular bmm
            torch._check(
                mat1.size(0) == mat2.size(0), lambda: "batched dimension has to match"
            )
            out_size = [mat1.size(0), mat1.size(1), mat2.size(-1)]

    out_dtype = out_dtype or mat1.dtype

    alignment = 16 // out_dtype.itemsize
    size_padded = (out_size[-1] + alignment - 1) // alignment * alignment
    if mat1_is_2d == mat2_is_2d:
        out_stride = [out_size[1] * size_padded, size_padded, 1]
    else:
        out_stride = [size_padded, 1]
    out = torch.empty_strided(out_size, out_stride, dtype=out_dtype, device=mat1.device)
    return out


def _meta_grouped_mm_common(
    mat_a: Tensor,
    mat_b: Tensor,
    scale_a: Optional[torch.Tensor],
    scale_b: Optional[torch.Tensor],
    offs: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    scale_result: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    use_fast_accum: bool = False,
):
    torch._check(
        (scale_a is None) == (scale_b is None),
        lambda: "Either both scale factors are given, or none",
    )
    scaled = scale_a is not None and scale_b is not None

    # Implementing all the checks from
    # _grouped_mm_cuda()/_scaled_grouped_mm_cuda() code in
    # aten/src/ATen/native/cuda/Blas.cpp.

    if scaled:
        torch._check(
            mat_a.dtype == torch.float8_e4m3fn and mat_b.dtype == torch.float8_e4m3fn,
            lambda: f"Expected inputs of E4M3 FP8 type but got mat_a.dtype={mat_a.dtype} and mat_b.dtype={mat_b.dtype}.",
        )
    else:
        torch._check(
            mat_a.dtype == torch.bfloat16 and mat_b.dtype == torch.bfloat16,
            lambda: f"Expected inputs of BF16 type but got mat_a.dtype={mat_a.dtype} and mat_b.dtype={mat_b.dtype}.",
        )

    torch._check(
        mat_a.dim() in [2, 3] and mat_b.dim() in [2, 3],
        lambda: f"Multiplicands must be 2D or 3D but got mat_a.dim()={mat_a.dim()} and mat_b.dim()={mat_b.dim()}",
    )

    mat_a_is_2d = mat_a.dim() == 2
    mat_b_is_2d = mat_b.dim() == 2

    if scaled:

        def is_row_major(mat):
            mat_stride = mat.stride()
            return mat_stride[-2] > 1 and mat_stride[-1] == 1

        def is_col_major(mat):
            mat_stride = mat.stride()
            return mat_stride[-2] == 1 and mat_stride[-1] > 1

        torch._check(
            is_row_major(mat_a),
            lambda: f"Expected mat_a tensor to be row major in the last two dimensions, got strides {mat_a.stride()[-2:]}",
        )
        torch._check(
            is_col_major(mat_b),
            lambda: f"Expected mat_b tensor to be column major in the last two dimensions, got strides {mat_b.stride()[-2:]}",
        )

    def check_valid_strides(mat_name, mat):
        end_dim = mat.dim() - 1
        alignment = 16 // mat.element_size()
        mat_stride = mat.stride()
        if mat_stride[end_dim - 1] == 1 and mat_stride[end_dim] >= max(
            1, mat.shape[end_dim - 1]
        ):
            torch._check(
                mat_stride[end_dim] % alignment == 0,
                lambda: f"Expected {mat_name} stride along {end_dim} dim to be multiple of 16 bytes, got {mat_stride[end_dim]}.",
            )
        elif mat_stride[end_dim] == 1 and mat_stride[end_dim - 1] >= max(
            1, mat.shape[end_dim]
        ):
            torch._check(
                mat_stride[end_dim - 1] % alignment == 0,
                lambda: f"Expected {mat_name} stride along {end_dim - 1} dim to be multiple of 16 bytes, got {mat_stride[end_dim - 1]}.",  # noqa: B950
            )
        else:
            torch._check(
                False,
                lambda: f"Invalid strides/sizes, got {mat_stride} for strides and {mat.shape} for sizes.",  # noqa: B950
            )

    check_valid_strides("mat_a", mat_a)
    check_valid_strides("mat_b", mat_b)

    if scale_a is not None and scale_b is not None:
        torch._check(
            scale_a.dtype == torch.float32 and scale_b.dtype == torch.float32,
            lambda: "Both scale_a and scale_b must be float (fp32) tensors, but got scale_a.dtype={scale_a.dtype} and scale_b.dtype={scale_b.dtype}.",  # noqa: B950
        )

        def check_scale(scale_name, scale, mat, scaled_dim, scale_multiplier=1):
            if mat.dim() == 2:
                torch._check(
                    scale.dim() == 1,
                    lambda: f"Expected {scale_name} to be 1D tensor, but got {scale.dim()}D tensor.",
                )
                torch._check(
                    scale.is_contiguous(),
                    lambda: f"Expected {scale_name} to be contiguous.",
                )
                torch._check(
                    scale.shape[0] == mat.shape[scaled_dim] * scale_multiplier,
                    lambda: f"Expected {scale_name} to have {mat.shape[scaled_dim] * scale_multiplier} elements, got {scale.shape[0]} elements.",  # noqa: B950
                )
            else:
                torch._check(
                    scale.dim() == 2,
                    lambda: f"Expected {scale_name} to be 2D tensor, but got {scale.dim()}D tensor.",
                )
                torch._check(
                    scale.stride(1) == 1,
                    lambda: f"Expected {scale_name} to be contiguous in the last dimension.",
                )
                torch._check(
                    scale.shape[0] == mat.shape[0],
                    lambda: f"Expected {scale_name} batch dimension to be {mat.shape[0]}, got {scale.shape[0]}.",
                )
                torch._check(
                    scale.shape[1] == mat.shape[1 + scaled_dim],
                    lambda: f"Expected {scale_name} non-batch dimension to be {mat.shape[1 + scaled_dim]}, got {scale.shape[1]}.",
                )

        scale_multiplier = (
            offs.shape[0] if offs is not None and mat_a_is_2d and mat_b_is_2d else 1
        )
        check_scale("scale_a", scale_a, mat_a, 0, scale_multiplier)
        check_scale("scale_b", scale_b, mat_b, 1, scale_multiplier)

        torch._check(
            scale_result is None,
            lambda: "Scale result tensor provided, but it is not supported yet.",
        )

    if mat_a_is_2d or mat_b_is_2d:
        torch._check(
            offs is not None,
            lambda: f"Offsets tensor not provided, but is needed for {mat_a.dim()}D/{mat_b.dim()}D multiplicand layouts.",
        )
        if offs is not None:  # to silence Mypy
            torch._check(
                offs.dim() == 1,
                lambda: f"Offsets tensor must be 1D, but got offs.dim()={offs.dim()}.",
            )
            torch._check(
                offs.dtype == torch.int32,
                lambda: f"Offsets tensor must be integer (int32) tensor, but got {offs.dtype}.",
            )
    else:
        torch._check(
            offs is None,
            lambda: "Offsets tensor provided, but is not needed for 3D/3D multiplicand layouts.",
        )

    torch._check(
        bias is None,
        lambda: "Bias tensor provided, but it is not supported yet.",
    )

    torch._check(
        out_dtype is None or out_dtype == torch.bfloat16,
        lambda: "If output dtype provided, it must be torch.bfloat16.",
    )

    return _create_grouped_mm_output_tensor(mat_a, mat_b, offs, out_dtype)


@register_meta(aten._grouped_mm)
@out_wrapper()
def meta_grouped_mm(
    mat_a: Tensor,
    mat_b: Tensor,
    offs: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> Tensor:
    return _meta_grouped_mm_common(
        mat_a,
        mat_b,
        scale_a=None,
        scale_b=None,
        offs=offs,
        bias=bias,
        scale_result=None,
        out_dtype=out_dtype,
    )


@register_meta([aten._scaled_grouped_mm])
def meta_scaled_grouped_mm(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    offs: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    scale_result: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    use_fast_accum: bool = False,
):
    return _meta_grouped_mm_common(
        mat_a,
        mat_b,
        scale_a=scale_a,
        scale_b=scale_b,
        offs=offs,
        bias=bias,
        scale_result=scale_result,
        out_dtype=out_dtype,
        use_fast_accum=use_fast_accum,
    )


@register_meta(aten._softmax)
@out_wrapper()
def softmax(x: Tensor, dim: int, half_to_float: bool) -> Tensor:
    if half_to_float:
        assert x.dtype == torch.half
    computation_dtype, result_dtype = utils.elementwise_dtypes(
        x, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    )

    result_dtype = result_dtype if not half_to_float else computation_dtype
    res = torch.empty_like(x, dtype=result_dtype, memory_format=torch.contiguous_format)
    return res


@register_meta(aten.constant_pad_nd)
@out_wrapper()
def _constant_pad_nd_meta(input, pad, value=0):
    # same checks as decomposition in torch/_refs/__init__.py:constant_pad_nd()
    torch._check(
        len(pad) % 2 == 0,
        lambda: f"Length of pad must be even but instead it equals {len(pad)}",
    )

    input_sizes = input.shape
    l_inp = len(input_sizes)
    l_pad = len(pad) // 2
    l_diff = l_inp - l_pad

    torch._check(
        l_inp >= l_pad,
        lambda: "Length of pad should be no more than twice the number of "
        f"dimensions of the input. Pad length is {len(pad)} while the input has "
        f"{l_inp} dimensions.",
    )

    new_shape = list(input_sizes[:l_diff])
    for i in range(l_pad):
        pad_idx = len(pad) - ((i + 1) * 2)
        new_dim = input_sizes[l_diff + i] + pad[pad_idx] + pad[pad_idx + 1]
        torch._check(
            new_dim >= 0,
            lambda: f"The input size {input_sizes[l_diff + i]}, plus negative padding "
            f"{pad[pad_idx]} and {pad[pad_idx + 1]} resulted in a negative output size, "
            f"which is invalid. Check dimension {l_diff + i} of your input.",
        )
        new_shape.append(new_dim)

    return torch.empty(
        new_shape,
        dtype=input.dtype,
        device=input.device,
        requires_grad=input.requires_grad,
        memory_format=suggest_memory_format(input),
    )


@register_meta(aten.embedding)
@out_wrapper()
def embedding(
    weight: Tensor,
    indices: Tensor,
    padding_idx: int = -1,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
) -> Tensor:
    assert weight.dim() == 2, "'weight' must be 2-D"
    weight_shape = weight.shape
    indices_shape = indices.shape

    if indices.ndim == 0:
        out_shape: tuple[int, ...] = (weight_shape[1],)
    elif indices.ndim == 1:
        out_shape = (indices_shape[0], weight_shape[1])
    else:
        out_shape = (*indices_shape, weight_shape[1])

    out_dtype = weight.dtype
    return weight.new_empty(out_shape, dtype=out_dtype)


@register_meta(aten._jagged_to_padded_dense_forward.default)
def meta__jagged_to_padded_dense_forward(
    values: Tensor,
    offsets: list[Tensor],
    max_lengths: list[int],
    padding_value: float = 0.0,
):
    # only one jagged dim is supported for now
    assert len(offsets) == 1
    assert len(max_lengths) == 1

    B = offsets[0].shape[0] - 1
    S = max_lengths[0]
    output_shape = (B, S, *values.shape[1:])
    return values.new_empty(output_shape)


def _create_unary_float_meta_func(func):
    @register_meta(func)
    @out_wrapper()
    def _f(x):
        return elementwise_meta(
            x, type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
        )

    return _f


def _create_binary_float_meta_func(func):
    @register_meta(func)
    @out_wrapper()
    def _f(x, y):
        return elementwise_meta(
            x, y, type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
        )

    return _f


_create_unary_float_meta_func(aten.special_airy_ai)
_create_unary_float_meta_func(aten.special_bessel_y0)
_create_unary_float_meta_func(aten.special_bessel_y1)
_create_unary_float_meta_func(aten.special_modified_bessel_i0)
_create_unary_float_meta_func(aten.special_modified_bessel_i1)
_create_unary_float_meta_func(aten.special_modified_bessel_k0)
_create_unary_float_meta_func(aten.special_modified_bessel_k1)
_create_unary_float_meta_func(aten.special_scaled_modified_bessel_k0)
_create_unary_float_meta_func(aten.special_scaled_modified_bessel_k1)


_create_binary_float_meta_func(aten.special_chebyshev_polynomial_t)
_create_binary_float_meta_func(aten.special_chebyshev_polynomial_u)
_create_binary_float_meta_func(aten.special_chebyshev_polynomial_v)
_create_binary_float_meta_func(aten.special_chebyshev_polynomial_w)
_create_binary_float_meta_func(aten.special_shifted_chebyshev_polynomial_t)
_create_binary_float_meta_func(aten.special_shifted_chebyshev_polynomial_u)
_create_binary_float_meta_func(aten.special_shifted_chebyshev_polynomial_v)
_create_binary_float_meta_func(aten.special_shifted_chebyshev_polynomial_w)
_create_binary_float_meta_func(aten.special_hermite_polynomial_h)
_create_binary_float_meta_func(aten.special_hermite_polynomial_he)
_create_binary_float_meta_func(aten.special_laguerre_polynomial_l)
_create_binary_float_meta_func(aten.special_legendre_polynomial_p)


def _register_inplace_meta(fn):
    @wraps(fn)
    def _fn(self, *args, **kwargs):
        out = fn(self, *args, **kwargs)
        check_inplace_broadcast(self.shape, out.shape)
        return self

    inplace_name = f"{fn.__name__}_"
    _fn.__name__ = inplace_name
    _fn = register_meta(getattr(aten, inplace_name))(_fn)  # type: ignore[assignment]

    return _fn


@register_meta(aten.lerp)
@out_wrapper()
def lerp(start, end, weight):
    torch._check(
        start.dtype == end.dtype,
        lambda: f"expected dtype {start.dtype} for `end`, but got dtype {end.dtype}",
    )
    args = [start, end]
    if isinstance(weight, TensorLike):
        if weight.ndim != 0:
            torch._check(
                start.dtype == weight.dtype,
                lambda: f"expected dtype {start.dtype} for `weight`, but got dtype {weight.dtype}",
            )
        args.append(weight)
    return elementwise_meta(
        *args, type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    )


@register_meta(aten.addcmul)
@out_wrapper()
def addcmul(input, tensor1, tensor2, *, value=1):
    return elementwise_meta(
        input, tensor1, tensor2, type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    )


@register_meta(aten.addcdiv)
@out_wrapper()
def addcdiv(input, tensor1, tensor2, *, value=1):
    torch._check(
        not (
            utils.is_integer_dtype(tensor1.dtype)
            and utils.is_integer_dtype(tensor2.dtype)
        ),
        lambda: (
            "Integer division with addcdiv is no longer supported, and in a future ",
            "release addcdiv will perform a true division of tensor1 and tensor2. ",
            "The historic addcdiv behavior can be implemented as ",
            "(input + value * torch.trunc(tensor1 / tensor2)).to(input.dtype) ",
            "for integer inputs and as ",
            "(input + value * tensor1 / tensor2) for float inputs. ",
            "The future addcdiv behavior is just the latter implementation: ",
            "(input + value * tensor1 / tensor2), for all dtypes.",
        ),
    )
    return elementwise_meta(
        input, tensor1, tensor2, type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    )


lerp_ = _register_inplace_meta(aten.lerp)
addcmul_ = _register_inplace_meta(aten.addcmul)
addcdiv_ = _register_inplace_meta(aten.addcdiv)


# We must also trigger meta registrations from PrimTorch ref
# decompositions
import torch._refs
import torch._refs.nn.functional
import torch._refs.special


def activate_meta():
    activate_meta_table = {}

    # For a given op, we pick the most specific decomp function from
    # global_decomp_table in the precedence order of meta > post_autograd > pre_autograd
    for type in ["meta", "post_autograd", "pre_autograd"]:
        registry = global_decomposition_table[type]

        for opo in registry:
            if opo not in activate_meta_table:
                activate_meta_table[opo] = registry[opo]

    for op_overload, fn in activate_meta_table.items():
        # Don't register meta for HigherOrderOp's decomp.
        # We can reconsider this in the future, but in general,
        # the way you do a meta for a HigherOrderOp is different from
        # OpOverload.
        if isinstance(op_overload, torch._ops.HigherOrderOperator):
            continue
        assert isinstance(op_overload, OpOverload)

        op_overload.py_impl(torch._C.DispatchKey.Meta)(fn)

        if torch._C._dispatch_has_kernel_for_dispatch_key(
            op_overload.name(), "CompositeImplicitAutograd"
        ):
            # Internally, we shouldn't be registering meta kernels for any operators that
            # have CompositeImplicitAutograd kernels.
            # Instead, we should be letting those decompositions run, and writing meta kernels
            # only for the base operators.
            if op_overload in global_decomposition_table["meta"]:
                raise RuntimeError(
                    f"{op_overload} is a CompositeImplicitAutograd op, we shouldn't "
                    "register meta function for it. Instead, we should let the decomposition run and write "
                    "meta kernels for the base operators."
                )
        elif op_overload.is_view:
            # Attempting to register a python meta kernel for a view operator.
            # We shouldn't do this, because the output will report as not having aliased storages.
            # All view ops have meta kernels in C++ today, so we should use those instead.
            pass
        elif (
            op_overload.name()
            in {
                "aten::empty_strided",  # causing infinite recursion, test_meta.py
                "aten::clone",  # causing infinite recursion
                "aten::_to_copy",  # causing infinite recursion, test_serialization.py -k test_tensor_subclass_getstate_overwrite  # noqa: B950
                "aten::copy_",  # Exception not raised, test_torch.py -k test_storage_meta_errors_cpu_int64  # noqa: B950
                "aten::constant_pad_nd",  # requires_grad mismatch, test_ops.py -k test_fake_crossref_backward_amp_istft_cuda_float32  # noqa: B950
                "aten::rot90",  # requires_grad mismatch! test_ops.py -k test_fake_crossref_backward_amp_rot90_cuda_float32  # noqa: B950
                "aten::as_strided_scatter",  # requires_grad mismatch, test_ops.py -k test_fake_crossref_backward_no_amp_as_strided_scatter_cuda_float32  # noqa: B950
            }
        ):
            pass
        else:
            if "mkldnn::" in op_overload.name():
                _meta_lib_dont_use_me_use_register_meta_for_mkldnn.impl(op_overload, fn)
            elif "mkl::" in op_overload.name():
                _meta_lib_dont_use_me_use_register_meta_for_mkl.impl(op_overload, fn)
            elif "onednn::" in op_overload.name():
                _meta_lib_dont_use_me_use_register_meta_for_onednn.impl(op_overload, fn)
            elif "quantized::" in op_overload.name():
                _meta_lib_dont_use_me_use_register_meta_for_quantized.impl(
                    op_overload, fn
                )
            else:
                _meta_lib_dont_use_me_use_register_meta.impl(op_overload, fn)


activate_meta()
