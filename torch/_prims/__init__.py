import torch
from torch import Tensor, _TypedStorage

import torch._prims.utils as utils
from torch._prims.utils import (
    TensorLike,
    TensorLikeType,
    ShapeType,
    getnvFuserDtype,
    DimsType,
    DimsSequenceType,
    StrideType,
    Number,
    NumberType,
    TensorMeta,
)
from torch.overrides import has_torch_function, handle_torch_function
import torch.library
from torch.utils._pytree import tree_map, tree_flatten, tree_unflatten
from torch._subclasses.fake_tensor import FakeTensor

import contextlib
from typing import Sequence, Optional, Union, Callable, List, Tuple, Any, Type
from functools import reduce, partial
from enum import Enum
import operator
import math

prim = torch.library.Library("prims", "DEF")
prim_impl = torch.library.Library("prims", "IMPL", "CompositeExplicitAutograd")
prim_autograd_impl = torch.library.Library("prims", "IMPL", "Autograd")
prim_meta_impl = torch.library.Library("prims", "IMPL", "Meta")

# Experimental module containing prototype "primitive" operations.

__all__ = [
    #
    # Common datastructures and helpers
    #
    "RETURN_TYPE",
    #
    # Elementwise unary prims
    #
    "abs",
    "acos",
    "acosh",
    "asin",
    "asinh",
    "atan",
    "atanh",
    "cos",
    "cosh",
    "bessel_i0",
    "bessel_i0e",
    "bessel_i1",
    "bessel_i1e",
    "bitwise_not",
    "cbrt",
    "ceil",
    "digamma",
    "erf",
    "erf_inv",
    "erfc",
    "exp",
    "expm1",
    "exp2",
    "fill",
    "floor",
    "isfinite",
    "is_infinite",
    "lgamma",
    "log",
    "log1p",
    "log2",
    "log10",
    "neg",
    "reciprocal",
    "round",
    "sign",
    "signbit",
    "sin",
    "sinh",
    "sqrt",
    "tan",
    "tanh",
    "trunc",
    #
    # Elementwise binary prims
    #
    "add",
    "atan2",
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    # 'complex',  # needs custom meta
    "div",
    "eq",
    "fmax",
    "fmin",
    "fmod",
    "gcd",
    "ge",
    "gt",
    "igamma",
    "igammac",
    "le",
    "lt",
    "maximum",
    "minimum",
    "mul",
    "ne",
    "nextafter",
    "pow",
    "remainder",
    "rsqrt",
    "shift_left",
    "shift_right_arithmetic",
    "shift_right_logical",  # not implemented
    "sub",
    "zeta",
    #
    # View prims
    #
    "as_strided",
    "broadcast_in_dim",
    "collapse_view",
    "expand_dims",
    "slice",
    "slice_in_dim",  # implemented using slice -- make this a ref?
    "split_dim",
    "squeeze",
    "transpose",
    "view_of",
    #
    # Shape prims
    #
    "collapse",
    "cat",
    "reshape",
    "rev",
    #
    # Conditional prims
    #
    "where",
    #
    # Data conversion and movement prims
    #
    "clone",
    "convert_element_type",
    "device_put",
    "item",
    "maximum_value",
    "minimum_value",
    "to_dtype",
    #
    # Inplace prims
    #
    "copy_to",
    "resize",
    # "_set",  # Commented out, see note below
    #
    # Reduction prims
    #
    "amax",
    "amin",
    "prod",
    "sum",
    "var",
    #
    # Tensor Creation Prims
    #
    "empty_strided",
    "scalar_tensor",
    #
    # Randomness Prims
    #
    "uniform",
]

#
# Common datastructures and helpers
#

_nvfuser_unary_ops = {
    "abs",
    "acos",
    "asin",
    "atan",
    "atanh",
    "cos",
    "cosh",
    "bitwise_not",
    "ceil",
    "erf",
    "erfc",
    "exp",
    "expm1",
    "floor",
    "isfinite",
    "lgamma",
    "log",
    "log1p",
    "log2",
    "log10",
    "reciprocal",
    "neg",
    "round",
    "rsqrt",
    "sin",
    "sinh",
    "sqrt",
    "tan",
    "tanh",
}


def _assert_nvfuser_op_exists(fname: str):
    try:
        from torch._C._nvfuser import FusionDefinition as fd  # type: ignore[import]

        assert getattr(fd.Ops, fname)
    except ImportError:
        # Not all PyTorch builds have nvfuser
        pass


for fname in _nvfuser_unary_ops:
    exec(
        f"""
# Ensure that the nvfuser implementation exists
_assert_nvfuser_op_exists("{fname}")

def _{fname}_nvfuser(fd: Any, a: TensorLikeType):
    return fd.Ops.{fname}(a)  # type: ignore[attr-defined]
"""
    )

_nvfuser_binary_ops = {
    "add",
    "atan2",
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "div",
    "eq",
    "fmod",
    "ge",
    "gt",
    "le",
    "lt",
    "mul",
    "ne",
    "pow",
    "sub",
}

for fname in _nvfuser_binary_ops:
    exec(
        f"""
# Ensure that the nvfuser implementation exists
_assert_nvfuser_op_exists("{fname}")

def _{fname}_nvfuser(fd: Any, a: TensorLikeType, b: TensorLikeType):
    return fd.Ops.{fname}(a, b)  # type: ignore[attr-defined]
"""
    )

_nvfuser_ternary_ops = {
    "where",
}

for fname in _nvfuser_ternary_ops:
    exec(
        f"""
# Ensure that the nvfuser implementation exists
_assert_nvfuser_op_exists("{fname}")

def _{fname}_nvfuser(fd: Any, a: TensorLikeType, b: TensorLikeType, c: TensorLikeType):
    return fd.Ops.{fname}(a, b, c)  # type: ignore[attr-defined]
"""
    )

# Describes the return type of the primitive:
#
#   - NEW, a new tensor is created
#   - VIEW, a view of an input tensor is returned
#   - INPLACE, one or more input tensors is modified
#
# these descriptors are mututally exclusive and exhaustive.
class RETURN_TYPE(Enum):
    NEW = (0,)
    VIEW = (1,)
    INPLACE = (2,)


def _wrap_tensor_meta(f):
    def wrap(t):
        if (
            isinstance(t, torch.Tensor)
            and not isinstance(t, FakeTensor)
            and not t.device.type == "meta"
        ):
            return FakeTensor.from_tensor(t, utils.get_prim_fake_mode())
        else:
            return t

    def wrapper(*args, **kwargs):
        wrapped_args = tree_map(wrap, args)
        wrapped_kwargs = tree_map(wrap, kwargs)
        return f(*wrapped_args, **wrapped_kwargs)

    return wrapper


def _make_prim(
    *,
    schema: str,
    return_type: RETURN_TYPE,
    meta: Callable,
    impl_aten: Callable,
    impl_nvfuser: Optional[Callable] = None,
    doc: str,
):
    """
    Creates a primitive operation.

    """

    prim.define(schema)

    def _prim_impl(*args, **kwargs):
        # always run the meta function because aten implementation will
        # typically accept more inputs (e.g., it will do promotion and
        # broadcasting) which we want to reject
        meta(*args, **kwargs)
        return impl_aten(*args, **kwargs)

    # Right now prims don't support autograd (we can and should add an
    # argument that provides an implementation for backward here.)  Because we
    # don't have derivative formulas, we must setup a custom autograd function
    # that raises an error if backwards is invoked
    class BackwardsNotSupported(torch.autograd.Function):
        @staticmethod
        def forward(ctx, args_spec, *flat_args):
            args, kwargs = tree_unflatten(flat_args, args_spec)  # type: ignore[arg-type]
            g = torch._C._AutoDispatchBelowAutograd()
            try:
                return _prim(*args, **kwargs)
            finally:
                del g

        @staticmethod
        def backward(ctx, *args):
            raise RuntimeError("backwards not supported on prim")

    def _autograd_impl(*args, **kwargs):
        flat_args, args_spec = tree_flatten((args, kwargs))
        return BackwardsNotSupported.apply(args_spec, *flat_args)

    name = schema.split("(")[0]
    prim_impl.impl(name, _prim_impl)
    prim_autograd_impl.impl(name, _autograd_impl)
    prim_meta_impl.impl(name, _wrap_tensor_meta(meta))

    _prim_packet = getattr(torch.ops.prims, name)
    _prim = _prim_packet.default

    for p in (_prim_packet, _prim):
        p.__doc__ = doc
        p.impl_nvfuser = impl_nvfuser  # type: ignore[attr-defined]
        p.return_type = return_type  # type: ignore[attr-defined]

    return _prim


class ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND(Enum):
    DEFAULT = (0,)
    ALWAYS_BOOL = (2,)
    COMPLEX_TO_FLOAT = (3,)


# TODO: implement dtype validation here, too, or on the corresponding refs
def _elementwise_meta(
    *args,
    type_promotion: ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND,
    args_with_fixed_dtypes: Tuple[TensorLikeType, ...] = None,
) -> FakeTensor:
    """
    Meta function for elementwise operations that produce outputs in the same dtype
    as their inputs.

    Stride logic is currently incorrect.
    """

    assert len(args) > 0

    utils.check_same_dtype(*args)

    args_ = list(args)
    if args_with_fixed_dtypes is not None:
        args_.extend(args_with_fixed_dtypes)

    utils.check_same_device(*args_, allow_cpu_scalar_tensors=True)
    utils.check_same_shape(*args_, allow_cpu_scalar_tensors=True)

    strides = utils.compute_elementwise_output_strides(*args_)
    shape = utils.extract_shape(*args_, allow_cpu_scalar_tensors=True)

    # Acquires the dtype
    dtype = None
    scalar_type = None
    for arg in args:
        if isinstance(arg, TensorLike):
            if not utils.is_cpu_scalar_tensor(arg):
                dtype = arg.dtype
                break
            else:
                dtype = arg.dtype
        elif isinstance(arg, Number):
            scalar_type = type(arg)

    if dtype is None and scalar_type is not None:
        dtype = utils.type_to_dtype(scalar_type)

    # Acquires the device (if it exists) or number
    device = None
    number = None
    for arg in args_:
        if isinstance(arg, TensorLike):
            device = arg.device
            break

        elif isinstance(arg, Number):
            if number is None:
                number = arg

    # NOTE: type promotion behavior here is mostly hidden from tests because
    # references will typically handle the type promotion properly even if this doesn't
    # (but getting it wrong will cause too many casts to be inserted in traces!)
    if device is not None:
        assert dtype is not None
        if type_promotion == ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT:
            dtype = dtype
        elif type_promotion == ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.ALWAYS_BOOL:
            dtype = torch.bool
        elif type_promotion == ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.COMPLEX_TO_FLOAT:
            if utils.is_complex_dtype(dtype):
                dtype = utils.corresponding_real_dtype(dtype)
            else:
                dtype = dtype

        return TensorMeta(device=device, shape=shape, strides=strides, dtype=dtype)

    # Number case
    # NOTE: this case is not currently exercised
    # TODO: fix number type promotion (bool, complex->float)
    return TensorMeta(number)


def _make_elementwise_unary_prim(
    name: str, *, type_promotion: ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND, **kwargs
):
    """
    Creates an elementwise unary prim.
    """

    return _make_prim(
        schema=f"{name}(Tensor self) -> Tensor",
        meta=partial(_elementwise_meta, type_promotion=type_promotion),
        return_type=RETURN_TYPE.NEW,
        **kwargs,
    )


def _make_elementwise_binary_prim(
    name: str, *, type_promotion: ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND, **kwargs
):
    """
    Creates an elementwise binary prim.
    """

    return _make_prim(
        schema=f"{name}(Tensor self, Tensor other) -> Tensor",
        meta=partial(_elementwise_meta, type_promotion=type_promotion),
        return_type=RETURN_TYPE.NEW,
        **kwargs,
    )


def _not_impl(*args, **kwargs):
    raise NotImplementedError


#
# Elementwise unary operations
#


abs = _make_elementwise_unary_prim(
    "abs",
    impl_aten=torch.abs,
    impl_nvfuser=_abs_nvfuser,  # type: ignore[name-defined]
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.COMPLEX_TO_FLOAT,
)

acos = _make_elementwise_unary_prim(
    "acos",
    impl_aten=torch.acos,
    impl_nvfuser=_acos_nvfuser,  # type: ignore[name-defined]
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

acosh = _make_elementwise_unary_prim(
    "acosh",
    impl_aten=torch.acosh,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

asin = _make_elementwise_unary_prim(
    "asin",
    impl_aten=torch.asin,
    impl_nvfuser=_asin_nvfuser,  # type: ignore[name-defined]
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

asinh = _make_elementwise_unary_prim(
    "asinh",
    impl_aten=torch.asinh,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

atan = _make_elementwise_unary_prim(
    "atan",
    impl_aten=torch.atan,
    impl_nvfuser=_atan_nvfuser,  # type: ignore[name-defined]
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

atanh = _make_elementwise_unary_prim(
    "atanh",
    impl_aten=torch.atanh,
    impl_nvfuser=_atanh_nvfuser,  # type: ignore[name-defined]
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

cos = _make_elementwise_unary_prim(
    "cos",
    impl_aten=torch.cos,
    impl_nvfuser=_cos_nvfuser,  # type: ignore[name-defined]
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

cosh = _make_elementwise_unary_prim(
    "cosh",
    impl_aten=torch.cosh,
    impl_nvfuser=_cosh_nvfuser,  # type: ignore[name-defined]
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

bessel_i0 = _make_elementwise_unary_prim(
    "bessel_i0",
    impl_aten=torch.i0,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

bessel_i0e = _make_elementwise_unary_prim(
    "bessel_i0e",
    impl_aten=torch.special.i0e,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

bessel_i1 = _make_elementwise_unary_prim(
    "bessel_i1",
    impl_aten=torch.special.i1,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

bessel_i1e = _make_elementwise_unary_prim(
    "bessel_i1e",
    impl_aten=torch.special.i1e,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

bitwise_not = _make_elementwise_unary_prim(
    "bitwise_not",
    impl_aten=torch.bitwise_not,
    impl_nvfuser=_bitwise_not_nvfuser,  # type: ignore[name-defined]
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)


def _cbrt_aten(a: torch.Tensor) -> Tensor:
    utils.check(
        not a.is_complex(),
        lambda: "cbrt: Complex inputs not supported. Consider calling torch.pow(a, 1.0/3.0)",
    )
    # Returns the real cubic root of the number.
    # Note that if a < 0, pow(a, (1. / 3.)) returns th complex number
    # exp(1/3 * log(a)) = exp(1/3 * (log(abs(a)) + pi*i)) = cbrt(abs(a)) * e^{pi/3*i}
    # which is a complex number.
    # For more info see the section Note in
    # https://en.cppreference.com/w/cpp/numeric/math/cbrt
    return torch.copysign(torch.pow(a.abs(), 1 / 3), a)


cbrt = _make_elementwise_unary_prim(
    "cbrt",
    impl_aten=_cbrt_aten,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

ceil = _make_elementwise_unary_prim(
    "ceil",
    impl_aten=torch.ceil,
    impl_nvfuser=_ceil_nvfuser,  # type: ignore[name-defined]
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

digamma = _make_elementwise_unary_prim(
    "digamma",
    impl_aten=torch.digamma,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

erf = _make_elementwise_unary_prim(
    "erf",
    impl_aten=torch.erf,
    impl_nvfuser=_erf_nvfuser,  # type: ignore[name-defined]
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

erf_inv = _make_elementwise_unary_prim(
    "erf_inv",
    impl_aten=torch.special.erfinv,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

erfc = _make_elementwise_unary_prim(
    "erfc",
    impl_aten=torch.special.erfc,
    impl_nvfuser=_erfc_nvfuser,  # type: ignore[name-defined]
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

exp = _make_elementwise_unary_prim(
    "exp",
    impl_aten=torch.exp,
    impl_nvfuser=_exp_nvfuser,  # type: ignore[name-defined]
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

expm1 = _make_elementwise_unary_prim(
    "expm1",
    impl_aten=torch.special.expm1,
    impl_nvfuser=_expm1_nvfuser,  # type: ignore[name-defined]
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

exp2 = _make_elementwise_unary_prim(
    "exp2",
    impl_aten=torch.special.exp2,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)


def _fill_meta(a: TensorLikeType, value: NumberType) -> TensorLikeType:
    return _elementwise_meta(
        a, type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT
    )


# See https://github.com/pytorch/pytorch/issues/77932 for out-of-place fill request
def _fill_aten(a: Tensor, value: NumberType) -> Tensor:
    t = a * False
    with torch.no_grad():
        t.fill_(value)  # type: ignore[arg-type]
    return t


# NOTE: fill uses _make_prim directly because it has a value parameter
fill = _make_prim(
    schema="fill(Tensor self, Scalar value) -> Tensor",
    return_type=RETURN_TYPE.NEW,
    meta=_fill_meta,
    impl_aten=_fill_aten,
    doc="",
)

floor = _make_elementwise_unary_prim(
    "floor",
    impl_aten=torch.floor,
    impl_nvfuser=_floor_nvfuser,  # type: ignore[name-defined]
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

isfinite = _make_elementwise_unary_prim(
    "isfinite",
    impl_aten=torch.isfinite,
    impl_nvfuser=_isfinite_nvfuser,  # type: ignore[name-defined]
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
)

is_infinite = _make_elementwise_unary_prim(
    "is_infinite",
    impl_aten=torch.isinf,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
)

lgamma = _make_elementwise_unary_prim(
    "lgamma",
    impl_aten=torch.lgamma,
    impl_nvfuser=_lgamma_nvfuser,  # type: ignore[name-defined]
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

log = _make_elementwise_unary_prim(
    "log",
    impl_aten=torch.log,
    impl_nvfuser=_log_nvfuser,  # type: ignore[name-defined]
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

log1p = _make_elementwise_unary_prim(
    "log1p",
    impl_aten=torch.log1p,
    impl_nvfuser=_log1p_nvfuser,  # type: ignore[name-defined]
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

log2 = _make_elementwise_unary_prim(
    "log2",
    impl_aten=torch.log2,
    impl_nvfuser=_log2_nvfuser,  # type: ignore[name-defined]
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

log10 = _make_elementwise_unary_prim(
    "log10",
    impl_aten=torch.log10,
    impl_nvfuser=_log10_nvfuser,  # type: ignore[name-defined]
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

reciprocal = _make_elementwise_unary_prim(
    "reciprocal",
    impl_aten=torch.reciprocal,
    impl_nvfuser=_reciprocal_nvfuser,  # type: ignore[name-defined]
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

neg = _make_elementwise_unary_prim(
    "neg",
    impl_aten=torch.neg,
    impl_nvfuser=_neg_nvfuser,  # type: ignore[name-defined]
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

round = _make_elementwise_unary_prim(
    "round",
    impl_aten=torch.round,
    impl_nvfuser=_round_nvfuser,  # type: ignore[name-defined]
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

rsqrt = _make_elementwise_unary_prim(
    "rsqrt",
    impl_aten=torch.rsqrt,
    impl_nvfuser=_rsqrt_nvfuser,  # type: ignore[name-defined]
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

sign = _make_elementwise_unary_prim(
    "sign",
    impl_aten=torch.sign,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

signbit = _make_elementwise_unary_prim(
    "signbit",
    impl_aten=torch.signbit,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

sin = _make_elementwise_unary_prim(
    "sin",
    impl_aten=torch.sin,
    impl_nvfuser=_sin_nvfuser,  # type: ignore[name-defined]
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

sinh = _make_elementwise_unary_prim(
    "sinh",
    impl_aten=torch.sinh,
    impl_nvfuser=_sinh_nvfuser,  # type: ignore[name-defined]
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

sqrt = _make_elementwise_unary_prim(
    "sqrt",
    impl_aten=torch.sqrt,
    impl_nvfuser=_sqrt_nvfuser,  # type: ignore[name-defined]
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

tan = _make_elementwise_unary_prim(
    "tan",
    impl_aten=torch.tan,
    impl_nvfuser=_tan_nvfuser,  # type: ignore[name-defined]
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

tanh = _make_elementwise_unary_prim(
    "tanh",
    impl_aten=torch.tanh,
    impl_nvfuser=_tanh_nvfuser,  # type: ignore[name-defined]
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)


def _trunc_nvfuser(fd: Any, a: TensorLikeType):
    return fd.Ops.trunc(a)  # type: ignore[attr-defined]


trunc = _make_elementwise_unary_prim(
    "trunc",
    impl_aten=torch.trunc,
    impl_nvfuser=_trunc_nvfuser,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

#
# Elementwise binary operations
#

add = _make_elementwise_binary_prim(
    name="add",
    impl_aten=torch.add,
    impl_nvfuser=_add_nvfuser,  # type: ignore[name-defined]
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

atan2 = _make_elementwise_binary_prim(
    name="atan2",
    impl_aten=torch.atan2,
    impl_nvfuser=_atan2_nvfuser,  # type: ignore[name-defined]
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

bitwise_and = _make_elementwise_binary_prim(
    "bitwise_and",
    impl_aten=torch.bitwise_and,
    impl_nvfuser=_bitwise_and_nvfuser,  # type: ignore[name-defined]
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

bitwise_or = _make_elementwise_binary_prim(
    "bitwise_or",
    impl_aten=torch.bitwise_or,
    impl_nvfuser=_bitwise_or_nvfuser,  # type: ignore[name-defined]
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

bitwise_xor = _make_elementwise_binary_prim(
    "bitwise_xor",
    impl_aten=torch.bitwise_xor,
    impl_nvfuser=_bitwise_xor_nvfuser,  # type: ignore[name-defined]
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

# TODO: complex needs a special meta to account for its float -> complex behavior
# complex = _make_elementwise_binary_prim(
#   impl_aten=torch.complex,
#   doc="",
# )

# div prim performs truncation division on integer inputs
#   and true division for floating and complex inputs
def _div_aten(a, b):
    if isinstance(a, (bool, int)):
        return torch.div(a, b, rounding_mode="trunc")
    return torch.true_divide(a, b)


div = _make_elementwise_binary_prim(
    "div",
    impl_aten=_div_aten,
    impl_nvfuser=_div_nvfuser,  # type: ignore[name-defined]
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

eq = _make_elementwise_binary_prim(
    "eq",
    impl_aten=torch.eq,
    impl_nvfuser=_eq_nvfuser,  # type: ignore[name-defined]
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
)

fmax = _make_elementwise_binary_prim(
    "fmax",
    impl_aten=torch.fmax,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

fmin = _make_elementwise_binary_prim(
    "fmin",
    impl_aten=torch.fmin,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

fmod = _make_elementwise_binary_prim(
    "fmod",
    impl_aten=torch.fmod,
    impl_nvfuser=_fmod_nvfuser,  # type: ignore[name-defined]
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)


gcd = _make_elementwise_binary_prim(
    "gcd",
    impl_aten=torch.gcd,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)


ge = _make_elementwise_binary_prim(
    "ge",
    impl_aten=torch.ge,
    impl_nvfuser=_ge_nvfuser,  # type: ignore[name-defined]
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
)

gt = _make_elementwise_binary_prim(
    "gt",
    impl_aten=torch.gt,
    impl_nvfuser=_gt_nvfuser,  # type: ignore[name-defined]
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
)

igamma = _make_elementwise_binary_prim(
    "igamma",
    impl_aten=torch.special.gammainc,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

igammac = _make_elementwise_binary_prim(
    "igammac",
    impl_aten=torch.special.gammaincc,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

le = _make_elementwise_binary_prim(
    "le",
    impl_aten=torch.le,
    impl_nvfuser=_le_nvfuser,  # type: ignore[name-defined]
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
)

lt = _make_elementwise_binary_prim(
    "lt",
    impl_aten=torch.lt,
    impl_nvfuser=_lt_nvfuser,  # type: ignore[name-defined]
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
)


# Note: the following impls are because torch.maximum and torch.mininum do not support scalar inputs
def _maximum_aten(
    a: Union[TensorLikeType, NumberType], b: Union[TensorLikeType, NumberType]
) -> TensorLikeType:
    if isinstance(a, TensorLike) and isinstance(b, Number):
        b = scalar_tensor(b, dtype=a.dtype, device=a.device)
    elif isinstance(b, TensorLike) and isinstance(a, Number):
        a = scalar_tensor(a, dtype=b.dtype, device=b.device)

    return torch.maximum(a, b)  # type: ignore[arg-type]


maximum = _make_elementwise_binary_prim(
    "maximum",
    impl_aten=_maximum_aten,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)


def _minimum_aten(
    a: Union[TensorLikeType, NumberType], b: Union[TensorLikeType, NumberType]
) -> TensorLikeType:
    if isinstance(a, TensorLike) and isinstance(b, Number):
        b = scalar_tensor(b, dtype=a.dtype, device=a.device)
    elif isinstance(b, TensorLike) and isinstance(a, Number):
        a = scalar_tensor(a, dtype=b.dtype, device=b.device)

    return torch.minimum(a, b)  # type: ignore[arg-type]


minimum = _make_elementwise_binary_prim(
    "minimum",
    impl_aten=_minimum_aten,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

mul = _make_elementwise_binary_prim(
    "mul",
    impl_aten=torch.mul,
    impl_nvfuser=_mul_nvfuser,  # type: ignore[name-defined]
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

ne = _make_elementwise_binary_prim(
    "ne",
    impl_aten=torch.ne,
    impl_nvfuser=_ne_nvfuser,  # type: ignore[name-defined]
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
)

nextafter = _make_elementwise_binary_prim(
    "nextafter",
    impl_aten=torch.nextafter,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

pow = _make_elementwise_binary_prim(
    "pow",
    impl_aten=torch.pow,
    impl_nvfuser=_pow_nvfuser,  # type: ignore[name-defined]
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)


def _remainder_nvfuser(fd: Any, a: TensorLikeType, b: TensorLikeType):
    return fd.Ops.remainder(a, b)  # type: ignore[attr-defined]


remainder = _make_elementwise_binary_prim(
    "remainder",
    impl_aten=torch.remainder,
    impl_nvfuser=_remainder_nvfuser,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)


shift_left = _make_elementwise_binary_prim(
    "shift_left",
    impl_aten=torch.bitwise_left_shift,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

shift_right_arithmetic = _make_elementwise_binary_prim(
    "shift_right_arithmetic",
    impl_aten=torch.bitwise_right_shift,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

shift_right_logical = _not_impl

sub = _make_elementwise_binary_prim(
    "sub",
    impl_aten=torch.sub,
    impl_nvfuser=_sub_nvfuser,  # type: ignore[name-defined]
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

zeta = _make_elementwise_binary_prim(
    "zeta",
    impl_aten=torch.special.zeta,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

#
# View operations
#
# TODO: model view relationships
# TODO: model storage
def _as_strided_meta(
    a: TensorLikeType, size: ShapeType, stride: StrideType, storage_offset: int
) -> TensorLikeType:
    assert len(size) == len(stride)
    assert storage_offset >= 0
    utils.validate_strides(stride)
    utils.validate_shape(size)

    if reduce(operator.mul, size) == 0:
        # NOTE: This special case is to avoid having to acquire the storage below
        # as_strided to shapes with no elements are trivially valid, so it's OK
        pass
    elif isinstance(a, torch.Tensor):
        utils.check_in_bounds_for_storage(a.storage(), size, stride, storage_offset)

    return TensorMeta(a, shape=size, strides=stride)


def _as_strided_aten(
    a: Tensor, size: ShapeType, stride: StrideType, storage_offset: int
) -> Tensor:
    return torch.as_strided(a, size, stride, storage_offset)


_as_strided_doc = """
    Creates a view of the tensor with the given shape (size), strides (stride) and
    storage offset (storage_offset).
"""

as_strided = _make_prim(
    schema="as_strided(Tensor(a!) a, int[] size, int[] stride, int storage_offset) -> Tensor(a!)",
    meta=_as_strided_meta,
    impl_aten=_as_strided_aten,
    return_type=RETURN_TYPE.VIEW,
    doc=_as_strided_doc,
)


def _broadcast_in_dim_meta(
    a: TensorLikeType, shape: ShapeType, broadcast_dimensions: Sequence[int]
):
    # Type checks
    assert isinstance(a, TensorLike)
    assert isinstance(shape, Sequence)
    assert isinstance(broadcast_dimensions, Sequence)

    # every dimension must be accounted for
    assert a.ndim == len(broadcast_dimensions)

    # broadcast shape must have weakly more dimensions
    assert len(shape) >= a.ndim

    # broadcast_dimensions must be an ascending sequence
    # (no relative reordering of dims) of integers and
    # each dimension must be within the new shape
    def _greater_than_reduce(acc, x):
        assert isinstance(x, int)
        assert x > acc
        assert x < len(shape)

        return x

    reduce(lambda acc, x: _greater_than_reduce(acc, x), broadcast_dimensions, -1)

    # shape must be broadcastable to
    for idx, new_idx in enumerate(broadcast_dimensions):
        assert a.shape[idx] == 1 or a.shape[idx] == shape[new_idx]

    new_strides = []
    original_idx = 0
    for idx in range(len(shape)):
        if idx in broadcast_dimensions:
            # Assigns a stride of zero to dimensions
            # which were actually broadcast
            if a.shape[original_idx] != shape[idx]:
                new_strides.append(0)
            else:
                new_strides.append(a.stride()[original_idx])
            original_idx = original_idx + 1
        else:
            new_strides.append(0)

    return TensorMeta(a, shape=shape, strides=new_strides)


def _broadcast_in_dim_aten(a, shape, broadcast_dimensions):
    s = list(shape)
    for broadcast_dimension in broadcast_dimensions:
        s[broadcast_dimension] = -1

    v = a
    for idx, x in enumerate(s):
        if x != -1:
            v = v.unsqueeze(idx)

    return v.expand(shape)


def _broadcast_in_dim_nvfuser(
    fd: Any,
    a: torch.Tensor,
    shape: ShapeType,
    broadcast_dimensions: ShapeType,
):
    return fd.Ops.broadcast_in_dim(a, shape, broadcast_dimensions)  # type: ignore[attr-defined]


_broadcast_in_dim_doc = """
  Creates a view of a with the specified shape.

  Allows adding dimensions of any length and broadcasting
  dimensions of length one in a to any length.

  The location of the broadcast dimensions must be specified
  using the broadcast_dimensions argument. Changing the
  relative order of dimensions is not supported.
  """

broadcast_in_dim = _make_prim(
    schema="broadcast_in_dim(Tensor(a) a, int[] shape, int[] broadcast_dimensions) -> Tensor(a)",
    meta=_broadcast_in_dim_meta,
    impl_aten=_broadcast_in_dim_aten,
    impl_nvfuser=_broadcast_in_dim_nvfuser,
    return_type=RETURN_TYPE.VIEW,
    doc=_broadcast_in_dim_doc,
)


def _collapse_view_helper(
    a: TensorLikeType, start: int, end: int
) -> Tuple[Optional[ShapeType], Optional[StrideType]]:
    assert isinstance(a, TensorLike)

    # Special-case for zero dimensional tensors
    if a.ndim == 0:
        shape = (1,)
        strides = (1,)
    else:
        shape = a.shape  # type: ignore[assignment]
        strides = a.stride()

    utils.validate_idx(len(shape), start)
    utils.validate_exclusive_idx(len(shape), end)

    # Verifies end is strictly greater than start
    # (Collapse requires a non-empty interval)
    if end <= start:
        msg = "Attempting to collapse but end, {0}, is less than or equal to start, {1}!".format(
            end, start
        )
        raise ValueError(msg)

    if a.ndim == 0 or (end - 1 == start):
        return shape, strides

    length = shape[end - 1]
    stride = strides[end - 1]
    for idx in reversed(range(start, end - 1)):
        if shape[idx] == 0 or shape[idx + 1] == 0:
            length = 0
            stride = 0
            break

        if shape[idx] == 1:
            continue

        length = length * shape[idx]
        stride = min(stride, strides[idx])

        if (
            a.numel() > 0
            and shape[idx + 1] != 1
            and not (strides[idx] == strides[idx + 1] * shape[idx + 1])
        ):
            return None, None

    new_shape = shape[:start] + (length,) + shape[end:]
    new_strides = strides[:start] + (stride,) + strides[end:]

    # NOTE: when the input has no elements it's restrided as if it were contiguous
    if a.numel() == 0:
        new_strides = utils.make_contiguous_strides_for(new_shape)

    return new_shape, new_strides


def _collapse_view_meta(a: TensorLikeType, start: int, end: int) -> TensorLikeType:
    new_shape, new_strides = _collapse_view_helper(a, start, end)

    if new_shape is None:
        msg = "Attempting to view a collapsed tensor, but no such view exists!"
        raise ValueError(msg)

    return TensorMeta(a, shape=new_shape, strides=new_strides)


def _collapse_view_aten(a: Tensor, start: int, end: int) -> Tensor:
    # Special-cases zero-dim tensors
    if a.ndim == 0:
        shape = (1,)
    else:
        shape = a.shape  # type: ignore[assignment]

    dim_length = 1
    for idx in range(start, end):
        dim_length = dim_length * shape[idx]

    new_shape = shape[0:start] + (dim_length,) + shape[end:]

    return a.view(new_shape)


_collapse_view_doc = """
  Creates a view of a with the dimensions between
  start (inclusive) and end (exclusive) merged into a
  single dimension.

  If it's not possible to take such a view then an error
  is thrown. See collapse instead.

  The dimensions can be merged if and only if
  they are all "nested" with each other. That is, they all
  have the property that

  stride[i] = stride[i+1] * shape[i+1]

  for all i in [start, end - 1).
  """

collapse_view = _make_prim(
    schema="collapse_view(Tensor(a) a, int start, int end) -> Tensor(a)",
    meta=_collapse_view_meta,
    impl_aten=_collapse_view_aten,
    return_type=RETURN_TYPE.VIEW,
    doc=_collapse_view_doc,
)


def expand_dims(a: TensorLikeType, dimensions: DimsSequenceType) -> TensorLikeType:
    """
    Creates a view of a with a.ndim + len(dimensions) dimensions, with new
    dimensions of length one at the dimensions specified by dimensions.
    """
    dims = sorted(utils.canonicalize_dims(a.ndim, dimensions))  # type: ignore[arg-type]
    if len(set(dims)) != len(dims):
        msg = "Received duplicate dimensions to expand in {0}".format(str(dimensions))
        raise ValueError(msg)

    new_shape = list(a.shape)
    for idx in dims:
        new_shape.insert(idx, 1)

    broadcast_dimensions = [
        idx for idx in range(len(new_shape)) if idx not in dimensions
    ]
    return broadcast_in_dim(a, new_shape, broadcast_dimensions)


# Note: saves the Python slice object because we're about to clobber its name with the slice prim
pyslice: Type[slice] = slice  # type: ignore[has-type]


def _slice_meta(
    a: TensorLikeType,
    start_indices: DimsSequenceType,
    limit_indices: DimsSequenceType,
    strides: Optional[StrideType] = None,
) -> TensorLikeType:
    _strides = strides if strides is not None else [1] * len(start_indices)

    if a.ndim != len(start_indices):
        msg = "Attempting to slice tensor of rank {0} with start_indices of length {1}!".format(
            a.ndim, len(start_indices)
        )
        raise ValueError(msg)

    if a.ndim != len(limit_indices):
        msg = "Attempting to slice tensor of rank {0} with limit_indices of length {1}!".format(
            a.ndim, len(limit_indices)
        )
        raise ValueError(msg)

    if a.ndim != len(_strides):
        msg = (
            "Attempting to slice tensor of rank {0} with strides of length {1}!".format(
                a.ndim, len(limit_indices)
            )
        )
        raise ValueError(msg)

    for x, y in zip(start_indices, a.shape):
        if x < 0:
            msg = "Attempting to slice a tensor with a negative start index of {0}!".format(
                x
            )
            raise ValueError(msg)
        if x > y:
            msg = (
                "Attempting to slice a tensor but a start index in {0} is greater than"
                " the length of its corresponding dimension in shape {1}".format(
                    start_indices, a.shape
                )
            )
            raise ValueError(msg)

    for x, y, z in zip(limit_indices, a.shape, start_indices):
        if x < 0:
            msg = "Attempting to slice a tensor with a negative stop index of {0}!".format(
                x
            )
            raise ValueError(msg)
        if x > y:
            msg = (
                "Attempting to slice a tensor but a stop index in {0} is greater than the length of "
                " its corresponding dimension in shape {1}".format(
                    limit_indices, a.shape
                )
            )
            raise ValueError(msg)
        if x < z:
            msg = (
                "Attempting to slice a tensor but a start index in {0} is greater than "
                " its corresponding stop index {1}".format(x, z)
            )

    for x in _strides:
        if x <= 0:
            msg = (
                "Attempting to slice a tensor with a non-positive step of {0}!".format(
                    x
                )
            )
            raise ValueError(msg)

    new_shape = []
    for x, y, z in zip(start_indices, limit_indices, _strides):
        new_shape.append(math.floor((y - x) / z))

    new_strides = []
    for x, y in zip(a.stride(), _strides):
        new_strides.append(x * y)

    return TensorMeta(a, shape=new_shape, strides=new_strides)


def _slice_aten(
    a: Tensor,
    start_indices: DimsSequenceType,
    limit_indices: DimsSequenceType,
    strides: Optional[StrideType] = None,
) -> Tensor:
    _strides = strides if strides is not None else [1] * len(start_indices)

    slices = []
    for start, stop, step in zip(start_indices, limit_indices, _strides):
        slices.append(pyslice(start, stop, step))

    return operator.getitem(a, slices)  # type: ignore[call-overload]


_slice_doc = """
    Creates a view of a "bounding box" within the tensor.

    The bounding box is specified independently in each of the tensor's dimensions.
    start_indices and limit_indices describe the box's boundaries for their corresponding
    dimensions. If strides is specified then they specify the step size between elements
    in their corresponding dimension.

    This operation is analogous to slicing in NumPy, but does not permit slices where
    the stop indices are less than the start indices.
    """

slice = _make_prim(
    schema="slice(Tensor(a) a, int[] start_indices, int[] limit_indices, int[]? strides=None) -> Tensor(a)",
    meta=_slice_meta,
    impl_aten=_slice_aten,
    return_type=RETURN_TYPE.VIEW,
    doc=_slice_doc,
)


def _slice_in_dim_meta(
    a: TensorLikeType,
    start_index: int,
    limit_index: int,
    stride: int = 1,
    axis: int = 0,
) -> TensorLikeType:
    if axis < 0:
        msg = "slice_in_dim: received a negative axis {0}".format(axis)
        raise ValueError(msg)
    if axis >= a.ndim:
        msg = "slice_in_dim: axis {0} is greater or equal to the rank {1} of the tensor".format(
            axis, a.ndim
        )
        raise ValueError(msg)

    if start_index < 0:
        msg = "slice_in_dim: received a negative start_index {0}".format(start_index)
        raise ValueError(msg)

    if start_index > a.shape[axis]:
        msg = "slice_in_dim: start_index is greater than the length {0} of dimension {1}".format(
            start_index, axis
        )
        raise ValueError(msg)

    if limit_index > a.shape[axis]:
        msg = "slice_in_dim: limit_index is greater than the length {0} of dimension {1}".format(
            limit_index, axis
        )
        raise ValueError(msg)

    if limit_index < start_index:
        msg = "slice_in_dim: received a limit_index {0} less than the start_index {1}".format(
            limit_index, start_index
        )
        raise ValueError(msg)

    if stride < 0:
        msg = "slice_in_dim: received a non-positive stride of {0}!".format(stride)
        raise ValueError(msg)

    start_indices = [0] * a.ndim
    limit_indices = list(a.shape)
    strides = [1] * a.ndim

    start_indices[axis] = start_index
    limit_indices[axis] = limit_index
    strides[axis] = stride

    return _slice_meta(a, start_indices, limit_indices, strides)


def _slice_in_dim_aten(
    a: Tensor,
    start_index: int,
    limit_index: int,
    stride: int = 1,
    axis: int = 0,
) -> Tensor:
    start_indices = [0] * a.ndim
    limit_indices = list(a.shape)
    strides = [1] * a.ndim

    start_indices[axis] = start_index
    limit_indices[axis] = limit_index
    strides[axis] = stride

    return slice(a, start_indices, limit_indices, strides)


_slice_in_dim_doc = """
    Convenience wrapper for slicing just one dimension using slice.
    """

slice_in_dim = _make_prim(
    schema="slice_in_dim(Tensor(a) a, int start_index, int limit_index, int stride=1, int axis=0) -> Tensor(a)",
    meta=_slice_in_dim_meta,
    impl_aten=_slice_in_dim_aten,
    return_type=RETURN_TYPE.VIEW,
    doc=_slice_in_dim_doc,
)


def _split_dim_meta(a: TensorLikeType, dim: int, outer_length: int) -> TensorLikeType:
    assert isinstance(a, TensorLike)
    utils.validate_idx(a.ndim, dim)
    utils.validate_dim_length(outer_length)

    # Verifies the dim can be split with the specified lhs_length
    _inner_length = a.shape[dim] / outer_length
    inner_length: int = int(_inner_length)

    if inner_length != _inner_length:
        msg = "Attempting to split dimension of length {0}, but outer length of {1} divides it with a remainder!".format(
            a.shape[dim], outer_length
        )
        raise ValueError(msg)

    new_shape: List[int] = []
    new_strides: List[int] = []
    for idx in range(a.ndim):
        if idx == dim:
            new_shape.extend((outer_length, inner_length))
            new_strides.extend((a.stride()[idx] * inner_length, a.stride()[idx]))
        else:
            new_shape.append(a.shape[idx])
            new_strides.append(a.stride()[idx])

    return TensorMeta(a, shape=new_shape, strides=new_strides)


def _split_dim_aten(a: Tensor, dim: int, outer_length: int) -> Tensor:
    inner_length = int(a.shape[dim] / outer_length)
    new_shape = a.shape[0:dim] + (outer_length, inner_length) + a.shape[dim + 1 :]

    return a.view(new_shape)


_split_dim_doc = """
  Creates a view of a with the given dimension (of length l) split
  into two dimensions, with the outer of the two having
  length outer_length and the inner of the two having computed
  length inner_length such outer_length * inner_length = l.
  """

# TODO: consider renaming split_dim_view
split_dim = _make_prim(
    schema="split_dim(Tensor(a) a, int dim, int outer_length) -> Tensor(a)",
    meta=_split_dim_meta,
    impl_aten=_split_dim_aten,
    return_type=RETURN_TYPE.VIEW,
    doc=_split_dim_doc,
)

# Note: allows dimensions to be specified redundantly
def _squeeze_meta(a: TensorLikeType, dimensions: Sequence) -> TensorLikeType:
    assert isinstance(a, TensorLike)

    for idx in dimensions:
        utils.validate_idx(a.ndim, idx)
        assert a.shape[idx] == 1

    new_shape = []
    new_strides = []
    for idx in range(len(a.shape)):
        if idx in dimensions:
            continue

        new_shape.append(a.shape[idx])
        new_strides.append(a.stride()[idx])

    return TensorMeta(a, shape=new_shape, strides=new_strides)


def _squeeze_aten(a: Tensor, dimensions: Sequence) -> Tensor:
    squeezes = 0
    for idx in dimensions:
        a = torch.squeeze(a, dim=(idx - squeezes))
        squeezes = squeezes + 1

    return a


_squeeze_doc = """
  Creates a view of the tensor with the specified dimensions removed.

  The removed dimensions must each have length one.
  """

squeeze = _make_prim(
    schema="squeeze(Tensor(a) a, int[] dimensions) -> Tensor(a)",
    meta=_squeeze_meta,
    impl_aten=_squeeze_aten,
    return_type=RETURN_TYPE.VIEW,
    doc=_squeeze_doc,
)


def _transpose_meta(a: TensorLikeType, permutation: DimsSequenceType) -> TensorLikeType:
    if a.ndim != len(permutation):
        msg = "Attempting to permute a tensor of rank {0}, but received a permutation of length {1}!".format(
            a.ndim, len(permutation)
        )
        raise ValueError(msg)

    if not utils.is_valid_permutation(a.ndim, permutation):
        msg = "Received an invalid permutation, {0}!".format(permutation)
        raise ValueError(msg)

    new_shape = [0] * a.ndim
    new_strides = [0] * a.ndim
    for idx, dim in enumerate(permutation):
        new_shape[idx] = a.shape[dim]
        new_strides[idx] = a.stride()[dim]

    return TensorMeta(a, shape=tuple(new_shape), strides=tuple(new_strides))


def _transpose_aten(a: Tensor, permutation: DimsSequenceType) -> Tensor:
    return torch.permute(a, permutation)


_transpose_doc = """
    Creates a view of the tensor with its dimensions permuted.

    The length of the permutation must be the rank of the tensor,
    and each element of the permutation specifies the new order
    for the corresponding dimension.
    """

transpose = _make_prim(
    schema="transpose(Tensor(a) a, int[] permutation) -> Tensor(a)",
    meta=_transpose_meta,
    impl_aten=_transpose_aten,
    return_type=RETURN_TYPE.VIEW,
    doc=_transpose_doc,
)


def _view_of_meta(a: TensorLikeType) -> TensorLikeType:
    return TensorMeta(a)


def _view_of_aten(a: Tensor) -> Tensor:
    return a.view(a.shape)


_view_of_doc = """
    Creates a view of the tensor.
    """

view_of = _make_prim(
    schema="view_of(Tensor(a) a) -> Tensor",
    meta=_view_of_meta,
    impl_aten=_view_of_aten,
    return_type=RETURN_TYPE.VIEW,
    doc=_view_of_doc,
)

#
# Shape operations
#
def collapse(a: Tensor, start: int, end: int) -> Tensor:
    """
    Wrapper around reshape that collapses a span of dimensions.

    See collapse_view for the corresponding view operation.
    """

    dim_length = 1
    for idx in range(start, end):
        dim_length = dim_length * a.shape[idx]

    new_shape = a.shape[0:start] + (dim_length,) + a.shape[end:]
    return reshape(a, new_shape)


# TODO: review stride logic
def _cat_meta(tensors: Sequence[TensorLikeType], dim: int) -> TensorLikeType:
    # Verifies same shape (except in the concat dimension)
    shape = tensors[0].shape
    concat_length = 0
    for tensor in tensors:
        for idx, (common_length, length) in enumerate(zip(shape, tensor.shape)):
            if idx == dim:
                concat_length = concat_length + length
            else:
                assert length == common_length

    new_shape = list(tensors[0].shape).copy()
    new_shape[dim] = concat_length
    return TensorMeta(
        tensors[0],
        shape=new_shape,
        strides=utils.make_contiguous_strides_for(new_shape),
    )


def _cat_aten(tensors: Union[Tuple[Tensor, ...], List[Tensor]], dim: int) -> Tensor:
    return torch.cat(tensors, dim)


_cat_doc = """
  Concatenates tensors along the specified dimension.

  The tensors' shapes must have the same rank and same length for other dimensions.
  """

cat = _make_prim(
    schema="cat(Tensor[] tensors, int dim) -> Tensor",
    meta=_cat_meta,
    impl_aten=_cat_aten,
    return_type=RETURN_TYPE.NEW,
    doc=_cat_doc,
)


def _reshape_meta(a: TensorLikeType, shape: ShapeType):
    assert isinstance(a, TensorLike)
    utils.validate_shape(shape)

    # Validates the tensor and the requested shape have the
    # same number of elements
    numel = reduce(operator.mul, shape)
    if numel != a.numel():
        msg = "Attempting to reshape a tensor with {0} elements to a shape with {1} elements!".format(
            a.numel(), numel
        )
        raise ValueError(msg)

    return TensorMeta(a, shape=shape, strides=utils.make_contiguous_strides_for(shape))


def _reshape_aten(a: Tensor, shape: ShapeType) -> Tensor:
    return a.reshape(shape).contiguous().clone()


_reshape_doc = """
  Creates a contiguous tensor with the specified shape
  containing a copy of the data in a.
  """
reshape = _make_prim(
    schema="reshape(Tensor a, int[] shape) -> Tensor",
    meta=_reshape_meta,
    impl_aten=_reshape_aten,
    return_type=RETURN_TYPE.NEW,
    doc=_reshape_doc,
)


def _rev_meta(a: TensorLikeType, dims: DimsSequenceType) -> TensorLikeType:
    utils.validate_dimension_indices(a.ndim, dims)
    return TensorMeta(a)


_rev_doc = """
    Reverses the order of elements along the given dimensions.
    """

rev = _make_prim(
    schema="rev(Tensor a, int[] dims) -> Tensor",
    meta=_rev_meta,
    impl_aten=torch.flip,
    return_type=RETURN_TYPE.NEW,
    doc=_rev_doc,
)

#
# Conditional prims
#


def _where_meta(
    pred: TensorLikeType, a: TensorLikeType, b: TensorLikeType
) -> TensorLikeType:

    return _elementwise_meta(
        a,
        b,
        type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
        args_with_fixed_dtypes=(pred,),
    )


_where_doc = """
  Selects elements from a and b according to pred.

  Where pred is true the result contains the element from a, and
  where pred is false the result contains the element from b.
  """

where = _make_prim(
    schema="where(Tensor pred, Tensor a, Tensor b) -> Tensor",
    meta=_where_meta,
    impl_aten=torch.where,
    impl_nvfuser=_where_nvfuser,  # type: ignore[name-defined]
    return_type=RETURN_TYPE.NEW,
    doc=_where_doc,
)

#
# Type conversions
#
# TODO: model memory format on TensorMeta
# TODO: make clone a reference following its implementation in TensorFactories.cpp
def _clone_meta(
    a: TensorLikeType, *, memory_format: torch.memory_format
) -> TensorLikeType:
    strides = utils.compute_elementwise_output_strides(a)
    return TensorMeta(a, strides=strides)


def _clone_aten(a: Tensor, *, memory_format: torch.memory_format) -> Tensor:
    return torch.clone(a, memory_format=memory_format)


_clone_doc = """
    Creates a copy of a tensor.
"""

clone = _make_prim(
    schema="clone(Tensor a, *, MemoryFormat memory_format) -> Tensor",
    meta=_clone_meta,
    impl_aten=_clone_aten,
    return_type=RETURN_TYPE.NEW,
    doc=_clone_doc,
)


def _convert_element_type_meta(a: TensorLikeType, dtype: torch.dtype) -> TensorLikeType:
    # Type checks
    assert isinstance(a, TensorLike)
    assert isinstance(dtype, torch.dtype)

    strides = utils.compute_elementwise_output_strides(a)

    return TensorMeta(a, strides=strides, dtype=dtype)


def _convert_element_type_aten(a: Tensor, dtype: torch.dtype) -> Tensor:

    # Propagates requires grad when possible
    if not utils.is_grad_dtype(dtype):
        requires_grad = False
    else:
        # TODO: update meta objects so this can be acquired directly
        try:
            requires_grad = a.requires_grad
        except Exception as e:
            requires_grad = False

    result = torch.empty_like(
        a, device=a.device, dtype=dtype, requires_grad=requires_grad
    )
    with torch.no_grad():
        return copy_to(result, a)


def _convert_element_type_nvfuser(fd: Any, a: Tensor, dtype: torch.dtype) -> Tensor:
    nvfuser_dtype = getnvFuserDtype(dtype)
    return fd.Ops.cast(nvfuser_dtype, a)  # type: ignore[attr-defined]


_convert_element_type_doc = """
  Creates a copy of a tensor with the given dtype.
  """

convert_element_type = _make_prim(
    schema="convert_element_type(Tensor a, ScalarType dtype) -> Tensor",
    meta=_convert_element_type_meta,
    impl_aten=_convert_element_type_aten,
    impl_nvfuser=_convert_element_type_nvfuser,
    return_type=RETURN_TYPE.NEW,
    doc=_convert_element_type_doc,
)


def _device_put_meta(
    a: TensorLikeType, device: Union[str, torch.device]
) -> TensorLikeType:
    assert isinstance(a, TensorLike)
    assert isinstance(device, (str, torch.device))

    return TensorMeta(a, device=utils.canonicalize_device(device))


def _device_put_aten(a: Tensor, device: Union[str, torch.device]) -> Tensor:
    return a.to(device)


_device_put_doc = """
  Creates a copy of a tensor on the given device.
  """

device_put = _make_prim(
    schema="device_put(Tensor a, Device device) -> Tensor",
    meta=_device_put_meta,
    impl_aten=_device_put_aten,
    return_type=RETURN_TYPE.NEW,
    doc=_device_put_doc,
)

# NOTE: need to model meta scalars
# See https://github.com/pytorch/pytorch/issues/78070
def _item_meta(a: TensorLikeType) -> FakeTensor:
    number_type = utils.dtype_to_type(a.dtype)
    return TensorMeta(number_type(-1))


_item_doc = """
    Converts a tensor with one element to a Python number.
"""

# TODO: create a new return type for scalars?
# FIXME: currently returns integers for boolean tensors
# https://github.com/pytorch/pytorch/issues/78071
item = _make_prim(
    schema="item(Tensor a) -> Scalar",
    meta=_item_meta,
    impl_aten=torch.Tensor.item,
    return_type=RETURN_TYPE.NEW,
    doc=_item_doc,
)

# NOTE: need to model meta scalars
# See https://github.com/pytorch/pytorch/issues/78070
def _maximum_value_meta(dtype: torch.dtype) -> FakeTensor:
    number_type = utils.dtype_to_type(dtype)
    return TensorMeta(number_type(-1))


def _maximum_value_aten(dtype: torch.dtype):
    if dtype == torch.bool:
        return True
    elif dtype.is_complex or dtype.is_floating_point:
        return torch.finfo(dtype).max
    else:
        return torch.iinfo(dtype).max


_maximum_value_doc = """
    Return the maximum finite value for a dtype.
"""

# TODO: create a new return type for scalars?
# FIXME: currently returns integers for boolean tensors
# https://github.com/pytorch/pytorch/issues/78071
maximum_value = _make_prim(
    schema="maximum_value(ScalarType dtype) -> Scalar",
    meta=_maximum_value_meta,
    impl_aten=_maximum_value_aten,
    return_type=RETURN_TYPE.NEW,
    doc=_maximum_value_doc,
)


# NOTE: need to model meta scalars
# See https://github.com/pytorch/pytorch/issues/78070
def _minimum_value_meta(dtype: torch.dtype) -> FakeTensor:
    number_type = utils.dtype_to_type(dtype)
    return TensorMeta(number_type(-1))


def _minimum_value_aten(dtype: torch.dtype):
    if dtype == torch.bool:
        return False
    elif dtype.is_complex or dtype.is_floating_point:
        return torch.finfo(dtype).min
    else:
        return torch.iinfo(dtype).min


_minimum_value_doc = """
    Return the mimimum finite value for a dtype.
"""

# TODO: create a new return type for scalars?
# FIXME: currently returns integers for boolean tensors
# https://github.com/pytorch/pytorch/issues/78071
minimum_value = _make_prim(
    schema="minium_value(ScalarType dtype) -> Scalar",
    meta=_minimum_value_meta,
    impl_aten=_minimum_value_aten,
    return_type=RETURN_TYPE.NEW,
    doc=_minimum_value_doc,
)

# TODO: FIXME: strides are incorrect
def _to_dtype_meta(a: TensorLikeType, dtype: torch.dtype) -> TensorLikeType:
    strides = utils.make_contiguous_strides_for(a.shape)
    return TensorMeta(a, strides=strides, dtype=dtype)


def _to_dtype_aten(a: Tensor, dtype: torch.dtype) -> Tensor:
    return a.to(dtype)


_to_dtype_doc = """
    Creates a contiguous copy of a tensor with the given dtype.
"""

to_dtype = _make_prim(
    schema=("to_dtype(Tensor a, ScalarType dtype) -> Tensor"),
    meta=_to_dtype_meta,
    impl_aten=_to_dtype_aten,
    return_type=RETURN_TYPE.NEW,
    doc=_to_dtype_doc,
)

#
# Inplace operators
#


def _copy_to_meta(a: TensorLikeType, b: TensorLikeType):
    assert isinstance(a, TensorLike)
    assert isinstance(b, TensorLike)

    # Validates the cast is safe
    # TODO: move this as an option on the reference
    # a_typ = utils.dtype_to_type(a.dtype)
    # b_typ = utils.dtype_to_type(b.dtype)
    # if a_typ is not utils.get_higher_type(a_typ, b_typ):
    #     raise RuntimeError(str(b.dtype), " can't be cast safely to ", str(a.dtype), "!")

    # Validates the tensors have the same number of elements
    if a.numel() != b.numel():
        msg = "Attempting to copy {0} elements to a tensor with {1} elements!".format(
            b.numel(), a.numel()
        )
        raise RuntimeError(msg)

    return a


def _copy_to_aten(a: Tensor, b: Tensor) -> Tensor:
    return a.copy_(b)


_copy_to_doc = """
  Copies the data in b to a and returns the modified a.
  """

# TODO: Remove safe casting and implement on reference instead
copy_to = _make_prim(
    schema="copy_to(Tensor(a!) a, Tensor b) -> Tensor(a!)",
    meta=_copy_to_meta,
    impl_aten=_copy_to_aten,
    return_type=RETURN_TYPE.INPLACE,
    doc=_copy_to_doc,
)


def _resize_meta(a: TensorLikeType, shape: ShapeType):
    return a.resize_(shape)


def _resize_aten(a: Tensor, shape: ShapeType) -> Tensor:
    return a.resize_(shape)


_resize_doc = """
  Gives a tensor with no elements a new shape, returning the modified tensor.

  The tensor's strides are contiguous and its values are unitialized.
  """

# TODO: review support arbitrary resizes
resize = _make_prim(
    schema="resize(Tensor(a!) a, int[] shape) -> Tensor(a!)",
    meta=_resize_meta,
    impl_aten=_resize_aten,
    return_type=RETURN_TYPE.INPLACE,
    doc=_resize_doc,
)


def _reduction_meta(inp, dims, *, output_dtype=None):
    """
    Meta function for single output reduction operations
    Stride logic is incorrect
    """
    assert isinstance(inp, TensorLike)
    if output_dtype is None:
        output_dtype = inp.dtype
    output_shape = utils.compute_reduction_output_shape(inp.shape, dims)
    return TensorMeta(
        shape=output_shape,
        strides=utils.make_contiguous_strides_for(output_shape),
        dtype=output_dtype,
        device=inp.device,
    )


def _var_reduction_meta(inp, dims, *, correction):
    if utils.is_complex_dtype(inp.dtype):
        output_dtype = utils.corresponding_real_dtype(inp.dtype)
    else:
        output_dtype = inp.dtype
    return _reduction_meta(inp, dims, output_dtype=output_dtype)


_sum_doc = """
    Computes the sum of elements in the input tensor over the list of dimensions
    specified in the dim argument
    """
_prod_doc = """
    Computes the product of elements in the input tensor over the list of dimensions
    specified in the dim argument
    """
_amax_doc = """
    Computes the maximum value of elements in the input tensor over the list of dimensions
    specified in the dim argument
    """
_amin_doc = """
    Computes the minimum value of elements in the input tensor over the list of dimensions
    specified in the dim argument
    """
_var_doc = """
    Computes the biased variance of x over the list of dimensions specified in the dim argument
    """


def _make_reduction_prim(name: str, impl_aten, doc, impl_nvfuser=None):
    """Creates a reduction prim."""
    return _make_prim(
        schema=f"{name}(Tensor inp, int[]? dims, *, ScalarType? output_dtype=None) -> Tensor",
        meta=_reduction_meta,
        impl_aten=impl_aten,
        impl_nvfuser=impl_nvfuser,
        return_type=RETURN_TYPE.NEW,
        doc=doc,
    )


def _make_var_reduction_prim(name: str, impl_aten, doc, impl_nvfuser):
    """Creates a reduction prim."""
    return _make_prim(
        schema=f"{name}(Tensor inp, int[]? dims, *, int correction, ScalarType? output_dtype=None) -> Tensor",
        meta=_var_reduction_meta,
        impl_aten=impl_aten,
        impl_nvfuser=impl_nvfuser,
        return_type=RETURN_TYPE.NEW,
        doc=doc,
    )


def _sum_nvfuser(
    fd: Any,
    a: TensorLikeType,
    dims: DimsSequenceType,
):
    keep_dims = False
    output_dtype = torch._C._nvfuser.DataType.Null
    return fd.Ops.sum(a, dims, keep_dims, output_dtype)


sum = _make_reduction_prim(
    name="sum",
    impl_aten=torch.sum,
    impl_nvfuser=_sum_nvfuser,
    doc=_sum_doc,
)


def _prod_aten(
    inp: TensorLikeType,
    dims: Optional[DimsSequenceType],
    *,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    if dims is not None:
        for d in sorted(dims, reverse=True):
            assert d >= 0
            inp = torch.prod(inp, d, dtype=dtype)
        return inp
    else:
        return torch.prod(inp, dims, dtype=dtype)


prod = _make_reduction_prim(
    name="prod",
    impl_aten=_prod_aten,
    doc=_prod_doc,
)


def _var_nvfuser(
    fd: Any,
    a: TensorLikeType,
    dims: DimsSequenceType,
    *,
    correction: int,
):
    keep_dims = False
    return fd.Ops.var(a, dims, correction, keep_dims)


def _amax_nvfuser(
    fd: Any,
    a: TensorLikeType,
    dims: DimsSequenceType,
):
    keep_dims = False
    return fd.Ops.max(a, dims, keep_dims)


def _amin_nvfuser(
    fd: Any,
    a: TensorLikeType,
    dims: DimsSequenceType,
):
    keep_dims = False
    return fd.Ops.min(a, dims, keep_dims)


var = _make_var_reduction_prim(
    name="var",
    impl_aten=torch.var,
    impl_nvfuser=_var_nvfuser,
    doc=_var_doc,
)

amax = _make_reduction_prim(
    name="amax",
    impl_aten=torch.amax,
    impl_nvfuser=_amax_nvfuser,
    doc=_amax_doc,
)

amin = _make_reduction_prim(
    name="amin",
    impl_aten=torch.amin,
    impl_nvfuser=_amin_nvfuser,
    doc=_amin_doc,
)

# TODO: layout, pin_memory, memory_format
# TODO: model requires_grad on TensorMeta
def _empty_meta(
    shape: ShapeType, *, dtype: torch.dtype, device: torch.device, requires_grad: bool
) -> TensorLikeType:
    strides = utils.make_contiguous_strides_for(shape)
    return TensorMeta(shape=shape, strides=strides, dtype=dtype, device=device)


def _empty_aten(
    shape: ShapeType, *, dtype: torch.dtype, device: torch.device, requires_grad: bool
) -> Tensor:
    return torch.empty(shape, dtype=dtype, device=device, requires_grad=requires_grad)


_empty_doc = """
    Creates a tensor with uninitialized values and the specified shape, dtype, and device.
"""

empty = _make_prim(
    schema="empty(int[] shape, *, ScalarType dtype, Device device, bool requires_grad) -> Tensor",
    meta=_empty_meta,
    impl_aten=_empty_aten,
    return_type=RETURN_TYPE.NEW,
    doc=_empty_doc,
)


def _empty_strided_meta(
    shape: ShapeType,
    strides: StrideType,
    *,
    dtype: torch.dtype,
    device: torch.device,
    requires_grad: bool,
) -> TensorLikeType:
    return TensorMeta(shape=shape, strides=strides, dtype=dtype, device=device)


_empty_strided_doc = """
    Creates a tensor with uninitialized values.
"""

# TODO: add layout, pin_memory
empty_strided = _make_prim(
    schema="empty_strided(int[] shape, int[] strides, *, ScalarType dtype, Device device, bool requires_grad) -> Tensor",
    return_type=RETURN_TYPE.NEW,
    meta=_empty_strided_meta,
    impl_aten=torch.empty_strided,
    doc=_empty_strided_doc,
)


def _full_meta(
    shape: ShapeType,
    fill_value: NumberType,
    *,
    dtype: torch.dtype,
    device: torch.device,
    requires_grad: bool,
) -> TensorLikeType:
    strides = utils.make_contiguous_strides_for(shape)
    return TensorMeta(shape=shape, strides=strides, dtype=dtype, device=device)


def _full_aten(
    shape: ShapeType,
    fill_value: NumberType,
    *,
    dtype: torch.dtype,
    device: torch.device,
    requires_grad: bool,
) -> Tensor:
    # Note that Mypy thinks torch.full can't accept a complex fill_value
    return torch.full(
        shape, fill_value, dtype=dtype, device=device, requires_grad=requires_grad  # type: ignore[arg-type]
    )


_full_doc = """
    Creates a tensor filled with the given fill value, and with the specified shape, dtype, and device.
"""

# TODO: add layout
full = _make_prim(
    schema="full(int[] shape, Scalar fill_value, *, ScalarType dtype, Device device, bool requires_grad) -> Tensor",
    meta=_full_meta,
    impl_aten=_full_aten,
    return_type=RETURN_TYPE.NEW,
    doc=_full_doc,
)


def _full_like_meta(
    a: TensorLikeType,
    fill_value: NumberType,
    *,
    dtype: torch.dtype,
    device: torch.device,
    requires_grad: bool,
) -> TensorLikeType:
    strides = strides = utils.compute_elementwise_output_strides(a)
    if a.numel() == 0:
        strides = a.stride()

    return TensorMeta(a, strides=strides, dtype=dtype, device=device)


def _full_like_aten(
    a: Tensor,
    fill_value: NumberType,
    *,
    dtype: torch.dtype,
    device: torch.device,
    requires_grad: bool,
) -> Tensor:
    # Note that Mypy thinks torch.full can't accept a complex fill_value
    return torch.full_like(
        a, fill_value, dtype=dtype, device=device, requires_grad=requires_grad  # type: ignore[arg-type]
    )


_full_like_doc = """
    Creates a tensor filled with the given fill value, and the same shape, dtype, and device as the
    given tensor by default. The dtype and device settings can be overridden
    by specifying them explicitly.
"""

full_like = _make_prim(
    schema="full_like(Tensor a, Scalar fill_value, *, ScalarType dtype, Device device, bool requires_grad) -> Tensor",
    meta=_full_like_meta,
    impl_aten=_full_like_aten,
    return_type=RETURN_TYPE.NEW,
    doc=_full_like_doc,
)


def _scalar_tensor_meta(
    scalar: NumberType,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> TensorLikeType:
    shape: ShapeType = []
    strides = utils.make_contiguous_strides_for(shape)
    return TensorMeta(scalar, shape=shape, strides=strides, dtype=dtype, device=device)


def _scalar_tensor_aten(
    scalar: NumberType,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    if isinstance(scalar, complex) and (
        dtype is None or not utils.is_complex_dtype(dtype)
    ):
        raise TypeError("Complex scalar requires complex tensor dtype.")
    # Note that Mypy thinks torch.scalar can't accept a complex scalar
    return torch.scalar_tensor(scalar, dtype=dtype, device=device)  # type: ignore[arg-type]


_scalar_tensor_doc = """
    Wraps a Number into a Tensor with the specified dtype and device.
"""

# TODO: add layout and pin_memory support
scalar_tensor = _make_prim(
    schema="scalar_tensor(Scalar s, *, ScalarType? dtype=None, Device? device=None) -> Tensor",
    meta=_scalar_tensor_meta,
    impl_aten=_scalar_tensor_aten,
    return_type=RETURN_TYPE.NEW,
    doc=_scalar_tensor_doc,
)

#
# Randomness Prims
#


def _uniform_meta(
    shape: ShapeType,
    *,
    low: float,
    high: float,
    dtype: torch.dtype,
    device: torch.device,
) -> TensorLikeType:
    strides = utils.make_contiguous_strides_for(shape)
    return TensorMeta(shape=shape, strides=strides, dtype=dtype, device=device)


def _uniform_aten(
    shape: ShapeType,
    *,
    low: float,
    high: float,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    a = torch.empty(shape, dtype=dtype, device=device)
    a.uniform_(low, high)
    return a


_uniform_doc = """
    Constructs a tensor filled with values drawn uniformly from low to high.
"""

# TODO: we should more seriously review randomness modeling and prims
uniform = _make_prim(
    schema=(
        "uniform(int[] shape, *, Scalar low, Scalar high, ScalarType dtype, Device device) -> Tensor"
    ),
    return_type=RETURN_TYPE.NEW,
    meta=_uniform_meta,
    impl_aten=_uniform_aten,
    doc=_uniform_doc,
)
