# mypy: allow-untyped-defs
import operator
from collections.abc import Callable, Sequence
from enum import Enum
from functools import partial, reduce
from typing import Optional, Union

import torch
import torch._prims_common as utils
import torch.library
from torch import sym_float, Tensor
from torch._C import _get_default_device
from torch._higher_order_ops.effects import new_token_tensor
from torch._library.utils import is_functional_schema
from torch._prims.debug_prims import register_debug_prims
from torch._prims.rng_prims import register_rng_prims
from torch._prims_common import (
    Dim,
    DimsSequenceType,
    DimsType,
    IntLike,
    Number,
    NumberType,
    RETURN_TYPE,
    ShapeType,
    StrideType,
    TensorLike,
    TensorLikeType,
    type_to_dtype,
)
from torch._prims_common.wrappers import backwards_not_supported
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.overrides import handle_torch_function, has_torch_function
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten


prim = torch.library.Library("prims", "DEF")
prim_impl = torch.library.Library("prims", "IMPL", "CompositeExplicitAutograd")
prim_backend_select_impl = torch.library.Library("prims", "IMPL", "BackendSelect")
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
    "bessel_j0",
    "bessel_j1",
    "bitwise_not",
    "cbrt",
    "ceil",
    "conj_physical",
    "digamma",
    "erf",
    "erf_inv",
    "erfc",
    "erfcx",
    "exp",
    "expm1",
    "exp2",
    "fill",
    "floor",
    "imag",
    "isfinite",
    "lgamma",
    "log",
    "log1p",
    "log2",
    "log10",
    "ndtri",
    "neg",
    "real",
    "reciprocal",
    "round",
    "sign",
    "signbit",
    "sin",
    "sinh",
    "spherical_bessel_j0",
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
    "frexp",
    "gcd",
    "ge",
    "gt",
    "hypot",
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
    "conj",
    "expand_dims",
    "slice",
    "split_dim",
    "squeeze",
    "transpose",
    "view_of",
    "view_element_type",
    #
    # Functionalized view mutations
    #
    "as_strided_scatter",
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
    "copy_strided",
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
    "xor_sum",
    "var",
    #
    # Tensor Creation Prims
    #
    "empty_strided",
    "empty_permuted",
    "scalar_tensor",
    "iota",
    #
    # Linear algebra (linalg) Prims
    #
    "svd",
    #
    # Randomness Prims
    #
    "normal",
    "_uniform_helper",
    #
    # FFT prims
    #
    "fft_r2c",
    "fft_c2c",
    "fft_c2r",
    #
    # prims for making/sinking tokens
    #
    "_make_token",
    "_sink_tokens",
]


def TensorMeta(
    tensorlike: NumberType | torch.Tensor | None = None,
    *,
    shape: ShapeType | None = None,
    strides: StrideType | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
):
    if isinstance(tensorlike, Number):
        if shape and not isinstance(shape, Sequence):
            raise AssertionError(
                f"shape must be None or a Sequence for Number input, got {type(shape)}"
            )
        if strides and not isinstance(strides, Sequence):
            raise AssertionError(
                f"strides must be None or a Sequence for Number input, got {type(strides)}"
            )
        inferred_shape: tuple[int, ...] = ()
        inferred_strides: tuple[int, ...] = ()
        inferred_dtype = type_to_dtype(type(tensorlike))
        inferred_device = torch.device("cpu")
        # TODO: This looks wrong, a number that is wrapped into a tensor
        # needs to behave differently than a scalar tensor for type
        # promotion purposes
    elif tensorlike is not None:
        if not isinstance(tensorlike, torch.Tensor):
            raise AssertionError(
                f"tensorlike must be torch.Tensor, got {type(tensorlike)}"
            )  # mypy
        inferred_shape = tuple(tensorlike.shape)
        inferred_strides = tuple(tensorlike.stride())
        inferred_dtype = tensorlike.dtype
        inferred_device = tensorlike.device
    else:
        # If no tensorlike "example" is given then all metadata
        # must be provided explicitly
        if shape is None:
            raise AssertionError("shape must be provided when tensorlike is None")
        if strides is None:
            raise AssertionError("strides must be provided when tensorlike is None")
        if dtype is None:
            raise AssertionError("dtype must be provided when tensorlike is None")
        if device is None:
            raise AssertionError("device must be provided when tensorlike is None")

    shape = inferred_shape if shape is None else tuple(shape)  # type: ignore[possibly-undefined]
    strides = inferred_strides if strides is None else tuple(strides)  # type: ignore[possibly-undefined]
    dtype = inferred_dtype if dtype is None else dtype  # type: ignore[possibly-undefined]
    device = inferred_device if device is None else device  # type: ignore[possibly-undefined]

    if isinstance(device, str):
        device = torch.device(device)

    return torch.empty_strided(shape, strides, dtype=dtype, device=device)


def _make_prim(
    *,
    schema: str,
    return_type: RETURN_TYPE | tuple[RETURN_TYPE, ...],
    meta: Callable,
    impl_aten: Callable,
    doc: str,
    tags: Sequence[torch.Tag] | None = None,
    use_old_custom_ops_api: bool = False,
    register_conj_neg_fallthrough: bool = False,
):
    """
    Creates a primitive operation.

    """

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
    def _autograd_impl(*args, **kwargs):
        return backwards_not_supported(_prim)(*args, **kwargs)

    def _backend_select_impl(*args, **kwargs):
        if kwargs.get("device") and kwargs["device"].type == "meta":
            return meta(*args, **kwargs)
        if any(isinstance(x, torch.device) and x.type == "meta" for x in args):
            return meta(*args, **kwargs)
        else:
            return _prim_impl(*args, **kwargs)

    name = schema.split("(", maxsplit=1)[0]
    schema = schema[len(name) :]

    # register non-functional ops with old custom ops API
    cpp_schema = torch._C.parse_schema(name + schema)
    if use_old_custom_ops_api or not is_functional_schema(cpp_schema):
        prim.define(name + schema, tags=torch.Tag.pt2_compliant_tag)
        prim_impl.impl(name, _prim_impl)
        prim_autograd_impl.impl(name, _autograd_impl)
        prim_meta_impl.impl(name, meta)
    else:
        mutates_args = [
            arg.name
            for arg in cpp_schema.arguments
            if arg.alias_info is not None and arg.alias_info.is_write
        ]
        prim_def = torch.library.custom_op(
            "prims::" + name,
            _prim_impl,
            mutates_args=tuple(mutates_args),
            schema=schema,
        )
        prim_def.register_fake(meta)

        # all view ops get conj/neg fallthroughs
        if return_type == RETURN_TYPE.VIEW or register_conj_neg_fallthrough:
            prim_def._lib.impl(name, torch.library.fallthrough_kernel, "Conjugate")
            prim_def._lib.impl(name, torch.library.fallthrough_kernel, "Negative")

    _prim_packet = getattr(torch._ops.ops.prims, name)
    _prim = _prim_packet.default
    if tags:
        _prim._tags = tags
    elif aten_packet := getattr(torch.ops.aten, name, None):
        overload_tags = [
            getattr(aten_packet, overload).tags for overload in aten_packet.overloads()
        ]
        tags_intersection = set(overload_tags[0])
        tags_intersection.intersection_update(*overload_tags[1:])

        # dont inadvertently add to prim ops
        tags_intersection.discard(torch.Tag.core)
        # causes errors with python ref executor tests, none of the
        # data dependent pytorch ops actually decompose to prims
        tags_intersection.discard(torch.Tag.data_dependent_output)

        # iter over first tags for determinism
        _prim._tags = tuple(t for t in overload_tags[0] if t in tags_intersection)

    from torch._subclasses.fake_tensor import contains_tensor_types

    if (
        not any(contains_tensor_types(a.type) for a in _prim._schema.arguments)
        or str(
            _prim
            # See https://github.com/pytorch/pytorch/issues/103532
        )
        == "prims.device_put.default"
    ):
        prim_backend_select_impl.impl(name, _backend_select_impl)

    for p in (_prim_packet, _prim):
        p.__doc__ = doc
        p.return_type = return_type  # type: ignore[attr-defined]

        p.schema = schema
        p.prim_impl = _prim_impl
        p.prim_meta_impl = meta
        p.impl_aten = impl_aten

    return _prim


class ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND(Enum):
    DEFAULT = (0,)
    INT_TO_FLOAT = (2,)
    ALWAYS_BOOL = (3,)
    COMPLEX_TO_FLOAT = (4,)


# TODO: implement dtype validation here, too, or on the corresponding refs
def _prim_elementwise_meta(
    *args,
    type_promotion: ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND,
    args_with_fixed_dtypes: tuple[TensorLikeType, ...] | None = None,
) -> FakeTensor:
    """
    Meta function for elementwise operations that produce outputs in the same dtype
    as their inputs.

    Stride logic is currently incorrect.
    """

    if len(args) == 0:
        raise AssertionError("elementwise operation requires at least one argument")

    utils.check_same_dtype(*args)

    args_ = list(args)
    if args_with_fixed_dtypes is not None:
        args_ = list(args_with_fixed_dtypes) + args_

    utils.check_same_device(*args_, allow_cpu_scalar_tensors=True)
    utils.check_same_shape(*args_, allow_cpu_scalar_tensors=True)

    l2p_perm, _ = utils.compute_elementwise_output_logical_to_physical_perm(*args_)
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
    # pyrefly: ignore [bad-assignment]
    for arg in args_:
        if isinstance(arg, TensorLike):
            if utils.is_cpu_scalar_tensor(arg):
                if device is None:
                    device = arg.device
                # keep going, in case there is a cuda tensor later
            else:
                device = arg.device
                break

        elif isinstance(arg, Number):
            if number is None:
                number = arg

    # NOTE: type promotion behavior here is mostly hidden from tests because
    # references will typically handle the type promotion properly even if this doesn't
    # (but getting it wrong will cause too many casts to be inserted in traces!)
    if device is not None:
        if dtype is None:
            raise AssertionError("dtype must not be None when device is not None")
        if type_promotion == ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.ALWAYS_BOOL:
            dtype = torch.bool
        elif type_promotion == ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.INT_TO_FLOAT:
            if utils.is_integer_dtype(dtype) or utils.is_boolean_dtype(dtype):
                dtype = torch.get_default_dtype()
        elif type_promotion == ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.COMPLEX_TO_FLOAT:
            if utils.is_complex_dtype(dtype):
                dtype = utils.corresponding_real_dtype(dtype)

        if shape is None:
            raise AssertionError("shape must not be None when device is not None")
        return torch.empty_permuted(shape, l2p_perm, device=device, dtype=dtype)  # type: ignore[return-value]

    # Number case
    # TODO: fix number type promotion (bool, complex->float)

    # For now for symint/float, just implementing the common / simple cases of (int,float,symint,symfloat)
    seen_float = False
    if isinstance(number, (torch.SymInt, torch.SymFloat)):
        for a in args:
            if not isinstance(a, (int, float, torch.SymInt, torch.SymFloat)):
                raise AssertionError(
                    f"Expected int, float, SymInt, or SymFloat, got {type(a)}"
                )
            seen_float = seen_float or isinstance(a, (float, torch.SymFloat))
        if seen_float:
            number = sym_float(number)

    return TensorMeta(number)  # type: ignore[arg-type]


def _complex_only_elementwise_meta(*args, **kwargs):
    torch._check(
        utils.is_complex_dtype(args[0].dtype), lambda: "Only complex dtype is supported"
    )
    return _prim_elementwise_meta(*args, **kwargs)


def _make_elementwise_unary_prim(
    name: str, *, type_promotion: ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND, **kwargs
):
    """
    Creates an elementwise unary prim.
    """

    return _make_prim(
        schema=f"{name}(Tensor self) -> Tensor",
        meta=partial(_prim_elementwise_meta, type_promotion=type_promotion),
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
        meta=partial(_prim_elementwise_meta, type_promotion=type_promotion),
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
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.COMPLEX_TO_FLOAT,
)

acos = _make_elementwise_unary_prim(
    "acos",
    impl_aten=torch.acos,
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
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

atanh = _make_elementwise_unary_prim(
    "atanh",
    impl_aten=torch.atanh,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

cos = _make_elementwise_unary_prim(
    "cos",
    impl_aten=torch.cos,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

cosh = _make_elementwise_unary_prim(
    "cosh",
    impl_aten=torch.cosh,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

bessel_j0 = _make_elementwise_unary_prim(
    "bessel_j0",
    impl_aten=torch.special.bessel_j0,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

bessel_j1 = _make_elementwise_unary_prim(
    "bessel_j1",
    impl_aten=torch.special.bessel_j1,
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
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)


def _cbrt_aten(a: torch.Tensor) -> Tensor:
    torch._check(
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
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)


def _conj_physical_meta(input: TensorLikeType) -> TensorLikeType:
    if not input.dtype.is_complex:
        raise RuntimeError("prims.conj_physical is only defined for complex dtypes")

    strides = utils.compute_elementwise_output_strides(input)
    return TensorMeta(input, strides=strides)


conj_physical = _make_prim(
    schema="conj_physical(Tensor self) -> Tensor",
    meta=_conj_physical_meta,
    impl_aten=torch._conj_physical,
    doc="Returns the physical conjugation of a complex tensor",
    return_type=RETURN_TYPE.NEW,
)


def _clone_meta(
    input: TensorLikeType, *, memory_format: torch.memory_format = torch.preserve_format
) -> TensorLikeType:
    if memory_format != torch.preserve_format:
        return torch.empty(
            input.shape,
            dtype=input.dtype,
            layout=input.layout,
            device=input.device,
            memory_format=memory_format,
        )
    else:
        # Match eager behavior by preserving strides for non_overlapping_and_dense tensors
        # If not, eager clone creates contiguous strides
        computed_stride = utils.compute_elementwise_output_strides(input)
        return torch.empty_strided(
            input.shape,
            computed_stride,
            dtype=input.dtype,
            layout=input.layout,
            device=input.device,
        )


clone = _make_prim(
    schema="clone(Tensor self, *, MemoryFormat? memory_format=None) -> Tensor",
    meta=_clone_meta,
    impl_aten=torch.clone,
    doc="Returns the copy of a tensor",
    return_type=RETURN_TYPE.NEW,
    register_conj_neg_fallthrough=True,
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
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

erfcx = _make_elementwise_unary_prim(
    "erfcx",
    impl_aten=torch.special.erfcx,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

exp = _make_elementwise_unary_prim(
    "exp",
    impl_aten=torch.exp,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

expm1 = _make_elementwise_unary_prim(
    "expm1",
    impl_aten=torch.special.expm1,
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
    return _prim_elementwise_meta(
        a, type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT
    )


# NOTE: fill uses _make_prim directly because it has a value parameter
fill = _make_prim(
    schema="fill(Tensor self, Scalar value) -> Tensor",
    return_type=RETURN_TYPE.NEW,
    meta=_fill_meta,
    impl_aten=torch.fill,
    doc="",
)

floor = _make_elementwise_unary_prim(
    "floor",
    impl_aten=torch.floor,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

imag = _make_prim(
    schema="imag(Tensor(a) self) -> Tensor(a)",
    meta=partial(
        _complex_only_elementwise_meta,
        type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.COMPLEX_TO_FLOAT,
    ),
    return_type=RETURN_TYPE.VIEW,
    impl_aten=torch.imag,
    doc="",
)

isfinite = _make_elementwise_unary_prim(
    "isfinite",
    impl_aten=torch.isfinite,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
)

lgamma = _make_elementwise_unary_prim(
    "lgamma",
    impl_aten=torch.lgamma,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

log = _make_elementwise_unary_prim(
    "log",
    impl_aten=torch.log,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

log1p = _make_elementwise_unary_prim(
    "log1p",
    impl_aten=torch.log1p,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

log2 = _make_elementwise_unary_prim(
    "log2",
    impl_aten=torch.log2,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

log10 = _make_elementwise_unary_prim(
    "log10",
    impl_aten=torch.log10,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

real = _make_prim(
    schema="real(Tensor(a) self) -> Tensor(a)",
    meta=partial(
        _complex_only_elementwise_meta,
        type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.COMPLEX_TO_FLOAT,
    ),
    return_type=RETURN_TYPE.VIEW,
    impl_aten=torch.real,
    doc="",
)

reciprocal = _make_elementwise_unary_prim(
    "reciprocal",
    impl_aten=torch.reciprocal,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

ndtri = _make_elementwise_unary_prim(
    "ndtri",
    impl_aten=torch.special.ndtri,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

neg = _make_elementwise_unary_prim(
    "neg",
    impl_aten=torch.neg,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

round = _make_elementwise_unary_prim(
    "round",
    impl_aten=torch.round,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

rsqrt = _make_elementwise_unary_prim(
    "rsqrt",
    impl_aten=torch.rsqrt,
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
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

sinh = _make_elementwise_unary_prim(
    "sinh",
    impl_aten=torch.sinh,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

spherical_bessel_j0 = _make_elementwise_unary_prim(
    "spherical_bessel_j0",
    impl_aten=torch.special.spherical_bessel_j0,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

sqrt = _make_elementwise_unary_prim(
    "sqrt",
    impl_aten=torch.sqrt,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

tan = _make_elementwise_unary_prim(
    "tan",
    impl_aten=torch.tan,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

tanh = _make_elementwise_unary_prim(
    "tanh",
    impl_aten=torch.tanh,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

trunc = _make_elementwise_unary_prim(
    "trunc",
    impl_aten=torch.trunc,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

#
# Elementwise binary operations
#

add = _make_elementwise_binary_prim(
    name="add",
    impl_aten=torch.add,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

atan2 = _make_elementwise_binary_prim(
    name="atan2",
    impl_aten=torch.atan2,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

bitwise_and = _make_elementwise_binary_prim(
    "bitwise_and",
    impl_aten=torch.bitwise_and,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

bitwise_or = _make_elementwise_binary_prim(
    "bitwise_or",
    impl_aten=torch.bitwise_or,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

bitwise_xor = _make_elementwise_binary_prim(
    "bitwise_xor",
    impl_aten=torch.bitwise_xor,
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
    is_integral = isinstance(a, (bool, int, torch.SymInt)) or (
        isinstance(a, torch.Tensor) and utils.is_integer_dtype(a.dtype)
    )

    if is_integral:
        return torch.div(a, b, rounding_mode="trunc")
    else:
        return torch.true_divide(a, b)


div = _make_elementwise_binary_prim(
    "div",
    impl_aten=_div_aten,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

eq = _make_elementwise_binary_prim(
    "eq",
    impl_aten=torch.eq,
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
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
)

gt = _make_elementwise_binary_prim(
    "gt",
    impl_aten=torch.gt,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
)

hypot = _make_elementwise_binary_prim(
    "hypot",
    impl_aten=torch.hypot,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
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
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
)

lt = _make_elementwise_binary_prim(
    "lt",
    impl_aten=torch.lt,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
)


# Note: the following impls are because torch.maximum and torch.minimum do not support scalar inputs
def _maximum_aten(
    a: TensorLikeType | NumberType, b: TensorLikeType | NumberType
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
    a: TensorLikeType | NumberType, b: TensorLikeType | NumberType
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
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

ne = _make_elementwise_binary_prim(
    "ne",
    impl_aten=torch.ne,
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
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

remainder = _make_elementwise_binary_prim(
    "remainder",
    impl_aten=torch.remainder,
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
def _as_strided_meta(
    a: TensorLikeType, size: ShapeType, stride: StrideType, storage_offset: int
) -> TensorLikeType:
    if len(size) != len(stride):
        raise AssertionError(f"len(size)={len(size)} != len(stride)={len(stride)}")
    if storage_offset < 0:
        raise AssertionError(f"storage_offset must be >= 0, got {storage_offset}")
    utils.validate_strides(stride)
    utils.validate_shape(size)

    if reduce(operator.mul, size) == 0:
        # NOTE: This special case is to avoid having to acquire the storage below
        # as_strided to shapes with no elements are trivially valid, so it's OK
        pass
    elif isinstance(a, torch.Tensor):
        utils.check_in_bounds_for_storage(
            a._typed_storage(), size, stride, storage_offset
        )

    return torch.as_strided(a, size, stride, storage_offset)


def _as_strided_aten(
    a: Tensor, size: ShapeType, stride: StrideType, storage_offset: int
) -> Tensor:
    return torch.as_strided(a, size, stride, storage_offset)


_as_strided_doc = """
    Creates a view of the tensor with the given shape (size), strides (stride) and
    storage offset (storage_offset).
"""

as_strided = _make_prim(
    schema="as_strided(Tensor(a!) a, SymInt[] size, SymInt[] stride, SymInt storage_offset) -> Tensor(a!)",
    meta=_as_strided_meta,
    impl_aten=_as_strided_aten,
    return_type=RETURN_TYPE.VIEW,
    doc=_as_strided_doc,
)


def _broadcast_in_dim_meta(
    a: TensorLikeType, shape: ShapeType, broadcast_dimensions: Sequence[int]
):
    from torch.fx.experimental.symbolic_shapes import (
        guard_or_false,
        guard_or_true,
        sym_or,
    )

    # Type checks
    if not isinstance(a, TensorLike):
        raise AssertionError(f"a must be TensorLike, got {type(a)}")  # mypy
    if not isinstance(shape, Sequence):
        raise AssertionError(f"shape must be a Sequence, got {type(shape)}")
    if not isinstance(broadcast_dimensions, Sequence):
        raise AssertionError(
            f"broadcast_dimensions must be a Sequence, got {type(broadcast_dimensions)}"
        )

    # every dimension must be accounted for
    if a.ndim != len(broadcast_dimensions):
        raise AssertionError(
            f"a.ndim ({a.ndim}) != len(broadcast_dimensions) ({len(broadcast_dimensions)})"
        )

    # broadcast shape must have weakly more dimensions
    if len(shape) < a.ndim:
        raise AssertionError(f"len(shape) ({len(shape)}) must be >= a.ndim ({a.ndim})")

    # broadcast_dimensions must be an ascending sequence
    # (no relative reordering of dims) of integers and
    # each dimension must be within the new shape
    def _greater_than_reduce(acc, x):
        if not isinstance(x, Dim):
            raise AssertionError(
                f"broadcast_dimensions element must be Dim, got {type(x)}"
            )
        if x <= acc:
            raise AssertionError(
                f"broadcast_dimensions must be strictly ascending: {x} <= {acc}"
            )
        if x >= len(shape):
            raise AssertionError(
                f"broadcast_dimension {x} out of bounds for shape of length {len(shape)}"
            )

        return x

    reduce(_greater_than_reduce, broadcast_dimensions, -1)

    # shape must be broadcastable to
    for idx, new_idx in enumerate(broadcast_dimensions):
        torch._check(
            sym_or(a.shape[idx] == 1, shape[new_idx] == a.shape[idx]),
            lambda: f"{a.shape[idx]} must be broadcastable to {shape[new_idx]}",
        )

    new_strides = []
    original_idx = 0
    for idx in range(len(shape)):
        if idx in broadcast_dimensions:
            # Assigns a stride of zero to dimensions
            # which were actually broadcast
            if guard_or_false(a.shape[original_idx] == 1):
                if guard_or_false(a.shape[original_idx] == shape[idx]):
                    new_strides.append(a.stride()[original_idx])
                else:
                    new_strides.append(0)
            else:
                torch._check(
                    a.shape[original_idx] == shape[idx],
                    lambda: f"non-broadcasting semantics require {a.shape[original_idx]} == {shape[idx]}",
                )
                new_strides.append(a.stride()[original_idx])
            original_idx = original_idx + 1
        else:
            if guard_or_true(shape[idx] != 1):
                # consistent with previous use of guard_size_oblivious
                new_strides.append(0)
            elif original_idx == a.ndim:
                new_strides.append(1)
            else:
                new_strides.append(a.stride()[original_idx] * a.size()[original_idx])

    return a.as_strided(shape, new_strides, a.storage_offset())


def _broadcast_in_dim_aten(a, shape, broadcast_dimensions):
    s = list(shape)
    for broadcast_dimension in broadcast_dimensions:
        s[broadcast_dimension] = -1

    v = a
    for idx, x in enumerate(s):
        if x != -1:
            v = v.unsqueeze(idx)

    return v.expand(shape)


_broadcast_in_dim_doc = """
  Creates a view of a with the specified shape.

  Allows adding dimensions of any length and broadcasting
  dimensions of length one in a to any length.

  The location of the broadcast dimensions must be specified
  using the broadcast_dimensions argument. Changing the
  relative order of dimensions is not supported.
  """

broadcast_in_dim = _make_prim(
    schema="broadcast_in_dim(Tensor(a) a, SymInt[] shape, int[] broadcast_dimensions) -> Tensor(a)",
    meta=_broadcast_in_dim_meta,
    impl_aten=_broadcast_in_dim_aten,
    return_type=RETURN_TYPE.VIEW,
    doc=_broadcast_in_dim_doc,
)


def _validate_collapse_args(a: Tensor, start: int, end: int) -> None:
    # Special-case for zero dimensional tensors
    ndim = max(1, a.dim())
    utils.validate_idx(ndim, start)
    utils.validate_idx(ndim, end)

    # Verifies end is strictly greater than start
    # (Collapse requires a non-empty interval)
    torch._check_value(
        end >= start,
        lambda: f"Attempting to collapse but end, {end}, is less than start, {start}!",
    )


def _collapsed_shape(shape: ShapeType, start: int, end: int) -> tuple[int, ...]:
    """
    Returns the shape of a with dims in [start, end) merged into a single dimension.
    """
    # Special-case for zero dimensional tensors
    shape = (1,) if len(shape) == 0 else tuple(shape)

    dim_length = 1
    for s in shape[start : end + 1]:
        dim_length = dim_length * s

    return shape[0:start] + (dim_length,) + shape[end + 1 :]


# If the collapse is invalid or cannot be determined (because of unbacked data)
# then `must_be_valid` determines the behavior:
#   None: return None, None.
#   str: Do a torch._check() to ensure the collapse is valid and if it isn't
#   then fail with the provided string.
def _collapse_view_helper(
    a: TensorLikeType, start: int, end: int, must_be_valid: str | None
) -> tuple[ShapeType | None, StrideType | None]:
    if not isinstance(a, TensorLike):
        raise AssertionError(f"a must be TensorLike, got {type(a)}")  # mypy

    from torch.fx.experimental.symbolic_shapes import (
        guard_or_false,
        guard_or_true,
        sym_and,
        sym_or,
    )

    _validate_collapse_args(a, start, end)

    # Special-case for zero dimensional tensors
    if a.ndim == 0:
        shape = (1,)
        strides = (1,)
    else:
        shape = a.shape  # type: ignore[assignment]
        strides = a.stride()  # type: ignore[assignment]

    if a.ndim == 0 or (end == start):
        return shape, strides

    valid_op = True
    if guard_or_false(a.numel() != 0):
        for idx in range(end - 1, start - 1, -1):
            valid_op = sym_and(
                valid_op,
                sym_or(
                    shape[idx] == 1,
                    shape[idx + 1] == 1,
                    strides[idx] == strides[idx + 1] * shape[idx + 1],
                ),
            )  # type: ignore[assignment]

            # early exit if we already know its invalid.
            if guard_or_false(valid_op is False):
                break

    # for unbacked this become a runtime assertion.
    valid_op = sym_or(valid_op, a.numel() == 0)

    if must_be_valid:
        torch._check(valid_op, lambda: must_be_valid)
    else:
        if not guard_or_false(valid_op):
            return None, None

    # compute stride
    stride = strides[end]
    for idx in range(end - 1, start - 1, -1):
        if shape[idx] != 1:
            # TODO with unbacked we should really exclude when shape[idx] == 1
            # something like
            # min(stride[end], torch.ite(shape[x]!=1,stride[idx], inf), ...)
            stride = min(stride, strides[idx])

    # compute length
    length = shape[end]
    if guard_or_true(length != 0):
        for idx in range(end - 1, start - 1, -1):
            if guard_or_false(shape[idx] == 0):
                length = 0
                stride = 0
                break
            length = length * shape[idx]
    else:
        stride = 0

    new_shape = shape[:start] + (length,) + shape[end + 1 :]
    new_strides = strides[:start] + (stride,) + strides[end + 1 :]

    # NOTE: when the input has no elements it's restrided as if it were contiguous
    # except for unbacked.
    if guard_or_false(a.numel() == 0):
        new_strides = utils.make_contiguous_strides_for(new_shape)

    return new_shape, new_strides


def _collapse_view_meta(a: TensorLikeType, start: int, end: int) -> TensorLikeType:
    new_shape, new_strides = _collapse_view_helper(
        a, start, end, "Attempting to view a collapsed tensor, but no such view exists!"
    )
    if new_strides is None:
        raise AssertionError(
            "new_strides should not be None after _collapse_view_helper"
        )
    if new_shape is None:
        raise AssertionError("new_shape should not be None after _collapse_view_helper")
    return a.as_strided(new_shape, new_strides, a.storage_offset())


def _collapse_view_aten(a: Tensor, start: int, end: int) -> Tensor:
    new_shape = _collapsed_shape(a.shape, start, end)
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


def _conj_meta(a: TensorLikeType) -> TensorLikeType:
    if not a.dtype.is_complex:
        raise RuntimeError("Expected complex dtype in prims.conj")
    out = a.as_strided(a.shape, a.stride(), a.storage_offset())
    torch._C._set_conj(out, not a.is_conj())
    return out


_conj_doc = """
Returns a conjugated view of the original tensor
"""

conj = _make_prim(
    schema="conj(Tensor(a) a) -> Tensor(a)",
    meta=_conj_meta,
    impl_aten=torch.conj,
    return_type=RETURN_TYPE.VIEW,
    doc=_conj_doc,
)


def expand_dims(
    a: TensorLikeType, dimensions: DimsSequenceType, ndim=None
) -> TensorLikeType:
    """
    Creates a view of a with a.ndim + len(dimensions) dimensions, with new
    dimensions of length one at the dimensions specified by dimensions.
    """
    if ndim is not None:
        # TODO: this is only here to support the unsqueeze ref
        dims = sorted(utils.canonicalize_dims(ndim, dimensions))  # type: ignore[arg-type]
    else:
        dims = sorted(utils.canonicalize_dims(a.ndim, dimensions))  # type: ignore[arg-type]
    if len(set(dims)) != len(dims):
        msg = f"Received duplicate dimensions to expand in {str(dimensions)}"
        raise ValueError(msg)

    new_shape = list(a.shape)
    for idx in dims:
        new_shape.insert(idx, 1)

    broadcast_dimensions = [
        idx for idx in range(len(new_shape)) if idx not in dimensions
    ]
    return broadcast_in_dim(a, new_shape, broadcast_dimensions)


def _split_dim_meta(a: TensorLikeType, dim: int, outer_length: int) -> TensorLikeType:
    if not isinstance(a, TensorLike):
        raise AssertionError(f"a must be TensorLike, got {type(a)}")  # mypy
    utils.validate_idx(a.ndim, dim)
    utils.validate_dim_length(outer_length)

    # Verifies the dim can be split with the specified lhs_length
    inner_length = a.shape[dim] // outer_length

    if (a.shape[dim] % outer_length) != 0:
        msg = (
            f"Attempting to split dimension of length {a.shape[dim]}, "
            f"but outer length of {outer_length} divides it with a remainder!"
        )
        raise ValueError(msg)

    new_shape: list[int] = []
    new_strides: list[int] = []
    for idx in range(a.ndim):
        if idx == dim:
            new_shape.extend((outer_length, inner_length))
            new_strides.extend((a.stride()[idx] * inner_length, a.stride()[idx]))
        else:
            new_shape.append(a.shape[idx])
            new_strides.append(a.stride()[idx])

    return a.as_strided(new_shape, new_strides, a.storage_offset())


def _split_dim_aten(a: Tensor, dim: int, outer_length: int) -> Tensor:
    inner_length = a.shape[dim] // outer_length
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
    schema="split_dim(Tensor(a) a, int dim, SymInt outer_length) -> Tensor(a)",
    meta=_split_dim_meta,
    impl_aten=_split_dim_aten,
    return_type=RETURN_TYPE.VIEW,
    doc=_split_dim_doc,
)


# Note: allows dimensions to be specified redundantly
def _squeeze_meta(a: TensorLikeType, dimensions: Sequence) -> TensorLikeType:
    if not isinstance(a, TensorLike):
        raise AssertionError(f"a must be TensorLike, got {type(a)}")  # mypy

    for idx in dimensions:
        utils.validate_idx(a.ndim, idx)
        if a.shape[idx] != 1:
            raise AssertionError(
                f"Cannot squeeze dimension {idx} with size {a.shape[idx]} (must be 1)"
            )

    new_shape = []
    new_strides = []
    for idx in range(len(a.shape)):
        if idx in dimensions:
            continue

        new_shape.append(a.shape[idx])
        new_strides.append(a.stride()[idx])

    return a.as_strided(new_shape, new_strides, a.storage_offset())


_squeeze_doc = """
  Creates a view of the tensor with the specified dimensions removed.

  The removed dimensions must each have length one.
  """

squeeze = _make_prim(
    schema="squeeze(Tensor(a) a, int[] dimensions) -> Tensor(a)",
    meta=_squeeze_meta,
    impl_aten=torch.squeeze,
    return_type=RETURN_TYPE.VIEW,
    doc=_squeeze_doc,
)


def _transpose_meta(a: TensorLikeType, permutation: DimsSequenceType) -> TensorLikeType:
    if a.ndim != len(permutation):
        msg = f"Attempting to permute a tensor of rank {a.ndim}, but received a permutation of length {len(permutation)}!"
        raise ValueError(msg)

    if not utils.is_valid_permutation(a.ndim, permutation):
        msg = f"Received an invalid permutation, {permutation}!"
        raise ValueError(msg)

    new_shape = [0] * a.ndim
    new_strides = [0] * a.ndim
    for idx, dim in enumerate(permutation):
        new_shape[idx] = a.shape[dim]
        new_strides[idx] = a.stride()[dim]

    return a.as_strided(tuple(new_shape), tuple(new_strides), a.storage_offset())


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
    return a.as_strided(a.shape, a.stride(), a.storage_offset())


def _view_of_aten(a: Tensor) -> Tensor:
    return a.view(a.shape)


_view_of_doc = """
    Creates a view of the tensor.
    """

view_of = _make_prim(
    schema="view_of(Tensor(a) a) -> Tensor(a)",
    meta=_view_of_meta,
    impl_aten=_view_of_aten,
    return_type=RETURN_TYPE.VIEW,
    doc=_view_of_doc,
)


def _view_element_type_meta(a: TensorLikeType, dtype: torch.dtype) -> TensorLikeType:
    return a.view(dtype)


def _view_element_type_aten(a: Tensor, dtype: torch.dtype) -> Tensor:
    return a.view(dtype)


_view_element_type_doc = """
    Creates a view of the tensor with a different dtype.
    """

view_element_type = _make_prim(
    schema="view_of_dtype(Tensor(a) a, ScalarType dtype) -> Tensor(a)",
    meta=_view_element_type_meta,
    impl_aten=_view_element_type_aten,
    return_type=RETURN_TYPE.VIEW,
    doc=_view_element_type_doc,
)

#
# Functionalized view mutations
#


def _as_strided_scatter_meta(
    input: TensorLikeType,
    src: TensorLikeType,
    size: ShapeType,
    stride: StrideType,
    storage_offset: int,
) -> TensorLikeType:
    utils.validate_shape(size)
    utils.validate_strides(stride)

    required_size = utils.compute_required_storage_length(size, stride, storage_offset)
    torch._check(
        input.numel() >= required_size,
        lambda: (
            f"as_strided_scatter: sizes {size}, strides {stride}, storage offset {storage_offset} "
            f" and itemsize {input.element_size()} requiring a storage size of "
            f"{required_size * input.element_size()} are out of bounds "
            f"for storage of size {input.numel() * input.element_size()}"
        ),
    )
    torch._check(
        utils.is_same_shape(src.shape, size),
        lambda: f"expected src to have a size equal to the slice of self. src size = {src.shape}, slice size = {size}",
    )

    return utils.clone_preserve_strides(input)


_as_strided_scatter_doc = """
    Creates a new tensor equivalent to ``out = input.clone()`` after mutation by
    ``out.as_strided(size, stride, storage_offset).copy_(src)``.
"""

as_strided_scatter = _make_prim(
    schema="as_strided_scatter(Tensor self, Tensor src, SymInt[] size, SymInt[] stride, SymInt storage_offset) -> Tensor",
    meta=_as_strided_scatter_meta,
    impl_aten=torch.as_strided_scatter,
    return_type=RETURN_TYPE.NEW,
    doc=_as_strided_scatter_doc,
)


#
# Shape operations
#


def _collapse_meta(a: Tensor, start: int, end: int) -> Tensor:
    # Special-case for zero dimensional tensors
    _validate_collapse_args(a, start, end)
    new_shape = _collapsed_shape(a.shape, start, end)
    return a.new_empty(new_shape)


def _collapse_aten(a: Tensor, start: int, end: int) -> Tensor:
    new_shape = _collapsed_shape(a.shape, start, end)
    out = a.new_empty(new_shape)
    with torch.no_grad():
        out.view_as(a).copy_(a)
    return out


_collapse_doc = """
Collapse a span of neighboring dimensions into one.

See collapse_view for the corresponding view operation.
"""
collapse = _make_prim(
    schema="collapse(Tensor a, int start, int end) -> Tensor",
    meta=_collapse_meta,
    impl_aten=_collapse_aten,
    return_type=RETURN_TYPE.NEW,
    doc=_collapse_doc,
)


# TODO: review stride logic
# NB: unlike torch.cat, this is more strict about empty tensors and dim is
# never negative
def _cat_meta(tensors: Sequence[TensorLikeType], dim: int) -> TensorLikeType:
    # Verifies same shape (except in the concat dimension)
    if dim < 0:
        raise AssertionError(f"dim must be non-negative, got {dim}")
    shape = tensors[0].shape
    sym_sum_args = []
    for tensor_idx, tensor in enumerate(tensors):
        if len(shape) != len(tensor.shape):
            raise AssertionError(
                f"All tensors must have the same number of dimensions. "
                f"Expected {len(shape)} but tensor {tensor_idx} has {len(tensor.shape)}"
            )
        for idx, (common_length, length) in enumerate(zip(shape, tensor.shape)):
            if idx == dim:
                sym_sum_args.append(length)
            else:
                torch._check(
                    length == common_length,
                    lambda: f"Sizes of tensors must match except in dimension {dim}. "
                    f"Expected {common_length} in dimension {idx} but got {length} for tensor number "
                    f"{tensor_idx} in the list",
                )

    new_shape = list(tensors[0].shape).copy()
    new_shape[dim] = torch.sym_sum(sym_sum_args)
    return TensorMeta(
        tensors[0],
        shape=new_shape,
        strides=utils.make_contiguous_strides_for(new_shape),
    )


def _cat_aten(tensors: tuple[Tensor, ...] | list[Tensor], dim: int) -> Tensor:
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
    if not isinstance(a, TensorLike):
        raise AssertionError(f"a must be TensorLike, got {type(a)}")  # mypy
    utils.validate_shape(shape)

    # Validates the tensor and the requested shape have the
    # same number of elements
    numel = reduce(operator.mul, shape)
    if numel != a.numel():
        msg = f"Attempting to reshape a tensor with {a.numel()} elements to a shape with {numel} elements!"
        raise ValueError(msg)

    return TensorMeta(a, shape=shape, strides=utils.make_contiguous_strides_for(shape))


def _reshape_aten(a: Tensor, shape: ShapeType) -> Tensor:
    return a.reshape(shape).clone(memory_format=torch.contiguous_format)


_reshape_doc = """
  Creates a contiguous tensor with the specified shape
  containing a copy of the data in a.
  """
reshape = _make_prim(
    schema="reshape(Tensor a, SymInt[] shape) -> Tensor",
    meta=_reshape_meta,
    impl_aten=_reshape_aten,
    return_type=RETURN_TYPE.NEW,
    doc=_reshape_doc,
)


def _rev_meta(a: TensorLikeType, dims: DimsSequenceType) -> TensorLikeType:
    utils.validate_dimension_indices(a.ndim, dims)
    return torch.empty_like(a, memory_format=torch.preserve_format)


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
    return _prim_elementwise_meta(
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
    return_type=RETURN_TYPE.NEW,
    doc=_where_doc,
)


#
# Type conversions
#
def _convert_element_type_meta(a: TensorLikeType, dtype: torch.dtype) -> TensorLikeType:
    # Type checks
    if not isinstance(a, TensorLike):
        raise AssertionError(f"a must be TensorLike, got {type(a)}")  # mypy
    if not isinstance(dtype, torch.dtype):
        raise AssertionError(f"dtype must be torch.dtype, got {type(dtype)}")

    # dtype conversion preserves dense strides
    if torch._prims_common.is_non_overlapping_and_dense_or_false(a):
        strides = a.stride()
    else:
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
        except Exception:
            requires_grad = False

    result = torch.empty_like(
        a, device=a.device, dtype=dtype, requires_grad=requires_grad
    )
    with torch.no_grad():
        return copy_to(result, a)


_convert_element_type_doc = """
  Creates a copy of a tensor with the given dtype.
  """

convert_element_type = _make_prim(
    schema="convert_element_type(Tensor a, ScalarType dtype) -> Tensor",
    meta=_convert_element_type_meta,
    impl_aten=_convert_element_type_aten,
    return_type=RETURN_TYPE.NEW,
    doc=_convert_element_type_doc,
    tags=(torch.Tag.pointwise,),
)


def _device_put_meta(
    a: TensorLikeType, device: str | torch.device, non_blocking=False
) -> TensorLikeType:
    if not isinstance(a, TensorLike):
        raise AssertionError(f"a must be TensorLike, got {type(a)}")  # mypy
    if not isinstance(device, (str, torch.device)):
        raise AssertionError(f"device must be str or torch.device, got {type(device)}")
    if not isinstance(non_blocking, bool):
        raise AssertionError(f"non_blocking must be bool, got {type(non_blocking)}")

    return TensorMeta(a, device=utils.canonicalize_device(device))


def _device_put_aten(
    a: Tensor, device: str | torch.device, non_blocking=False
) -> Tensor:
    return a.to(device, non_blocking=non_blocking)


_device_put_doc = """
  Creates a copy of a tensor on the given device.
  """

device_put = _make_prim(
    schema="device_put(Tensor a, Device device, bool non_blocking=False) -> Tensor",
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


# We can't call into python dispatcher for item again
# because the current prim decomp calls into python dispatcher
# again. https://github.com/pytorch/pytorch/issues/136050
def _item_aten_no_python_dispatcher(*args, **kwargs):
    with torch._dispatch.python.no_python_dispatcher():
        return torch.Tensor.item(*args, **kwargs)


# TODO: create a new return type for scalars?
# FIXME: currently returns integers for boolean tensors
# https://github.com/pytorch/pytorch/issues/78071
item = _make_prim(
    schema="item(Tensor a) -> Scalar",
    meta=_item_meta,
    impl_aten=_item_aten_no_python_dispatcher,
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
    Return the minimum finite value for a dtype.
"""

# TODO: create a new return type for scalars?
# FIXME: currently returns integers for boolean tensors
# https://github.com/pytorch/pytorch/issues/78071
minimum_value = _make_prim(
    schema="minimum_value(ScalarType dtype) -> Scalar",
    meta=_minimum_value_meta,
    impl_aten=_minimum_value_aten,
    return_type=RETURN_TYPE.NEW,
    doc=_minimum_value_doc,
)

#
# Inplace operators
#


def _copy_to_meta(a: TensorLikeType, b: TensorLikeType):
    if not isinstance(a, TensorLike):
        raise AssertionError(f"a must be TensorLike, got {type(a)}")  # mypy
    if not isinstance(b, TensorLike):
        raise AssertionError(f"b must be TensorLike, got {type(b)}")  # mypy

    # Validates the cast is safe
    # TODO: move this as an option on the reference
    # a_typ = utils.dtype_to_type(a.dtype)
    # b_typ = utils.dtype_to_type(b.dtype)
    # if a_typ is not utils.get_higher_type(a_typ, b_typ):
    #     raise RuntimeError(str(b.dtype), " can't be cast safely to ", str(a.dtype), "!")

    # Validates the tensors have the same number of elements
    if a.numel() != b.numel():
        msg = f"Attempting to copy {b.numel()} elements to a tensor with {a.numel()} elements!"
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
    register_conj_neg_fallthrough=True,
)


def _copy_strided_meta(a: TensorLikeType, stride: ShapeType):
    if not isinstance(a, TensorLike):
        raise AssertionError(f"a must be TensorLike, got {type(a)}")  # mypy
    return torch.empty_strided(
        a.shape,
        stride,
        dtype=a.dtype,
        layout=a.layout,
        device=a.device,
        requires_grad=a.requires_grad,
    )


def _copy_strided_aten(a: Tensor, stride: ShapeType) -> Tensor:
    out = torch.empty_strided(
        a.size(),
        stride=stride,
        dtype=a.dtype,
        layout=a.layout,
        device=a.device,
        requires_grad=a.requires_grad,
    )
    out.copy_(a)
    return out


_copy_strided_doc = """
  Copies the data in a to a new tensor, the new tensor has same shape with a size, but has different stride.
  """


copy_strided = _make_prim(
    schema="copy_strided(Tensor a, SymInt[] stride) -> Tensor",
    meta=_copy_strided_meta,
    impl_aten=_copy_strided_aten,
    return_type=RETURN_TYPE.NEW,
    doc=_copy_strided_doc,
)


def _resize_meta(a: TensorLikeType, shape: ShapeType):
    return a.resize_(shape)


def _resize_aten(a: Tensor, shape: ShapeType) -> Tensor:
    return a.resize_(shape)


_resize_doc = """
  Gives a tensor with no elements a new shape, returning the modified tensor.

  The tensor's strides are contiguous and its values are uninitialized.
  """

# TODO: review support arbitrary resizes
resize = _make_prim(
    schema="resize(Tensor(a!) a, SymInt[] shape) -> Tensor(a!)",
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
    if not isinstance(inp, TensorLike):
        raise AssertionError(f"inp must be TensorLike, got {type(inp)}")  # mypy
    if output_dtype is None:
        output_dtype = inp.dtype
    output_shape = utils.compute_reduction_output_shape(inp.shape, dims)
    return TensorMeta(
        shape=output_shape,
        strides=utils.make_contiguous_strides_for(output_shape),
        dtype=output_dtype,
        device=inp.device,
    )


def _var_reduction_meta(inp, dims, correction):
    if utils.is_complex_dtype(inp.dtype):
        output_dtype = utils.corresponding_real_dtype(inp.dtype)
    else:
        output_dtype = inp.dtype
    return _reduction_meta(inp, dims, output_dtype=output_dtype)


_sum_doc = """
    Computes the sum of elements in the input tensor over the list of dimensions
    specified in the dim argument
    """
_xor_sum_doc = """
    Computes the xor sum of elements in the input tensor over the list of dimensions
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


def _make_reduction_prim(name: str, impl_aten, doc):
    """Creates a reduction prim."""
    return _make_prim(
        schema=f"{name}(Tensor inp, int[]? dims, *, ScalarType? output_dtype=None) -> Tensor",
        meta=_reduction_meta,
        impl_aten=impl_aten,
        return_type=RETURN_TYPE.NEW,
        doc=doc,
    )


def _make_var_reduction_prim(name: str, impl_aten, doc):
    """Creates a reduction prim."""
    return _make_prim(
        schema=f"{name}(Tensor inp, int[]? dims, float? correction=1, *, ScalarType? output_dtype=None) -> Tensor",
        meta=_var_reduction_meta,
        impl_aten=impl_aten,
        return_type=RETURN_TYPE.NEW,
        doc=doc,
    )


sum = _make_reduction_prim(
    name="sum",
    impl_aten=torch.sum,
    doc=_sum_doc,
)


def _xor_sum_aten(
    inp: TensorLikeType,
    dims: DimsSequenceType | None,
    *,
    dtype: torch.dtype | None = None,
) -> Tensor:
    raise NotImplementedError("xor_sum only implemented with inductor")


xor_sum = _make_reduction_prim(
    name="xor_sum",
    impl_aten=_xor_sum_aten,
    doc=_xor_sum_doc,
)


def _prod_aten(
    inp: TensorLikeType,
    dims: DimsSequenceType | None,
    *,
    dtype: torch.dtype | None = None,
) -> Tensor:
    if dims is not None:
        if len(dims) == 0:
            return inp.clone()
        for d in sorted(dims, reverse=True):
            if d < 0:
                raise AssertionError(f"dimension must be non-negative, got {d}")
            inp = torch.prod(inp, d, dtype=dtype)
        return inp
    else:
        return torch.prod(inp, dims, dtype=dtype)


prod = _make_reduction_prim(
    name="prod",
    impl_aten=_prod_aten,
    doc=_prod_doc,
)


# torch.var, but correction is not kwarg-only
def torch_var(input, dim=None, correction=1, **kwargs):
    return torch.var(input, dim=dim, correction=correction, **kwargs)


var = _make_var_reduction_prim(
    name="var",
    impl_aten=torch_var,
    doc=_var_doc,
)

amax = _make_reduction_prim(
    name="amax",
    impl_aten=torch.amax,
    doc=_amax_doc,
)

amin = _make_reduction_prim(
    name="amin",
    impl_aten=torch.amin,
    doc=_amin_doc,
)


_iota_doc = """
    Constructs a 1-D tensor t where ``t[i] == start + i * step``.
"""


# TODO: layout, pin_memory, memory_format
# TODO: model requires_grad on TensorMeta
def _iota_meta(
    length: int,
    *,
    start: int,
    step: int,
    dtype: torch.dtype,
    device: torch.device,
    requires_grad: bool,
) -> TensorLikeType:
    torch._check(
        utils.is_integer_dtype(dtype),
        lambda: "prims.iota only supports integer dtypes",
    )
    torch._check(step != 0, lambda: "step must be nonzero")
    return torch.empty(
        length,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )


def _iota_aten(
    length: int,
    *,
    start: int,
    step: int,
    dtype: torch.dtype,
    device: torch.device,
    requires_grad: bool,
) -> TensorLikeType:
    end = start + length * step
    return torch.arange(
        start, end, step, dtype=dtype, device=device, requires_grad=requires_grad
    )


iota = _make_prim(
    schema="iota(SymInt length, *, SymInt start, SymInt step, ScalarType dtype, Device device, bool requires_grad) -> Tensor",  # noqa: B950
    return_type=RETURN_TYPE.NEW,
    meta=_iota_meta,
    impl_aten=_iota_aten,
    doc=_iota_doc,
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
    schema="empty(SymInt[] shape, *, ScalarType dtype, Device device, bool requires_grad) -> Tensor",
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
    schema="empty_strided(SymInt[] shape, SymInt[] strides, *, ScalarType dtype, Device device, bool requires_grad) -> Tensor",
    return_type=RETURN_TYPE.NEW,
    meta=_empty_strided_meta,
    impl_aten=torch.empty_strided,
    doc=_empty_strided_doc,
)


def _empty_permuted_meta(
    shape: ShapeType,
    physical_layout: DimsSequenceType,
    *,
    dtype: torch.dtype,
    device: torch.device,
    requires_grad: bool,
) -> TensorLikeType:
    p_strides = utils.make_contiguous_strides_for([shape[l] for l in physical_layout])
    dim = len(shape)
    torch._check(
        len(physical_layout) == dim,
        lambda: (
            "Number of dimensions in the tensor input does not match the "
            f"length of the physical layout; i.e. len(size) = {dim} "
            f"is not equal to len(physical_layout) = {len(physical_layout)}"
        ),
    )
    strides = [0] * len(shape)
    seen_dims = set()
    for p, l in enumerate(physical_layout):
        torch._check(
            0 <= l < dim,
            lambda: (
                f"Dimension out of range (expected to be between 0 and {dim - 1}, but got "
                f"{l} at index {p}).  NB: negative dims "
                "not currently supported; file an issue if you want it."
            ),
        )
        torch._check(l not in seen_dims, lambda: "Duplicate dim not allowed")
        strides[l] = p_strides[p]
        seen_dims.add(l)
    return TensorMeta(
        shape=shape,
        strides=strides,
        dtype=dtype,
        device=device,
    )


_empty_permuted_doc = """
    Creates a tensor with uninitialized values according to some physical layout,
    that is guaranteed to be non-overlapping and dense.
"""

# TODO: add layout, pin_memory
empty_permuted = _make_prim(
    schema="empty_permuted(SymInt[] shape, int[] physical_layout, *, ScalarType dtype, Device device, bool requires_grad) -> Tensor",  # noqa: B950
    return_type=RETURN_TYPE.NEW,
    meta=_empty_permuted_meta,
    impl_aten=torch.empty_permuted,
    doc=_empty_permuted_doc,
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
        shape,
        fill_value,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,  # type: ignore[arg-type]
    )


_full_doc = """
    Creates a tensor filled with the given fill value, and with the specified shape, dtype, and device.
"""

# TODO: add layout
full = _make_prim(
    schema="full(SymInt[] shape, Scalar fill_value, *, ScalarType dtype, Device device, bool requires_grad) -> Tensor",
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
    strides = utils.compute_elementwise_output_strides(a)
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
        a,
        fill_value,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,  # type: ignore[arg-type]
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
# Linear algebra (linalg) prims
#


def _svd_meta(
    A: TensorLikeType, *, full_matrices: bool
) -> tuple[TensorLikeType, TensorLikeType, TensorLikeType]:
    utils.check_is_matrix(A, "linalg.svd")
    utils.check_fp_or_complex(A.dtype, "linalg.svd", allow_low_precision_dtypes=False)

    A_shape = A.shape
    batch = A_shape[:-2]
    m, n = A_shape[-2:]
    k = min(m, n)

    shape_U = batch + (m, m if full_matrices else k)
    strides_U = utils.make_contiguous_strides_for(shape_U, row_major=False)
    U = TensorMeta(shape=shape_U, strides=strides_U, dtype=A.dtype, device=A.device)

    shape_S = batch + (k,)
    strides_S = utils.make_contiguous_strides_for(shape_S)
    S = TensorMeta(
        shape=shape_S,
        strides=strides_S,
        dtype=utils.corresponding_real_dtype(A.dtype) if A.is_complex() else A.dtype,
        device=A.device,
    )

    shape_Vh = batch + (n if full_matrices else k, n)
    # The CPU backend returns V, but the cuSolver backend returns V^H
    # TODO The MAGMA backend returns V, so this is wrong if used with the MAGMA backend
    is_cuda = A.device.type == "cuda"
    strides_Vh = utils.make_contiguous_strides_for(shape_Vh, row_major=is_cuda)
    Vh = TensorMeta(shape=shape_Vh, strides=strides_Vh, dtype=A.dtype, device=A.device)
    # Also makes sure this is CUDA or HIP:
    # https://pytorch.org/docs/stable/notes/hip.html#checking-for-hip
    if A.numel() != 0 and Vh.is_complex() and torch.cuda.is_available():
        Vh = Vh.conj()
    return U, S, Vh


def _svd_aten(
    A: TensorLikeType, *, full_matrices: bool
) -> tuple[Tensor, Tensor, Tensor]:
    return torch.linalg.svd(A, full_matrices=full_matrices)


_svd_doc = """
    Returns the SVD of a matrix or batch of matrices.

    The `full_matrices` flag controls whether the full or reduced SVD decomposition is returned.
"""

svd = _make_prim(
    schema="svd(Tensor A, *, bool full_matrices) -> (Tensor U, Tensor S, Tensor Vh)",
    meta=_svd_meta,
    impl_aten=_svd_aten,
    return_type=(RETURN_TYPE.NEW, RETURN_TYPE.NEW, RETURN_TYPE.NEW),
    doc=_svd_doc,
)


#
# Randomness Prims
#


def _normal_meta(
    shape: ShapeType,
    *,
    mean: float | complex,
    std: float,
    dtype: torch.dtype,
    device: torch.device,
    requires_grad: bool,
    generator: torch.Generator | None = None,
) -> TensorLikeType:
    torch._check(
        std >= 0.0,
        lambda: f"expected non-negative standard deviation, but got std={std}",
    )

    torch._check(
        utils.is_float_dtype(dtype) or utils.is_complex_dtype(dtype),
        lambda: f"expected a floating-point or complex dtype, but got dtype={dtype}",
    )

    strides = utils.make_contiguous_strides_for(shape)
    return TensorMeta(shape=shape, strides=strides, dtype=dtype, device=device)


def _normal_aten(
    shape: ShapeType,
    *,
    mean: float | complex,
    std: float,
    dtype: torch.dtype,
    device: torch.device,
    requires_grad: bool,
    generator: torch.Generator | None = None,
) -> Tensor:
    a = torch.empty(shape, dtype=dtype, device=device, requires_grad=requires_grad)
    with torch.no_grad():
        # NOTE: normal_ is incorrectly annotated to expect mean to be a float
        a.normal_(mean, std, generator=generator)  # type: ignore[arg-type]
    return a


_normal_doc = """
    Constructs a tensor filled with values drawn from a normal distribution with the specified mean
    and standard deviation.

    Only supports floating-point types.
"""

normal = _make_prim(
    schema=(
        "normal(SymInt[] shape, *, Scalar mean, Scalar std, ScalarType dtype, Device device, bool requires_grad, Generator? generator=None) -> Tensor"  # noqa: B950
    ),
    return_type=RETURN_TYPE.NEW,
    meta=_normal_meta,
    impl_aten=_normal_aten,
    doc=_normal_doc,
)


def _uniform_meta(
    shape: ShapeType,
    *,
    low: float,
    high: float,
    dtype: torch.dtype,
    device: torch.device,
    generator: torch.Generator | None = None,
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
    generator: torch.Generator | None = None,
) -> Tensor:
    a = torch.empty(shape, dtype=dtype, device=device)
    a.uniform_(low, high, generator=generator)
    return a


_uniform_doc = """
    Constructs a tensor filled with values drawn uniformly from low to high.
"""

# TODO: we should more seriously review randomness modeling and prims
_uniform_helper = _make_prim(
    schema=(
        "uniform(SymInt[] shape, *, Scalar low, Scalar high, ScalarType dtype, Device device, Generator? generator=None) -> Tensor"
    ),
    return_type=RETURN_TYPE.NEW,
    meta=_uniform_meta,
    impl_aten=_uniform_aten,
    doc=_uniform_doc,
)

#
# FFT prims
#


def _fft_r2c_meta(
    input: TensorLike,
    *,
    dim: DimsSequenceType,
    onesided: bool,
) -> TensorLikeType:
    dim = utils.canonicalize_dims(input.ndim, dim)
    utils.validate_no_repeating_dims(dim)

    shape = list(input.shape)
    if onesided:
        last_dim = dim[-1]
        shape[last_dim] = shape[last_dim] // 2 + 1

    dtype = utils.corresponding_complex_dtype(input.dtype)
    strides = utils.make_contiguous_strides_for(shape)
    return TensorMeta(shape=shape, strides=strides, dtype=dtype, device=input.device)


def _fft_r2c_aten(
    input: TensorLike,
    *,
    dim: DimsSequenceType,
    onesided: bool,
) -> TensorLikeType:
    normalization = 0  # No normalization
    return torch._fft_r2c(input, dim, normalization, onesided)


_fft_r2c_doc = """
    Performs a real to complex Fast Fourier Transform
"""


fft_r2c = _make_prim(
    schema="fft_r2c(Tensor self, *, int[] dim, bool onesided) -> Tensor",
    meta=_fft_r2c_meta,
    impl_aten=_fft_r2c_aten,
    return_type=RETURN_TYPE.NEW,
    doc=_fft_r2c_doc,
)


def _fft_c2c_meta(
    input: TensorLike,
    *,
    dim: DimsSequenceType,
    forward: bool,
) -> TensorLikeType:
    dim = utils.canonicalize_dims(input.ndim, dim)
    utils.validate_no_repeating_dims(dim)

    shape = input.shape
    strides = utils.make_contiguous_strides_for(shape)
    return TensorMeta(
        shape=shape, strides=strides, dtype=input.dtype, device=input.device
    )


def _fft_c2c_aten(
    input: TensorLike,
    *,
    dim: DimsSequenceType,
    forward: bool,
) -> TensorLikeType:
    normalization = 0  # No normalization
    return torch._fft_c2c(input, dim, normalization, forward)


_fft_c2c_doc = """
    Performs either a Fast Fourier Transform, or its inverse
"""


fft_c2c = _make_prim(
    schema="fft_c2c(Tensor self, *, int[] dim, bool forward) -> Tensor",
    meta=_fft_c2c_meta,
    impl_aten=_fft_c2c_aten,
    return_type=RETURN_TYPE.NEW,
    doc=_fft_c2c_doc,
)


def _fft_c2r_meta(
    input: TensorLike,
    *,
    dim: DimsSequenceType,
    last_dim_size: int,
) -> TensorLikeType:
    dim = utils.canonicalize_dims(input.ndim, dim)
    utils.validate_no_repeating_dims(dim)

    shape = list(input.shape)
    shape[dim[-1]] = last_dim_size
    dtype = utils.corresponding_real_dtype(input.dtype)
    strides = utils.make_contiguous_strides_for(shape)
    return TensorMeta(shape=shape, strides=strides, dtype=dtype, device=input.device)


def _fft_c2r_aten(
    input: TensorLike,
    *,
    dim: DimsSequenceType,
    last_dim_size: int,
) -> TensorLikeType:
    normalization = 0  # No normalization
    return torch._fft_c2r(input, dim, normalization, last_dim_size)


_fft_c2r_doc = """
    Performs a complex to real Inverse Fast Fourier Transform
"""


fft_c2r = _make_prim(
    schema="fft_c2r(Tensor self, *, int[] dim, SymInt last_dim_size) -> Tensor",
    meta=_fft_c2r_meta,
    impl_aten=_fft_c2r_aten,
    return_type=RETURN_TYPE.NEW,
    doc=_fft_c2r_doc,
)


def _frexp_meta(self: TensorLikeType) -> tuple[TensorLikeType, TensorLikeType]:
    torch._check(
        self.dtype.is_floating_point,
        lambda: "torch.frexp() only supports floating-point dtypes",
    )
    return torch.empty_like(self), torch.empty_like(self, dtype=torch.int32)


frexp = _make_prim(
    schema="frexp(Tensor self) -> (Tensor mantissa, Tensor exponent)",
    meta=_frexp_meta,
    return_type=(RETURN_TYPE.NEW, RETURN_TYPE.NEW),
    impl_aten=torch.frexp,
    doc="",
)


def _make_token_aten() -> TensorLikeType:
    return new_token_tensor()


_make_token = _make_prim(
    schema="_make_token() -> Tensor",
    meta=_make_token_aten,
    return_type=RETURN_TYPE.NEW,
    impl_aten=_make_token_aten,
    doc="Creates a token used for keeping track of side effects.",
)


def _sink_tokens_aten(tokens) -> None:
    pass


_sink_tokens = _make_prim(
    schema="_sink_tokens(Tensor[] tokens) -> ()",
    meta=_sink_tokens_aten,
    return_type=RETURN_TYPE.NONE,
    impl_aten=_sink_tokens_aten,
    doc="Sink all of the tokens which were previously used for keeping track of side effects.",
)

torch.fx.node.has_side_effect(_sink_tokens)


register_rng_prims()
register_debug_prims()
