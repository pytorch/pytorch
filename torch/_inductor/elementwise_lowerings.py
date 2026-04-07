# mypy: allow-untyped-defs
from __future__ import annotations

import math
from unittest.mock import patch

import sympy

import torch

from .ir import BaseView, MutableBox
from .lowering import (
    _register_foreach_lowering,
    aten,
    clone,
    copy,
    ELEMENTWISE_TYPE_PROMOTION_KIND,
    expand,
    fallback_handler,
    floor,
    full_like,
    get_promoted_dtype,
    inplace_foreach_ops,
    inplaceable_foreach_ops,
    ir,
    is_boolean_type,
    is_float_dtype,
    is_integer_dtype,
    is_integer_type,
    make_pointwise,
    Number,
    ops,
    ops_wrapper,
    Pointwise,
    prims,
    promote_constants,
    register_foreach_pointwise,
    register_lowering,
    register_pointwise,
    TensorBox,
    to_device,
    to_dtype,
    trunc,
    V,
)
from .utils import register_op_requires_libdevice_fp64


def pow_recursive(x, y, dtype):
    if y < 0:
        return pow_recursive(ops.reciprocal(x), -y, dtype)
    if y == 0:
        return ops.constant(1, dtype)
    if y == 1:
        return x

    result = pow_recursive(x, y // 2, dtype)
    result = ops.mul(result, result)
    if (y % 2) == 1:
        result = ops.mul(result, x)
    return result


@make_pointwise
def pow_native(a, b):
    return ops.pow(a, b)


fallback_pow_tensor_tensor = fallback_handler(
    aten.pow.Tensor_Tensor, add_to_fallback_set=False
)
fallback_pow_scalar = fallback_handler(aten.pow.Scalar, add_to_fallback_set=False)
fallback_pow_tensor_scalar = fallback_handler(
    aten.pow.Tensor_Scalar, add_to_fallback_set=False
)


@register_lowering(aten.pow, broadcast=True)
def pow(a, b):
    if isinstance(b, float) and b.is_integer():
        return pow(a, int(b))
    elif isinstance(b, float) and b == 0.5:
        return sqrt(a)
    elif isinstance(b, int) and b == 1:
        return clone(a)

    # Type promotion ensures all tensor arguments have the same type
    dtype = next(x.get_dtype() for x in (a, b) if isinstance(x, ir.TensorBox))
    is_integer_pow = is_integer_dtype(dtype)

    # Optimize away small fixed powers, or for integers avoid falling back to ATen
    embed_exponent = isinstance(b, int) and (
        -32 < b < 32 or (is_integer_pow and b >= 0)
    )
    if embed_exponent:
        loader = a.make_loader()

        def fn(idx):
            return pow_recursive(loader(idx), b, a.get_dtype())

        return Pointwise.create(
            device=a.get_device(),
            dtype=a.get_dtype(),
            inner_fn=fn,
            ranges=a.get_size(),
        )

    if isinstance(a, Number):
        if a == 1:
            return full_like(b, 1)

        if a == 2 and is_float_dtype(b.get_dtype()):
            return exp2(b)

    if is_integer_pow:
        # ops.pow doesn't work for integers
        if isinstance(a, Number):
            return fallback_pow_scalar(a, b)
        elif isinstance(b, Number):
            return fallback_pow_tensor_scalar(a, b)
        else:
            return fallback_pow_tensor_tensor(a, b)

    return pow_native(a, b)


def mutate_to(changed, val, unsafe_alias=False):
    if isinstance(changed, TensorBox):
        changed_data = changed.data
    else:
        changed_data = changed
    if isinstance(val, TensorBox):
        val = val.data

    if not isinstance(val, ir.StorageBox):
        # introduce a copy to handle views
        node = Pointwise.create(
            device=changed.get_device(),
            dtype=changed.get_dtype(),
            inner_fn=val.make_loader(),
            ranges=changed.get_size(),
        )
        assert isinstance(node, (BaseView, MutableBox))
        val = node.data
        assert isinstance(val, ir.StorageBox)

    if isinstance(changed_data, ir.StorageBox) and not (
        changed_data.is_input_buffer()
        # In AOTI, module parameters and buffers are not lifted as graph inputs
        or changed_data.is_module_buffer()
        or isinstance(changed_data.data, ir.NopKernel)
    ):
        # Fast path, just swing the data pointer
        val.realize()
        changed_data.data = val.data
        return changed

    ir.MutationLayoutSHOULDREMOVE.realize_into(
        val, changed_data, unsafe_alias=unsafe_alias
    )
    return changed


@register_lowering(aten.fill_)
def fill_(x, fill_value):
    return mutate_to(x, full_like(x, fill_value))


@register_lowering(aten.copy_, type_promotion_kind=None)
def copy_(dst, src, non_blocking=False):
    if dst is src:
        # dst.copy_(dst) can happen from the reinplacing pass
        return dst
    src = to_device(src, dst.get_device())
    src = to_dtype(src, dst.get_dtype())
    src = expand(src, dst.get_size())
    return mutate_to(dst, src)


@make_pointwise
def floordiv(a, b):
    return ops.floordiv(a, b)


@make_pointwise
def truncdiv(a, b):
    return ops.truncdiv(a, b)


@make_pointwise
def _div_rn(a, b):
    return ops.div_rn(a, b)


@register_lowering(aten.div, broadcast=True)
def div_mode(a, b, rounding_mode=None):
    both_integer = is_integer_type(a) and is_integer_type(b)
    both_boolean = is_boolean_type(a) and is_boolean_type(b)

    # floordiv and truncdiv need special handling for integer tensors on Triton,
    # see the discussion at https://github.com/triton-lang/triton/issues/605
    if rounding_mode == "floor":
        assert not both_boolean, "floordiv operands can not be boolean at the same time"
        # Use div_rn (IEEE round-to-nearest) instead of truediv here because
        # Triton's default division uses an approximate reciprocal, which can
        # produce a result slightly below the true quotient and cause floor()
        # to round down by one.
        return floordiv(a, b) if both_integer else floor(_div_rn(a, b))
    if rounding_mode == "trunc":
        assert not both_boolean, "truncdiv operands can not be boolean at the same time"
        return truncdiv(a, b) if both_integer else trunc(div(a, b))
    return div(a, b)


@register_lowering([aten.mul], broadcast=True)
def mul(a, b):
    both_bool = is_boolean_type(a) and is_boolean_type(b)
    if both_bool:
        return logical_and(a, b)
    else:
        fn = ops_wrapper(aten.mul.__name__)
        return make_pointwise(fn)(a, b)


def get_constant_value(x: ir.IRNode) -> ir.Constant | None:
    """Try convert an arbitrary IR node into an ir.Constant value"""

    # First try unwrapping the IRNode to see if it is already an ir.Constant
    # Optional step, but avoids unnecessary inner_fn evaluation.
    if isinstance(x, ir.MutableBox):
        return get_constant_value(x.data)
    if isinstance(x, ir.BaseView):
        return get_constant_value(x.unwrap_view())
    if isinstance(x, ir.Constant):
        return x

    # If the unwrapped node is not an ir.Constant, try evaluating inner_fn
    # to see if the returned value is from an `ops.constant` call
    if not isinstance(x, ir.Loops):
        return None

    handler = torch._inductor.ops_handler.ExtractConstantsHandler(x.get_device())
    with (
        V.set_ops_handler(handler),
        patch.object(ir.FlexibleLayout, "allow_indexing", True),
    ):
        out = x.inner_fn(*x.inner_fn_args())

    assert isinstance(out, torch._inductor.virtualized.OpsValue)
    if isinstance(out.value, ir.Constant):
        return out.value
    return None


# NOTE: prims.div maps to a / b in C, so performs truncation division on
#   integer inputs and true division for floating and complex inputs.
@register_lowering([prims.div], broadcast=True)
def div_prim(a, b):
    is_integral = all(is_boolean_type(x) or is_integer_type(x) for x in [a, b])

    if is_integral:
        return truncdiv(a, b)

    # Disable CPU optimization to avoid precision issues.
    # see https://github.com/pytorch/pytorch/issues/157959
    if (divisor := get_constant_value(b)) is not None and a.get_device().type != "cpu":
        # Replace divide by constant with multiply by reciprocal

        if divisor.value == 0:
            reciprocal = math.copysign(float("inf"), divisor.value)
        else:
            reciprocal = 1.0 / divisor.value
        return mul(a, reciprocal)

    def fn(*args):
        return ops.truediv(*args)

    return make_pointwise(fn)(a, b)


@register_lowering(
    [aten.true_divide, aten.div.Tensor],
    broadcast=True,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def div(a, b):
    a, b = promote_constants(
        (a, b), type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    )
    return div_prim(a, b)


@register_lowering([aten.fmod, prims.fmod], broadcast=True)
def fmod(a, b):
    is_integral = is_boolean_type(a) or is_integer_type(a)

    if is_integral:

        def fn(a, b):
            return ops.mod(a, b)

    else:

        def fn(a, b):
            return ops.fmod(a, b)

    return make_pointwise(fn)(a, b)


add = register_pointwise(
    aten.add,
    allow_alpha=True,
    use_fma_for_alpha=True,
    override_fn_when_input_bool="logical_or",
)


def register_pointwise_numeric(op, name=None, triton_fallback=None):
    return register_pointwise(
        op,
        name=name,
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        triton_fallback=triton_fallback,
    )


def register_pointwise_numeric_ldf64(op: torch._ops.OpOverloadPacket):
    register_op_requires_libdevice_fp64(op.__name__)
    return register_pointwise(
        op,
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
    )


rsqrt = register_pointwise_numeric(aten.rsqrt)
exp = register_pointwise_numeric_ldf64(aten.exp)
exp2 = register_pointwise_numeric(aten.exp2)
expm1 = register_pointwise_numeric(aten.expm1)
relu = register_pointwise(aten.relu)
sigmoid = register_pointwise_numeric_ldf64(aten.sigmoid)
sqrt = register_pointwise_numeric_ldf64(aten.sqrt)
square = register_pointwise(aten.square)
sub = register_pointwise(aten.sub, allow_alpha=True)


@register_lowering(aten.addcmul, broadcast=True)
def addcmul(self, tensor1, tensor2, *, value=1):
    """
    Computes self + value * tensor1 * tensor2 using FMA for better precision.

    Matches eager CUDA kernel order: self + value * (tensor1 * tensor2)
    This is computed as: fma(value, tensor1 * tensor2, self)

    Note: FMA is only used for floating-point types on non-AMD GPUs. For integer types,
    we fall back to regular arithmetic since FMA doesn't support integers.

    For floating-point types, we use mul_rn (round-to-nearest multiplication)
    to force rounding of the product before the FMA. This prevents Triton's
    compiler from fusing the multiplication with the FMA, matching eager's
    rounding behavior.
    """
    dtype = get_promoted_dtype(
        self,
        tensor1,
        tensor2,
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    )

    self_loader = self.make_loader()
    t1_loader = tensor1.make_loader()
    t2_loader = tensor2.make_loader()

    # FMA/mul_rn/div_rn are only available for floating-point types on CUDA (non-AMD)
    device = self.get_device()
    use_fma = (
        dtype.is_floating_point
        and not torch.version.hip
        and device is not None
        and device.type in ["cuda", "xpu"]
    )

    def inner_fn(idx):
        self_val = self_loader(idx)
        t1_val = t1_loader(idx)
        t2_val = t2_loader(idx)

        if value == 1 and use_fma:
            return ops.fma(t1_val, t2_val, self_val)

        # Match eager order: self + value * (tensor1 * tensor2)
        # Compute tensor1 * tensor2 first
        if use_fma:
            # Use mul_rn to force rounding of the product, preventing Triton
            # from fusing t1*t2 with the subsequent FMA
            t1_times_t2 = ops.mul_rn(t1_val, t2_val)
        else:
            t1_times_t2 = ops.mul(t1_val, t2_val)

        # Use index_expr for sympy expressions (e.g., from .item()), constant otherwise
        if isinstance(value, sympy.Basic):
            value_expr = ops.index_expr(value, dtype)
        else:
            value_expr = ops.constant(value, dtype)

        if use_fma:
            # Use FMA for floating-point types for better precision
            return ops.fma(value_expr, t1_times_t2, self_val)
        else:
            # Fall back to regular arithmetic for integer types
            return ops.add(self_val, ops.mul(value_expr, t1_times_t2))

    return Pointwise.create(
        device=self.get_device(),
        dtype=dtype,
        inner_fn=inner_fn,
        ranges=self.get_size(),
    )


@register_lowering(aten.addcdiv, broadcast=True)
def addcdiv(self, tensor1, tensor2, *, value=1):
    """
    Computes self + value * (tensor1 / tensor2) using FMA for better precision.

    Matches eager CUDA kernel order: self + value * (tensor1 / tensor2)
    This is computed as: fma(value, tensor1 / tensor2, self)

    For value=1: self + tensor1 / tensor2 (no FMA needed, just add the division)
    For value!=1: fma(value, div_rn(tensor1, tensor2), self)

    Note: FMA is only used for floating-point types on non-AMD GPUs. For integer types,
    we fall back to regular arithmetic since FMA doesn't support integers.

    We use div_rn (round-to-nearest division) to force proper rounding, preventing
    Triton from fusing operations in ways that change the rounding behavior.
    """
    dtype = get_promoted_dtype(
        self,
        tensor1,
        tensor2,
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
    )

    self_loader = self.make_loader()
    t1_loader = tensor1.make_loader()
    t2_loader = tensor2.make_loader()

    # FMA/mul_rn/div_rn are only available for floating-point types on CUDA (non-AMD)
    device = self.get_device()
    use_fma = (
        dtype.is_floating_point
        and not torch.version.hip
        and device is not None
        and device.type in ["cuda", "xpu"]
    )

    def inner_fn(idx):
        self_val = self_loader(idx)
        t1_val = t1_loader(idx)
        t2_val = t2_loader(idx)

        # Compute tensor1 / tensor2 first
        # Use div_rn for round-to-nearest division on CUDA to match eager behavior
        if use_fma:
            t1_div_t2 = ops.div_rn(t1_val, t2_val)
        else:
            t1_div_t2 = ops.truediv(t1_val, t2_val)

        if value == 1:
            # For value=1, just add the division result (no FMA needed)
            return ops.add(self_val, t1_div_t2)

        # Use index_expr for sympy expressions (e.g., from .item()), constant otherwise
        if isinstance(value, sympy.Basic):
            value_expr = ops.index_expr(value, dtype)
        else:
            value_expr = ops.constant(value, dtype)

        if use_fma:
            # Use FMA for floating-point types for better precision
            return ops.fma(value_expr, t1_div_t2, self_val)
        else:
            # Fall back to regular arithmetic for integer types
            return ops.add(self_val, ops.mul(value_expr, t1_div_t2))

    return Pointwise.create(
        device=self.get_device(),
        dtype=dtype,
        inner_fn=inner_fn,
        ranges=self.get_size(),
    )


_foreach_addcmul_scalar = register_foreach_pointwise(
    aten._foreach_addcmul.Scalar, addcmul, allow_alpha=True, scalar_kwarg="value"
)
_foreach_addcdiv_scalar = register_foreach_pointwise(
    aten._foreach_addcdiv.Scalar, addcdiv, allow_alpha=True, scalar_kwarg="value"
)


register_pointwise_numeric_ldf64(aten.cos)
register_pointwise_numeric_ldf64(aten.sin)
bitwise_and = register_pointwise(aten.bitwise_and)
bitwise_left_shift = register_pointwise(aten.bitwise_left_shift)
bitwise_not = register_pointwise(
    aten.bitwise_not, override_fn_when_input_bool="logical_not"
)
bitwise_or = register_pointwise(aten.bitwise_or)
bitwise_right_shift = register_pointwise(aten.bitwise_right_shift)
bitwise_xor = register_pointwise(aten.bitwise_xor)
register_pointwise_numeric(aten.lgamma)
erf = register_pointwise_numeric(aten.erf)
register_lowering(
    aten.special_erf, type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
)(erf)

register_pointwise_numeric(aten.log1p)
register_pointwise_numeric(aten.tan)
register_pointwise_numeric(aten.tanh)
register_pointwise_numeric_ldf64(aten.log)
logical_and = register_pointwise(
    aten.logical_and,
    type_promotion_kind=None,
    convert_input_to_bool=True,
    override_return_dtype=torch.bool,
)
logical_not = register_pointwise(
    aten.logical_not,
    type_promotion_kind=None,
    convert_input_to_bool=True,
    override_return_dtype=torch.bool,
)
logical_or = register_pointwise(
    aten.logical_or,
    type_promotion_kind=None,
    convert_input_to_bool=True,
    override_return_dtype=torch.bool,
)
logical_xor = register_pointwise(
    aten.logical_xor,
    type_promotion_kind=None,
    convert_input_to_bool=True,
    override_return_dtype=torch.bool,
)
maximum = register_pointwise(aten.maximum)
minimum = register_pointwise(aten.minimum)
register_lowering(aten.clamp_min)(maximum)
register_lowering(aten.clamp_max)(minimum)
neg = register_pointwise(aten.neg)
abs = register_pointwise(aten.abs)
reciprocal = register_pointwise_numeric(aten.reciprocal)
register_pointwise(aten.remainder)
sign = register_pointwise(aten.sign, override_fn_when_input_bool="identity")
register_pointwise(aten.ceil)
register_pointwise(aten.signbit, override_return_dtype=torch.bool)

register_lowering(aten._neg_view)(neg)

register_pointwise(aten.le, override_return_dtype=torch.bool)
register_pointwise(aten.lt, override_return_dtype=torch.bool)
register_pointwise(aten.ge, override_return_dtype=torch.bool)
gt = register_pointwise(aten.gt, override_return_dtype=torch.bool)
register_pointwise(aten.eq, override_return_dtype=torch.bool)
register_pointwise(aten.ne, override_return_dtype=torch.bool)

register_pointwise_numeric(aten.cosh)
register_pointwise_numeric(aten.sinh)
register_pointwise_numeric(aten.acos)
register_pointwise_numeric(aten.acosh)
register_pointwise_numeric(aten.asin)
register_pointwise_numeric(aten.asinh)
register_pointwise_numeric(aten.atan2)
register_pointwise_numeric(aten.atan)
register_pointwise_numeric(aten.atanh)
register_pointwise_numeric(aten.copysign)
register_pointwise_numeric(aten.erfc)
register_pointwise_numeric(aten.erfinv)
register_pointwise_numeric(aten.hypot)
register_pointwise_numeric(aten.log10)
register_pointwise_numeric(aten.log2)
register_pointwise_numeric(aten.nextafter)

from .codegen.common import pointwise_overrides_data


def _get_pointwise_overrides(ns, name):
    data = pointwise_overrides_data[name]
    op = getattr(ns, data.name, None)
    if op is None:
        return

    def make_triton_fallback(op):
        if data.triton is None:
            return fallback_handler(op)

    if isinstance(op, torch._ops.OpOverloadPacket):
        for olname in op.overloads():
            ol = getattr(op, olname)
            yield ol, data.type_promotion_kind, make_triton_fallback(ol)
    else:
        yield op, data.type_promotion_kind, make_triton_fallback(op)


for name in pointwise_overrides_data:
    for op, type_promotion_kind, triton_fallback in _get_pointwise_overrides(
        aten, name
    ):
        register_pointwise(
            op,
            name=name,
            type_promotion_kind=type_promotion_kind,
            triton_fallback=triton_fallback,
        )

    for op, type_promotion_kind, triton_fallback in _get_pointwise_overrides(
        prims, name
    ):
        register_pointwise(
            op,
            name=name,
            type_promotion_kind=type_promotion_kind,
            triton_fallback=triton_fallback,
        )


foreach_add_list = register_foreach_pointwise(
    aten._foreach_add.List, add, allow_alpha=True
)
foreach_add_scalar = register_foreach_pointwise(
    aten._foreach_add.Scalar, add, allow_alpha=True
)
register_foreach_pointwise(aten._foreach_add.Tensor, add, allow_alpha=True)
foreach_mul_list = register_foreach_pointwise(aten._foreach_mul.List, mul)
register_foreach_pointwise(aten._foreach_mul.Tensor, mul)
foreach_mul_scalar = register_foreach_pointwise(aten._foreach_mul.Scalar, mul)
register_foreach_pointwise(aten._foreach_sub.List, sub)
register_foreach_pointwise(aten._foreach_sub.Scalar, sub)
register_foreach_pointwise(aten._foreach_neg.default, neg)
register_foreach_pointwise(aten._foreach_abs.default, abs)
register_foreach_pointwise(aten._foreach_pow.Scalar, pow)
register_foreach_pointwise(aten._foreach_pow.List, pow)
register_foreach_pointwise(aten._foreach_pow.ScalarAndTensor, pow)
foreach_div_list = register_foreach_pointwise(aten._foreach_div.List, div)
register_foreach_pointwise(aten._foreach_div.Tensor, div)
foreach_div_scalar = register_foreach_pointwise(aten._foreach_div.Scalar, div)
register_foreach_pointwise(aten._foreach_sqrt, sqrt)
register_foreach_pointwise(aten._foreach_rsqrt, rsqrt)
register_foreach_pointwise(aten._foreach_maximum.List, maximum)
register_foreach_pointwise(aten._foreach_maximum.Scalar, maximum)
register_foreach_pointwise(aten._foreach_minimum.List, minimum)
register_foreach_pointwise(aten._foreach_minimum.Scalar, minimum)
register_foreach_pointwise(aten._foreach_clamp_min.List, maximum)
register_foreach_pointwise(aten._foreach_clamp_min.Scalar, maximum)
register_foreach_pointwise(aten._foreach_clamp_max.List, minimum)
register_foreach_pointwise(aten._foreach_clamp_max.Scalar, minimum)
register_foreach_pointwise(aten._foreach_reciprocal, reciprocal)
register_foreach_pointwise(aten._foreach_sign, sign)
register_foreach_pointwise(aten._foreach_clone, clone)
foreach_copy = register_foreach_pointwise(aten._foreach_copy, copy)


# these are only encountered as outputs of the graph
# reinplacing epilogue copies improves compile time
# by removing extra buffers sent to the scheduler.
def register_foreach_inplace(aten_op, outplace_aten_op, outplace_op):
    inplaceable_foreach_ops[outplace_aten_op] = aten_op
    inplace_foreach_ops.add(aten_op)

    def fn(*args, **kwargs):
        results = outplace_op(*args, **kwargs)
        mut_results = []
        for arg, result in zip(args[0], results):
            mut_results.append(mutate_to(arg, result, unsafe_alias=True))

        return mut_results

    _register_foreach_lowering(aten_op, fn)


register_foreach_inplace(
    aten._foreach_add_.List, aten._foreach_add.List, foreach_add_list
)
register_foreach_inplace(
    aten._foreach_add_.Scalar, aten._foreach_add.Scalar, foreach_add_scalar
)
register_foreach_inplace(
    aten._foreach_mul_.List, aten._foreach_mul.List, foreach_mul_list
)
register_foreach_inplace(
    aten._foreach_mul_.Scalar, aten._foreach_mul.Scalar, foreach_mul_scalar
)
register_foreach_inplace(
    aten._foreach_div_.List, aten._foreach_div.List, foreach_div_list
)
register_foreach_inplace(
    aten._foreach_div_.Scalar, aten._foreach_div.Scalar, foreach_div_scalar
)
register_foreach_inplace(
    aten._foreach_copy_.default, aten._foreach_copy.default, foreach_copy
)
register_foreach_inplace(
    aten._foreach_addcmul_.Scalar,
    aten._foreach_addcmul.Scalar,
    _foreach_addcmul_scalar,
)
register_foreach_inplace(
    aten._foreach_addcdiv_.Scalar,
    aten._foreach_addcdiv.Scalar,
    _foreach_addcdiv_scalar,
)


def register_inplace(aten_op, outplace_op):
    @register_lowering(aten_op, type_promotion_kind=None)
    def fn(*args, **kwargs):
        result = outplace_op(*args, **kwargs)
        result = to_dtype(result, args[0].get_dtype())
        return mutate_to(args[0], result)

    return fn


register_inplace(aten.add_, add)
register_inplace(aten.bitwise_and_, bitwise_and)
register_inplace(aten.bitwise_left_shift_, bitwise_left_shift)
register_inplace(aten.bitwise_not_, bitwise_not)
register_inplace(aten.bitwise_or_, bitwise_or)
register_inplace(aten.bitwise_right_shift_, bitwise_right_shift)
register_inplace(aten.bitwise_xor_, bitwise_xor)
register_inplace(aten.mul_, mul)
register_inplace(aten.div_.Tensor, div)
register_inplace(aten.div_.Tensor_mode, div_mode)
register_inplace(aten.logical_and_, logical_and)
register_inplace(aten.logical_not_, logical_not)
register_inplace(aten.logical_or_, logical_or)
register_inplace(aten.logical_xor_, logical_xor)
register_inplace(aten.sub_, sub)
register_inplace(aten.relu_, relu)
register_inplace(aten.sigmoid_, sigmoid)


register_lowering(aten.__and__)(bitwise_and)
register_lowering(aten.__lshift__)(bitwise_left_shift)
register_lowering(aten.__or__)(bitwise_or)
register_lowering(aten.__rshift__)(bitwise_right_shift)
register_lowering(aten.__xor__)(bitwise_xor)

register_inplace(aten.__iand__, aten.__and__)
register_inplace(aten.__ilshift__, aten.__lshift__)
register_inplace(aten.__ior__, aten.__or__)
register_inplace(aten.__irshift__, aten.__rshift__)
register_inplace(aten.__ixor__, aten.__xor__)

__all__ = [
    "_foreach_addcdiv_scalar",
    "_foreach_addcmul_scalar",
    "abs",
    "add",
    "addcdiv",
    "addcmul",
    "bitwise_and",
    "bitwise_left_shift",
    "bitwise_not",
    "bitwise_or",
    "bitwise_right_shift",
    "bitwise_xor",
    "copy_",
    "div",
    "div_prim",
    "div_mode",
    "exp",
    "exp2",
    "expm1",
    "fill_",
    "floordiv",
    "fmod",
    "foreach_add_list",
    "foreach_add_scalar",
    "foreach_copy",
    "foreach_div_list",
    "foreach_div_scalar",
    "foreach_mul_list",
    "foreach_mul_scalar",
    "get_constant_value",
    "gt",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "maximum",
    "minimum",
    "mul",
    "mutate_to",
    "neg",
    "pow",
    "pow_native",
    "pow_recursive",
    "reciprocal",
    "register_inplace",
    "register_pointwise_numeric",
    "register_pointwise_numeric_ldf64",
    "relu",
    "rsqrt",
    "sigmoid",
    "sign",
    "sqrt",
    "square",
    "sub",
    "truncdiv",
]
