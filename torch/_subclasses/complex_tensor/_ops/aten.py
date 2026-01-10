from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import torch

from .._core import ComplexTensor
from .common import (
    _get_func_name,
    COMPLEX_TO_REAL,
    complex_to_real_dtype,
    is_complex,
    OpType,
    promote_tensors,
    register_binary_nonlinear,
    register_complex,
    register_error,
    register_force_test,
    register_simple,
    split_complex_arg,
    split_complex_tensor,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

from typing import Any


aten = torch.ops.aten


def register_binary_linear(op: OpType) -> Callable[..., Any]:
    def impl_with_alpha(
        lhs: ComplexTensor,
        rhs: ComplexTensor,
        *args: Any,
        alpha: complex,
        **kwargs: Any,
    ) -> ComplexTensor:
        return op(lhs, aten.mul(rhs, alpha, *args, **kwargs), *args, **kwargs)

    def impl(
        lhs: ComplexTensor, rhs: ComplexTensor, *args: Any, **kwargs: Any
    ) -> ComplexTensor:
        alpha = kwargs.pop("alpha", None)
        if alpha is not None:
            return impl_with_alpha(lhs, rhs, *args, alpha=alpha, **kwargs)
        a_r, a_i = split_complex_arg(lhs)
        b_r, b_i = split_complex_arg(rhs)
        out_dt, (a_r, a_i, b_r, b_i) = promote_tensors(a_r, a_i, b_r, b_i)
        u = op(a_r, b_r, *args, **kwargs)
        v = op(a_i, b_i, *args, **kwargs)
        return ComplexTensor(u.to(out_dt), v.to(out_dt))

    return register_complex(op, impl)


@register_complex(aten.real)
def real_impl(self: ComplexTensor) -> torch.Tensor:
    re, _ = split_complex_tensor(self)
    return re


@register_complex(aten.imag)
def imag_impl(self: ComplexTensor) -> torch.Tensor:
    _, im = split_complex_tensor(self)
    return im


@register_complex(aten.is_pinned)
def is_pinned_impl(self: ComplexTensor, device: torch.device | None = None) -> bool:
    return self.is_pinned(device)


SIMPLE_OPS_LIST = [
    aten.slice,
    aten.flatten,
    aten.view,
    aten.diagonal,
    aten.expand,
    aten.unsqueeze,
    aten.unsqueeze_,
    aten.mean,
    aten.sum,
    aten.clone,
    aten.neg,
    aten.flip,
    aten.permute,
    aten.repeat,
    aten.index_select,
    aten.split,
    aten.split_with_sizes,
    aten.cumsum,
    aten.detach,
    aten.select,
    aten.squeeze,
    aten.zero_,
    aten.transpose,
    aten.t,
    aten.gather,
]

for simple_op in SIMPLE_OPS_LIST:
    globals()[_get_func_name(simple_op)] = register_simple(simple_op)

# TODO (hameerabbasi): Not being tested
SIMPLE_FORCE_TESTED_OPS = [
    aten.copy,
    aten.col2im,
    aten.alias,
    aten.lift_fresh,
    aten._unsafe_view,
    aten.index,
    aten._neg_view,
    aten.avg_pool2d,
    aten.avg_pool3d,
    aten.avg_pool2d_backward,
    aten.avg_pool3d_backward,
    aten.masked_scatter_backward,
    aten.select_backward,
    aten.slice_backward,
    aten.embedding,
]

for simple_op in SIMPLE_FORCE_TESTED_OPS:
    globals()[_get_func_name(simple_op)] = register_force_test(
        simple_op, register_simple(simple_op)
    )

del simple_op

# some binary ops which we can stamp out
mul_impl = register_binary_nonlinear(aten.mul)
mul__impl = register_binary_nonlinear(aten.mul_)
mm_impl = register_binary_nonlinear(aten.mm)
dot_impl = register_binary_nonlinear(aten.dot)
bmm_impl = register_binary_nonlinear(aten.bmm)

# TODO (hameerabbasi): Not being tested
convolution_impl = register_force_test(
    aten.convolution, register_binary_nonlinear(aten.convolution)
)

slice_scatter_impl = register_force_test(
    aten.slice_scatter, register_binary_linear(aten.slice_scatter)
)
select_scatter_impl = register_force_test(
    aten.select_scatter, register_binary_linear(aten.select_scatter)
)

add_impl = register_binary_linear(aten.add)
add__impl = register_binary_linear(aten.add_)
sub_impl = register_binary_linear(aten.sub)
sub__impl = register_binary_linear(aten.sub_)
diagonal_scatter_impl = register_binary_linear(aten.diagonal_scatter)
fill__impl = register_binary_linear(aten.fill_)


@register_complex(aten.rsub)
def rsub_impl(
    lhs: ComplexTensor, rhs: ComplexTensor, alpha: complex | None = None
) -> ComplexTensor:
    if alpha is None:
        return torch.sub(rhs, lhs)  # type: ignore[bad-return]
    return torch.sub(rhs, lhs, alpha=alpha)  # type: ignore[bad-return]


@register_complex(aten.div)
@register_complex(aten.true_divide)
def div_impl(
    lhs: ComplexTensor, rhs: ComplexTensor, *, rounding_mode: str | None = None
) -> ComplexTensor:
    if rounding_mode is not None:
        raise NotImplementedError(
            "`rounding_mode` other than `None` not implemented for`ComplexTensor`."
        )
    a_r, a_i = split_complex_arg(lhs)
    if not is_complex(rhs):
        return ComplexTensor(a_r / rhs, a_i / rhs)
    b_r, b_i = split_complex_arg(rhs)
    out_dt, (a_r, a_i, b_r, b_i) = promote_tensors(a_r, a_i, b_r, b_i)
    num_r = a_r * b_r + a_i * b_i
    num_i = a_i * b_r - a_r * b_i
    den = b_r * b_r + b_i * b_i
    return ComplexTensor(
        (num_r / den).to(out_dt),
        (num_i / den).to(out_dt),
    )


@register_complex(aten.reciprocal)
def reciprocal_impl(self: ComplexTensor) -> ComplexTensor:
    self_r, self_i = split_complex_tensor(self)
    out_dt, (self_r, self_i) = promote_tensors(self_r, self_i)
    den = self_r * self_r + self_i * self_i
    return ComplexTensor(
        aten.div(self_r, den).to(out_dt),
        aten.div(-self_i, den).to(out_dt),
    )


# reductions
@register_complex(aten.prod)
def prod_impl(self: ComplexTensor, *args: Any, **kwargs: Any) -> ComplexTensor:
    out_dt, (self,) = promote_tensors(self)
    dtype = kwargs.pop("dtype", out_dt)
    kwargs["dtype"] = complex_to_real_dtype(self.dtype)

    prod_r = torch.prod(torch.abs(self), *args, **kwargs)
    sum_phi = torch.sum(torch.angle(self), *args, **kwargs)
    u = prod_r * torch.cos(sum_phi)
    v = prod_r * torch.sin(sum_phi)
    return ComplexTensor(u, v).to(dtype)  # type: ignore[bad-return]


@register_complex(aten.pow)
def pow_impl(self: ComplexTensor, exponent: ComplexTensor) -> ComplexTensor:
    out_dt, (self, exponent) = promote_tensors(self, exponent)
    return torch.exp(exponent * torch.log(self)).to(out_dt)  # type: ignore[bad-return]


@register_complex(aten.cumprod)
def cumprod_impl(self: ComplexTensor, *args: Any, **kwargs: Any) -> ComplexTensor:
    dtype = kwargs.pop("dtype", self.dtype)
    kwargs["dtype"] = complex_to_real_dtype(dtype)

    prod_r = torch.cumprod(torch.abs(self), *args, **kwargs)
    sum_phi = torch.cumsum(torch.angle(self), *args, **kwargs)
    u = prod_r * torch.cos(sum_phi)
    v = prod_r * torch.sin(sum_phi)
    return ComplexTensor(u, v)


# unary funcs,
# most of these are simple or require some kind of identity
@register_complex(aten.abs)
def abs_impl(self: ComplexTensor) -> torch.Tensor:
    x, y = split_complex_tensor(self)
    out_dt, (x, y) = promote_tensors(x, y)
    result = torch.hypot(x, y)
    return result.to(out_dt)


@register_complex(aten.angle)
def angle_impl(self: ComplexTensor) -> torch.Tensor:
    x, y = split_complex_tensor(self)
    return torch.atan2(y, x)


@register_complex(aten.acos)
def acos_impl(self: ComplexTensor) -> ComplexTensor:
    _, y = split_complex_tensor(self)
    acosh_z = torch.acosh(self)
    assert isinstance(acosh_z, ComplexTensor)
    acosh_z_re, acosh_z_im = split_complex_tensor(acosh_z)
    sign_im = 2 * torch.signbit(y) - 1
    return ComplexTensor(torch.abs(acosh_z_im), sign_im * torch.abs(acosh_z_re))


@register_complex(aten.asin)
def asin_impl(self: ComplexTensor) -> ComplexTensor:
    x, y = split_complex_tensor(self)
    asinh_iz = torch.asinh(ComplexTensor(-y, x))
    assert isinstance(asinh_iz, ComplexTensor)
    asinh_iz_re, asinh_iz_im = split_complex_tensor(asinh_iz)
    return ComplexTensor(asinh_iz_im, -asinh_iz_re)


@register_complex(aten.atan)
def atan_impl(self: ComplexTensor) -> ComplexTensor:
    x, y = split_complex_tensor(self)
    tanh_iz = torch.atanh(ComplexTensor(-y, x))
    assert isinstance(tanh_iz, ComplexTensor)
    tanh_iz_re, tanh_iz_im = split_complex_tensor(tanh_iz)
    return ComplexTensor(tanh_iz_im, -tanh_iz_re)


@register_complex(aten.asinh)
def asinh_impl(self: ComplexTensor) -> ComplexTensor:
    out_dt, (self,) = promote_tensors(self)
    return torch.log(self + torch.sqrt(self * self + 1)).to(out_dt)  # type: ignore[bad-return]


@register_complex(aten.acosh)
def acosh_impl(self: ComplexTensor) -> ComplexTensor:
    out_dt, (self,) = promote_tensors(self)
    return torch.log(self + torch.sqrt(self * self - 1)).to(out_dt)  # type: ignore[bad-return]


@register_complex(aten.atanh)
def atanh_impl(self: ComplexTensor) -> ComplexTensor:
    x, y = split_complex_tensor(self)
    out_dt, (x, y) = promote_tensors(x, y)

    ret = 0.5 * (
        torch.log(ComplexTensor(1 + x, y)) - torch.log(ComplexTensor(1 - x, -y))
    )
    assert isinstance(ret, ComplexTensor)
    ret_re, ret_im = split_complex_tensor(ret)

    return ComplexTensor(ret_re.to(out_dt), ret_im.to(out_dt))


@register_complex(aten.cos)
def cos_impl(self: ComplexTensor) -> ComplexTensor:
    x, y = split_complex_tensor(self)
    return torch.cosh(ComplexTensor(-y, x))  # type: ignore[bad-return]


@register_complex(aten.cosh)
def cosh_impl(self: ComplexTensor) -> ComplexTensor:
    x, y = split_complex_tensor(self)
    out_dt, (x, y) = promote_tensors(x, y)
    u = torch.cosh(x) * torch.cos(y)
    v = torch.sinh(x) * torch.sin(y)
    return ComplexTensor(u.to(out_dt), v.to(out_dt))


@register_complex(aten.sin)
def sin_impl(self: ComplexTensor) -> ComplexTensor:
    x, y = split_complex_tensor(self)
    sinh_iz = torch.sinh(ComplexTensor(-y, x))
    assert isinstance(sinh_iz, ComplexTensor)
    sinh_iz_re, sinh_iz_im = split_complex_tensor(sinh_iz)
    return ComplexTensor(sinh_iz_im, -sinh_iz_re)


@register_complex(aten.sinh)
def sinh_impl(self: ComplexTensor) -> ComplexTensor:
    x, y = split_complex_tensor(self)
    out_dt, (x, y) = promote_tensors(x, y)
    u = torch.sinh(x) * torch.cos(y)
    v = torch.cosh(x) * torch.sin(y)
    return ComplexTensor(u.to(out_dt), v.to(out_dt))


@register_complex(aten.tan)
def tan_impl(self: ComplexTensor) -> ComplexTensor:
    x, y = split_complex_tensor(self)
    tanh_iz = torch.tanh(ComplexTensor(-y, x))
    assert isinstance(tanh_iz, ComplexTensor)
    tanh_iz_re, tanh_iz_im = split_complex_tensor(tanh_iz)
    return ComplexTensor(tanh_iz_im, -tanh_iz_re)


@register_complex(aten.tanh)
def tanh_impl(self: ComplexTensor) -> ComplexTensor:
    x, y = split_complex_tensor(self)
    out_dt, (x, y) = promote_tensors(x, y)

    _2x = 2 * x
    _2y = 2 * y
    _d = torch.cosh(_2x) + torch.cos(_2y)
    _2xsh = torch.sinh(_2x)

    out_re = _2xsh / _d
    out_im = torch.sin(_2y) / _d

    return ComplexTensor(out_re.to(out_dt), out_im.to(out_dt))


@register_complex(aten.exp)
def exp_impl(self: ComplexTensor) -> ComplexTensor:
    x, y = split_complex_tensor(self)
    out_dt, (x, y) = promote_tensors(x, y)
    ex = torch.exp(x)
    u = ex * torch.cos(y)
    v = ex * torch.sin(y)
    return ComplexTensor(u.to(out_dt), v.to(out_dt))


@register_complex(aten.expm1)
def expm1_impl(self: ComplexTensor) -> ComplexTensor:
    x, y = split_complex_tensor(self)
    out_dt, (x, y) = promote_tensors(x, y)
    # TODO (hameerabbasi): The two lines below may have numerical issues
    ex = torch.exp(x)
    u = ex * torch.cos(y) - 1
    v = ex * torch.sin(y)
    return ComplexTensor(u.to(out_dt), v.to(out_dt))


@register_complex(aten.log)
def log_impl(self: ComplexTensor) -> ComplexTensor:
    out_dt, (self,) = promote_tensors(self)
    re = torch.log(torch.abs(self))
    im = torch.angle(self)
    return ComplexTensor(re, im).to(out_dt)  # type: ignore[bad-return]


@register_complex(aten.log1p)
def log1p_impl(self: ComplexTensor) -> ComplexTensor:
    x, y = split_complex_tensor(self)
    # TODO (hameerabbasi): The line below may have numerical issues
    return torch.log(ComplexTensor(x + 1, y))  # type: ignore[bad-return]


@register_complex(aten.any)
def any_impl(self: ComplexTensor, *args: Any, **kwargs: Any) -> torch.Tensor:
    x, y = split_complex_tensor(self)
    return torch.any(x, *args, **kwargs) | torch.any(y, *args, **kwargs)


@register_complex(aten.all)
def all_impl(self: ComplexTensor, *args: Any, **kwargs: Any) -> torch.Tensor:
    x, y = split_complex_tensor(self)
    return torch.any(x, *args, **kwargs) & torch.any(y, *args, **kwargs)


@register_complex(aten.eq)
def eq_impl(
    self: ComplexTensor, rhs: ComplexTensor, *args: Any, **kwargs: Any
) -> torch.Tensor:
    a_r, a_i = split_complex_arg(self)
    b_r, b_i = split_complex_arg(rhs)
    return torch.eq(a_r, b_r, *args, **kwargs) & torch.eq(a_i, b_i, *args, **kwargs)


@register_complex(aten.ne)
def ne_impl(
    self: ComplexTensor, rhs: ComplexTensor, *args: Any, **kwargs: Any
) -> torch.Tensor:
    a_r, a_i = split_complex_tensor(self)
    b_r, b_i = split_complex_arg(rhs)
    return torch.ne(a_r, b_r, *args, **kwargs) | torch.ne(a_i, b_i, *args, **kwargs)


@register_complex(aten.isnan)
def isnan_impl(self: ComplexTensor) -> torch.Tensor:
    re, im = split_complex_tensor(self)
    return torch.isnan(re) | torch.isnan(im)


@register_complex(aten.isinf)
def isinf_impl(self: ComplexTensor) -> torch.Tensor:
    re, im = split_complex_tensor(self)
    return torch.isinf(re) | torch.isinf(im)


@register_complex(aten.isfinite)
def isfinite_impl(self: ComplexTensor) -> torch.Tensor:
    re, im = split_complex_tensor(self)
    return torch.isfinite(re) & torch.isfinite(im)


@register_complex(aten.isclose)
def isclose_impl(
    self: ComplexTensor,
    rhs: ComplexTensor,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    equal_nan: bool = False,
) -> torch.Tensor:
    abs_diff = torch.abs(self - rhs)
    abs_other = torch.abs(rhs)
    basic_condition = abs_diff <= (rtol * abs_other + atol)

    # This is the nontrivial part
    if equal_nan:
        a_r, a_i = split_complex_tensor(self)
        b_r, b_i = split_complex_arg(rhs)

        a_r_nan = torch.isnan(a_r)
        b_r_nan = torch.isnan(b_r)
        a_i_nan = torch.isnan(a_i)
        b_i_nan = torch.isnan(b_i)
        a_nan = a_r_nan | a_i_nan

        # This logical expression makes sure that the isnan of both the real and imaginary parts
        # matches (so 1 + nan*i doesn't equal nan + 1*i)
        equal_nan_condition = ((a_r_nan == b_r_nan) & (a_i_nan == b_i_nan)) & a_nan
        return basic_condition | equal_nan_condition

    return basic_condition


ERROR_OPS_LIST = [
    aten.lt,
    aten.le,
    aten.gt,
    aten.ge,
    aten.amin,
    aten.amax,
    aten.clamp,
    aten.ceil,
    aten.floor,
    aten.minimum,
    aten.maximum,
    aten.trunc,
    aten.sign,
    aten.argmax,
    aten.argmin,
    aten.sort,
    aten.topk,
    aten.round,
    aten.fmod,
]


ERROR_TYPES = {
    aten.minimum: RuntimeError,
    aten.maximum: RuntimeError,
    aten.argmax: RuntimeError,
    aten.argmin: RuntimeError,
    aten.sort: RuntimeError,
    aten.topk: RuntimeError,
}


for err_op in ERROR_OPS_LIST:
    globals()[_get_func_name(err_op)] = register_error(
        err_op, ERROR_TYPES.get(err_op, NotImplementedError)
    )

del err_op


@register_complex(aten.masked_scatter)
def masked_scatter_impl(
    self: ComplexTensor, mask: torch.Tensor, source: ComplexTensor
) -> ComplexTensor:
    self_r, self_i = split_complex_tensor(self)
    source_r, source_i = split_complex_arg(source)
    ret_r = torch.masked_scatter(self_r, mask, source_r)
    ret_i = torch.masked_scatter(self_i, mask, source_i)

    return ComplexTensor(ret_r, ret_i)


@register_complex(aten.where)
def where_impl(mask: torch.Tensor, x: ComplexTensor, y: ComplexTensor) -> ComplexTensor:
    x_r, x_i = split_complex_arg(x)
    y_r, y_i = split_complex_arg(y)

    ret_r = torch.where(mask, x_r, y_r)
    ret_i = torch.where(mask, x_i, y_i)

    return ComplexTensor(ret_r, ret_i)


@register_complex(aten.full_like)
def full_like_impl(
    input: ComplexTensor,
    fill_value: complex,
    *args: Any,
    dtype: torch.dtype | None = None,
    **kwargs: Any,
) -> torch.Tensor | ComplexTensor:
    # Note: Cannot be merged with the cases below due to the `fill_value` argument
    input_r, input_i = split_complex_tensor(input)
    if dtype is not None and dtype not in COMPLEX_TO_REAL:
        return torch.full_like(input_r, fill_value, *args, dtype=dtype, **kwargs)

    if dtype is not None:
        kwargs["dtype"] = COMPLEX_TO_REAL[dtype]

    fv_r, fv_i = split_complex_arg(fill_value)
    ret_r = torch.full_like(input_r, fv_r, *args, **kwargs)
    ret_i = torch.full_like(input_i, fv_i, *args, **kwargs)

    return ComplexTensor(ret_r, ret_i)


def register_like(op: OpType) -> Callable[..., Any]:
    def impl(
        self: ComplexTensor, *args: Any, dtype: torch.dtype | None = None, **kwargs: Any
    ) -> torch.Tensor | ComplexTensor:
        self_re, self_im = split_complex_tensor(self)

        if dtype is not None and dtype not in COMPLEX_TO_REAL:
            return op(self_re, *args, dtype=dtype, **kwargs)

        if dtype is not None:
            kwargs["dtype"] = COMPLEX_TO_REAL[dtype]

        ret_re = op(self_re, *args, **kwargs)
        ret_im = op(self_im, *args, **kwargs)

        return ComplexTensor(ret_re, ret_im)

    func_name = _get_func_name(op)
    impl.__name__ = func_name
    impl.__qualname__ = func_name

    return register_complex(op, impl)


LIKE_OPS_LIST = [
    aten.empty_like,
    aten.zeros_like,
    aten.randn_like,
    aten.new_zeros,
]

for like_op in LIKE_OPS_LIST:
    globals()[_get_func_name(like_op)] = register_like(like_op)

del like_op


@register_complex(aten.cat)
def cat_impl(tensors: Sequence[ComplexTensor], dim: int = 0) -> ComplexTensor:
    tensors_r = []
    tensors_i = []

    for t in tensors:
        t_r, t_i = split_complex_arg(t)
        tensors_r.append(t_r)
        tensors_i.append(t_i)

    ret_r = torch.cat(tensors_r, dim=dim)
    ret_i = torch.cat(tensors_i, dim=dim)

    return ComplexTensor(ret_r, ret_i)


@register_complex(aten.sgn)
def sgn_impl(self: ComplexTensor) -> ComplexTensor:
    self_r, self_i = split_complex_tensor(self)
    out_dt, (self_r, self_i) = promote_tensors(self_r, self_i)
    abs_self = torch.abs(ComplexTensor(self_r, self_i))
    mask = (self_r != 0) | (self_i != 0)
    masked_sgn = ComplexTensor(
        (self_r / abs_self).to(out_dt), (self_i / abs_self).to(out_dt)
    )
    return torch.where(mask, masked_sgn, 0)  # type: ignore[bad-return]


@register_complex(aten.sqrt)
def sqrt_impl(self: ComplexTensor) -> ComplexTensor:
    self_r, self_i = split_complex_tensor(self)
    out_dt, (self_r, self_i) = promote_tensors(self_r, self_i)
    self = ComplexTensor(self_r, self_i)
    self_abs_sqrt = torch.sqrt(torch.abs(self))
    self_half_angle = 0.5 * torch.angle(self)

    ret_r = self_abs_sqrt * torch.cos(self_half_angle)
    ret_i = self_abs_sqrt * torch.sin(self_half_angle)

    return ComplexTensor(ret_r.to(out_dt), ret_i.to(out_dt))


@register_complex(aten.rsqrt)
def rsqrt_impl(self: ComplexTensor) -> ComplexTensor:
    self_r, self_i = split_complex_tensor(self)
    out_dt, (self_r, self_i) = promote_tensors(self_r, self_i)
    self = ComplexTensor(self_r, self_i)
    self_abs_rsqrt = torch.rsqrt(torch.abs(self))
    self_neg_half_angle = -0.5 * torch.angle(self)

    ret_r = self_abs_rsqrt * torch.cos(self_neg_half_angle)
    ret_i = self_abs_rsqrt * torch.sin(self_neg_half_angle)

    return ComplexTensor(ret_r.to(out_dt), ret_i.to(out_dt))


@register_complex(aten.addmm)
def addmm_impl(
    input: ComplexTensor,
    mat1: ComplexTensor,
    mat2: ComplexTensor,
    out_dtype: torch.dtype | None = None,
    beta: complex = 1,
    alpha: complex = 1,
) -> ComplexTensor:
    ret = beta * input + alpha * torch.mm(mat1, mat2)
    assert isinstance(ret, ComplexTensor)
    ret_r, ret_i = split_complex_tensor(ret)
    if out_dtype is not None:
        out_dtype = COMPLEX_TO_REAL[out_dtype]
        ret_r, ret_i = ret_r.to(out_dtype), ret_i.to(out_dtype)
    return ComplexTensor(ret_r, ret_i)


def elemwise_nonzero(self: ComplexTensor) -> torch.Tensor:
    re, im = split_complex_tensor(self)
    return (re != 0) | (im != 0)


def register_nonzero_impl(op: OpType) -> Callable[..., Any]:
    def nonzero_impl(
        self: ComplexTensor, other: ComplexTensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        return op(elemwise_nonzero(self), elemwise_nonzero(other), *args, **kwargs)

    func_name = _get_func_name(op)
    nonzero_impl.__name__ = func_name
    nonzero_impl.__qualname__ = func_name

    return register_complex(op, nonzero_impl)


logical_and_impl = register_nonzero_impl(aten.logical_and)
logical_or_impl = register_nonzero_impl(aten.logical_or)
logical_xor_impl = register_nonzero_impl(aten.logical_xor)


@register_complex(aten.logical_not)
def logical_not_impl(self: ComplexTensor, *args: Any, **kwargs: Any) -> torch.Tensor:
    return torch.logical_not(elemwise_nonzero(self), *args, **kwargs)


@register_complex(aten.view_as_real)
def view_as_real_impl(self: ComplexTensor) -> torch.Tensor:
    re, im = split_complex_tensor(self)
    return torch.stack([re, im], dim=-1)


@register_complex(aten.linalg_vector_norm)
def linalg_vector_norm_impl(
    self: ComplexTensor, *args: Any, **kwargs: Any
) -> torch.Tensor:
    return torch.linalg.vector_norm(torch.abs(self), *args, **kwargs)


@register_force_test(aten.copy_)
def copy__impl(
    self: ComplexTensor | torch.Tensor,
    src: ComplexTensor | torch.Tensor,
    *args: Any,
    **kwargs: Any,
) -> ComplexTensor | torch.Tensor:
    if not self.dtype.is_complex:
        warnings.warn(
            "Casting complex values to real discards the imaginary part", UserWarning
        )
        src_re, src_im = split_complex_arg(src)
        return self.copy_(src_re)

    self_re, self_im = split_complex_arg(self)
    src_re, src_im = split_complex_arg(src)

    ret_re = self_re.copy_(src_re, *args, **kwargs)
    ret_im = self_im.copy_(src_im, *args, **kwargs)

    return ComplexTensor(ret_re, ret_im)


@register_complex(aten._local_scalar_dense)
def _local_scalar_dense_impl(self: ComplexTensor, *args: Any, **kwargs: Any) -> complex:
    x, y = split_complex_tensor(self)
    u = aten._local_scalar_dense(x, *args, **kwargs)
    v = aten._local_scalar_dense(y, *args, **kwargs)
    return complex(u, v)


@register_complex(aten.allclose)
def allclose_impl(
    input: torch.Tensor,
    other: torch.Tensor,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
) -> bool:
    # pyrefly: ignore [bad-return]
    return torch.all(
        torch.isclose(input, other, rtol=rtol, atol=atol, equal_nan=equal_nan)
    ).item()  # type: ignore[bad-return]


@register_complex(aten.stack)
def stack_impl(self: list[ComplexTensor], *args: Any, **kwargs: Any) -> ComplexTensor:
    re_im_tuples = [split_complex_arg(self_i) for self_i in self]
    u = torch.stack([c[0] for c in re_im_tuples], *args, **kwargs)
    v = torch.stack([c[1] for c in re_im_tuples], *args, **kwargs)
    return ComplexTensor(u, v)


# TODO (hameerabbasi): Not being tested
@register_complex(aten._conj_physical)
@register_complex(aten.conj_physical)
def conj_physical_impl(self: ComplexTensor) -> ComplexTensor:
    re, im = split_complex_tensor(self)
    return ComplexTensor(re, -im)


# TODO (hameerabbasi): Not being tested
@register_complex(aten._conj)
def _conj_impl(self: ComplexTensor) -> ComplexTensor:
    re, im = split_complex_tensor(self)
    return ComplexTensor(re, torch._neg_view(im))


@register_complex(aten.index_add)
def index_add_impl(
    self: ComplexTensor,
    dim: int,
    index: torch.Tensor,
    source: ComplexTensor,
    **kwargs: Any,
) -> ComplexTensor:
    alpha = kwargs.pop("alpha", None)
    if alpha is not None:
        source = source * alpha
    self_re, self_im = split_complex_arg(self)
    source_re, source_im = split_complex_arg(source)

    ret_re = self_re.index_add(dim, index, source_re)
    ret_im = self_im.index_add(dim, index, source_im)

    return ComplexTensor(ret_re, ret_im)


# TODO (hameerabbasi): Not being tested
@register_complex(aten.index_add_)
def index_add__impl(
    self: ComplexTensor,
    dim: int,
    index: torch.Tensor,
    source: ComplexTensor,
    **kwargs: Any,
) -> ComplexTensor:
    alpha = kwargs.pop("alpha", None)
    if alpha is not None:
        source = source * alpha

    self_re, self_im = split_complex_arg(self)
    source_re, source_im = split_complex_arg(source)

    ret_re = self_re.index_add_(dim, index, source_re)
    ret_im = self_im.index_add_(dim, index, source_im)

    return ComplexTensor(ret_re, ret_im)


@register_complex(aten.masked_fill)
def masked_fill_impl(
    self: ComplexTensor, mask: torch.Tensor, value: complex
) -> ComplexTensor:
    self_re, self_im = split_complex_arg(self)
    value_re, value_im = split_complex_arg(value)

    ret_re = self_re.masked_fill(mask, value_re)
    ret_im = self_im.masked_fill(mask, value_im)

    return ComplexTensor(ret_re, ret_im)


# TODO (hameerabbasi): Not being tested
@register_complex(aten.masked_fill_)
def masked_fill__impl(
    self: ComplexTensor, mask: torch.Tensor, value: complex
) -> ComplexTensor:
    self_re, self_im = split_complex_arg(self)
    value_re, value_im = split_complex_arg(value)

    ret_re = self_re.masked_fill_(mask, value_re)
    ret_im = self_im.masked_fill_(mask, value_im)

    return ComplexTensor(ret_re, ret_im)


@register_complex(aten.constant_pad_nd)
def constant_pad_nd_impl(
    self: ComplexTensor, pad: Sequence[int], value: complex | None = None
) -> ComplexTensor:
    self_re, self_im = split_complex_tensor(self)
    if value is None:
        ret_re = aten.constant_pad_nd(self_re, pad)
        ret_im = aten.constant_pad_nd(self_im, pad)
    else:
        value_re, value_im = split_complex_arg(value)
        ret_re = aten.constant_pad_nd(self_re, pad, value_re)
        ret_im = aten.constant_pad_nd(self_im, pad, value_im)

    return ComplexTensor(ret_re, ret_im)


@register_complex(aten.var)
def var_impl(self: ComplexTensor, *args: Any, **kwargs: Any) -> torch.Tensor:
    self_re, self_im = split_complex_tensor(self)
    return torch.var(self_re, *args, **kwargs) + torch.var(self_im, *args, **kwargs)


@register_complex(aten.scatter_add)
def scatter_add_impl(
    self: ComplexTensor, dim: int, index: torch.Tensor, src: ComplexTensor
) -> ComplexTensor:
    self_re, self_im = split_complex_arg(self)
    src_re, src_im = split_complex_arg(src)

    ret_re = torch.scatter_add(self_re, dim, index, src_re)
    ret_im = torch.scatter_add(self_im, dim, index, src_im)

    return ComplexTensor(ret_re, ret_im)


@register_complex(aten.scatter_add_)
def scatter_add__impl(
    self: ComplexTensor, dim: int, index: torch.Tensor, src: ComplexTensor
) -> ComplexTensor:
    self_re, self_im = split_complex_arg(self)
    src_re, src_im = split_complex_arg(src)

    out_re = self_re.scatter_add_(dim, index, src_re)
    out_im = self_im.scatter_add_(dim, index, src_im)

    return ComplexTensor(out_re, out_im)


@register_complex(aten.index_put_)
def index_put__impl(
    self: ComplexTensor,
    indices: tuple[torch.Tensor, ...],
    values: ComplexTensor,
    accumulate: bool = False,
) -> ComplexTensor:
    self_re, self_im = split_complex_arg(self)
    values_re, values_im = split_complex_arg(values)

    out_re = self_re.index_put_(indices, values_re, accumulate=accumulate)
    out_im = self_im.index_put_(indices, values_im, accumulate=accumulate)

    return ComplexTensor(out_re, out_im)


@register_complex(aten.tanh_backward)
def tanh_backward(out_grad: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return out_grad * (1.0 - y * y).conj_physical()


@register_complex(aten.diagonal_backward)
def diagonal_backward(
    grad_output: torch.Tensor, input_sizes: list[int], offset: int, dim1: int, dim2: int
) -> torch.Tensor:
    grad_input = grad_output.new_zeros(input_sizes)
    return torch.diagonal_scatter(grad_input, grad_output, offset, dim1, dim2)


def _dt_to_real(dt: torch.dtype | Any) -> torch.dtype | Any:
    if not isinstance(dt, torch.dtype):
        return dt

    return COMPLEX_TO_REAL[dt]


def register_to_impl(op: OpType) -> Callable[..., Any]:
    """Register an op similar to `aten.to`, but may have different signatures."""

    def impl(
        self: ComplexTensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor | ComplexTensor:
        x, y = split_complex_tensor(self)
        try:
            args = tuple(_dt_to_real(a) for a in args)
            kwargs = {k: _dt_to_real(v) for k, v in kwargs.items()}
        except KeyError:
            return op(x, *args, **kwargs)

        return ComplexTensor(op(x, *args, **kwargs), op(y, *args, **kwargs))

    func_name = _get_func_name(op)
    impl.__name__ = func_name
    impl.__qualname__ = func_name

    return register_complex(op, impl)


to_impl = register_to_impl(aten.to)
_to_copy_impl = register_to_impl(aten._to_copy)
