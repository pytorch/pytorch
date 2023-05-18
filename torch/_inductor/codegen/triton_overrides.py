import math

import torch

import torch._inductor.config as config
from torch._prims_common import is_integer_dtype

from ..virtualized import ops, V
from .common import OpOverrides


def triton_constant(value):
    if value == float("inf"):
        return 'float("inf")'
    elif value == float("-inf"):
        return 'float("-inf")'
    elif math.isnan(value):
        return 'float("nan")'
    return repr(value)


def triton_acc_type(dtype):
    if is_integer_dtype(dtype) and dtype.is_signed:
        nbits = 64 if dtype == torch.int64 else 32
        return f"tl.int{nbits}"
    return triton_compute_type(dtype)


def triton_compute_type(dtype):
    triton_type_name = str(dtype).split(".")[-1]
    if triton_type_name == "bool":
        triton_type_name = "int1"
    if triton_type_name in ("float16", "bfloat16"):
        # float16 math is done in float32 inside the kernel
        triton_type_name = "float32"
    return f"tl.{triton_type_name}"


class TritonOverrides(OpOverrides):
    """Map element-wise ops to Triton"""

    @staticmethod
    def to_dtype(x, dtype: torch.dtype):
        if dtype == torch.bool:
            return f"({x} != 0)"
        elif dtype == torch.uint8:
            # to work around llvm uint conversion semantics
            # that produces 0's for negative values
            return f"{x}.to(tl.int8).to(tl.uint8)"
        return f"{x}.to({triton_compute_type(dtype)})"

    @staticmethod
    def constant(value, dtype):
        type_ = torch._prims_common.dtype_to_type(dtype)
        return triton_constant(type_(value))

    @staticmethod
    def abs(x):
        return f"tl.abs({x})"

    @staticmethod
    def libdevice_abs(x):
        return f"tl.math.abs({x})"

    @staticmethod
    def exp(x):
        return f"tl.exp({x})"

    @staticmethod
    def libdevice_exp(x):
        return f"tl.math.exp({x})"

    @staticmethod
    def exp2(x):
        return f"tl.math.exp2({x})"

    @staticmethod
    def expm1(x):
        return f"tl.math.expm1({x})"

    @staticmethod
    def sqrt(x):
        return f"tl.sqrt({x})"

    @staticmethod
    def libdevice_sqrt(x):
        return f"tl.math.sqrt({x})"

    @staticmethod
    def relu(x):
        bug = config.triton.inject_relu_bug_TESTING_ONLY
        if bug == "compile_error":
            return "compile error!"
        elif bug == "runtime_error":
            # NB: this only triggers runtime error as long as input
            # is not all zero
            return f'triton_helpers.device_assert_then({x} == 0, "injected assert fail", {x})'
        elif bug == "accuracy":
            return f"{x} + 1"
        elif bug is None:
            return ops.maximum("0", x)
        else:
            raise AssertionError(
                f"unrecognized config triton.inject_relu_bug_TESTING_ONLY = {bug!r}"
            )

    @staticmethod
    def minimum(a, b):
        return f"triton_helpers.minimum({a}, {b})"

    @staticmethod
    def maximum(a, b):
        return f"triton_helpers.maximum({a}, {b})"

    @staticmethod
    def where(a, b, c):
        return f"tl.where({a}, {b}, {c})"

    @staticmethod
    def cos(x):
        return f"tl.cos({x})"

    @staticmethod
    def libdevice_cos(x):
        return f"tl.math.cos({x})"

    @staticmethod
    def sin(x):
        return f"tl.sin({x})"

    @staticmethod
    def libdevice_sin(x):
        return f"tl.math.sin({x})"

    @staticmethod
    def index_expr(expr, dtype):
        index_str, mask_vars, mask, expand_str = V.kernel.indexing(expr)
        var = V.kernel.cse.generate(V.kernel.compute, index_str)
        var.mask_vars = mask_vars
        return var

    @staticmethod
    def masked(mask, body, other):
        with V.kernel.mask_loads(mask) as new_mask:
            result = body()
        return ops.where(new_mask, result, triton_constant(other))

    @staticmethod
    def lgamma(x):
        return f"tl.math.lgamma({x})"

    @staticmethod
    def erf(x):
        return f"tl.math.erf({x})"

    @staticmethod
    def cosh(x):
        return f"tl.math.cosh({x})"

    @staticmethod
    def sinh(x):
        return f"tl.math.sinh({x})"

    @staticmethod
    def acos(x):
        return f"tl.math.acos({x})"

    @staticmethod
    def acosh(x):
        return f"tl.math.acosh({x})"

    @staticmethod
    def asin(x):
        return f"tl.math.asin({x})"

    @staticmethod
    def asinh(x):
        return f"tl.math.asinh({x})"

    @staticmethod
    def atan2(x, y):
        return f"tl.math.atan2({x}, {y})"

    @staticmethod
    def atan(x):
        return f"tl.math.atan({x})"

    @staticmethod
    def atanh(x):
        return f"tl.math.atanh({x})"

    @staticmethod
    def copysign(x, y):
        return f"tl.math.copysign({x}, {y})"

    @staticmethod
    def erfc(x):
        return f"tl.math.erfc({x})"

    @staticmethod
    def hypot(x, y):
        return f"tl.math.hypot({x}, {y})"

    @staticmethod
    def log10(x):
        return f"tl.math.log10({x})"

    @staticmethod
    def nextafter(x, y):
        return f"tl.math.nextafter({x}, {y})"

    @staticmethod
    def logical_and(a, b):
        return f"{a} & {b}"

    @staticmethod
    def logical_or(a, b):
        return f"{a} | {b}"

    @staticmethod
    def rand(seed, offset, _):  # _ here to keep the contract identical to CPU rand op
        offset = f"({offset}).to(tl.uint32)"
        return f"tl.rand({seed}, {offset})"

    @staticmethod
    def randn(seed, offset, _):  # _ here to keep the contract identical to CPU randn op
        offset = f"({offset}).to(tl.uint32)"
        return f"tl.randn({seed}, {offset})"

    # TODO: work out how to use randint4x
    @staticmethod
    def randint(
        seed, offset, _
    ):  # _ here to keep the contract identical to CPU randint op
        offset = f"({offset}).to(tl.uint32)"
        return f"tl.randint({seed}, {offset}).to(tl.int32)"

    @staticmethod
    def rsqrt(x):
        return f"tl.math.rsqrt({x})"

    @staticmethod
    def log1p(x):
        return f"tl.math.log1p({x})"

    @staticmethod
    def tan(x):
        return f"tl.math.tan({x})"

    @staticmethod
    def tanh(x):
        return f"tl.math.tanh({x})"

    @staticmethod
    def sigmoid(x):
        return f"tl.sigmoid({x})"

    @staticmethod
    def libdevice_sigmoid(x):
        return f"1/(1 + tl.math.exp(-({x})))"

    @staticmethod
    def signbit(x):
        # XX: This is wrong for the value -0.0 in floating point
        return f"tl.math.signbit({x}) if ({x}).dtype is tl.float32 else {x} < 0"

    @staticmethod
    def fmod(a, b):
        return f"tl.math.fmod({a}, {b})"

    @staticmethod
    def pow(a, b):
        return f"tl.math.pow({a}, {b})"

    @staticmethod
    def log(x):
        return f"tl.log({x})"

    @staticmethod
    def libdevice_log(x):
        return f"tl.math.log({x})"

    @staticmethod
    def isinf(x):
        return f"tl.math.isinf({x}).to(tl.int1)"

    @staticmethod
    def isnan(x):
        return f"tl.math.isnan({x}).to(tl.int1)"

    @staticmethod
    def round(x):
        return f"tl.math.nearbyint({x})"

    @staticmethod
    def floor(x):
        return f"tl.math.floor({x})"

    @staticmethod
    def floordiv(a, b):
        # See the comment in lowering.div_mode. a and b are integer type.
        # Similar to div_floor_kernel_cuda in pytorch core.
        # Notice that // in triton behaves as truncdiv instead of floordiv
        quot = f"{a} // {b}"
        rem = f"{a} % {b}"
        return f"tl.where(({a} < 0) != ({b} < 0), tl.where({rem} != 0, {quot} - 1, {quot}), {quot})"

    @staticmethod
    def sign(x):
        left = ops.where(ops.lt("0", x), 1, 0)
        right = ops.where(ops.lt(x, "0"), 1, 0)
        sub = ops.sub(left, right)
        return f"{sub}.to({x}.dtype)"

    @staticmethod
    def trunc(x):
        return f"tl.math.trunc({x})"

    @staticmethod
    def truncdiv(a, b):
        # See the comment in lowering.div_mode. a and b are integer type.
        # Notice that // in triton behaves as truncdiv instead of floordiv
        return f"{a} // {b}"

    @staticmethod
    def ceil(x):
        return f"tl.math.ceil({x})"
