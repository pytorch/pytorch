import contextlib
import dataclasses
import functools
import itertools
import logging
import math
import re
import sys
from copy import copy, deepcopy
from pathlib import Path
from typing import Dict, List

import numpy
import sympy

import torch
import torch.fx
from torch._inductor import dependencies
from torch._inductor.ir import StorageBox, TensorBox
from torch._prims_common import is_float_dtype

from .. import codecache, config, ir, metrics
from ..codegen.wrapper import WrapperCodeGen
from ..scheduler import SchedulerNode
from ..utils import (
    cache_on_self,
    get_fused_kernel_name,
    sympy_product,
    sympy_subs,
    sympy_symbol,
)
from ..virtualized import ops, V
from .common import (
    BracesBuffer,
    CppWrapperKernelArgs,
    CSE,
    data_type_propagation,
    DeferredLine,
    ExprPrinter,
    IndentedBuffer,
    Kernel,
    KernelArgs,
    OpOverrides,
    OptimizationContext,
)

schedule_log = torch._logging.getArtifactLogger(__name__, "schedule")

DTYPE_TO_CPP = {
    torch.float32: "float",
    torch.float64: "double",
    torch.float16: "half",
    torch.int64: "long",
    torch.int32: "int",
    torch.int16: "short",
    torch.int8: "signed char",
    torch.uint8: "unsigned char",
    torch.bool: "bool",
    torch.bfloat16: "bfloat16",
}

DTYPE_TO_ATEN = {
    torch.float32: "at::kFloat",
    torch.float64: "at::kDouble",
    torch.float16: "at::kHalf",
    torch.int64: "at::kLong",
    torch.int32: "at::kInt",
    torch.int16: "at::kShort",
    torch.int8: "at::kChar",
    torch.uint8: "at::kByte",
    torch.bool: "at::kBool",
    torch.bfloat16: "at::kBFloat16",
}

DEVICE_TO_ATEN = {
    "cpu": "at::kCPU",
    "cuda": "at::kCUDA",
}

INDEX_TYPE = "long"

RTYPE_TO_CPP = {
    "sum": "+",
    "prod": "*",
    "xor_sum": "^",
    "min": "min",
    "max": "max",
    "argmin": "argmin",
    "argmax": "argmax",
    "any": "||",
}

PYTHON_TO_CPP = {
    "int": "long",
    "float": "double",
    "bool": "bool",
}


def reduction_init(reduction_type, dtype):
    if dtype in (torch.float16, torch.bfloat16):
        # Since load promotes all half-precision inputs to float, the initial
        # constant for reduction must be promoted as well
        dtype = torch.float32
    if reduction_type in ("xor_sum", "sum", "any"):
        return 0
    if reduction_type == "prod":
        return 1
    if reduction_type in {"max", "argmax"}:
        return (
            f"-std::numeric_limits<{DTYPE_TO_CPP[dtype]}>::infinity()"
            if is_float_dtype(dtype)
            else f"std::numeric_limits<{DTYPE_TO_CPP[dtype]}>::min()"
        )
    if reduction_type in {"min", "argmin"}:
        return (
            f"std::numeric_limits<{DTYPE_TO_CPP[dtype]}>::infinity()"
            if is_float_dtype(dtype)
            else f"std::numeric_limits<{DTYPE_TO_CPP[dtype]}>::max()"
        )
    raise AssertionError(reduction_type)


def reduction_combine(reduction_type, var, next_value):
    if reduction_type == "sum":
        return f"{var} += {next_value}"
    if reduction_type == "prod":
        return f"{var} *= {next_value}"
    if reduction_type == "xor_sum":
        return f"{var} ^= {next_value}"
    if reduction_type == "any":
        return f"{var} = {var} || {next_value}"
    if reduction_type in ("min", "max"):
        return f"{var} = {reduction_type}_propagate_nan({var}, {next_value})"
    raise AssertionError(reduction_type)


def reduction_combine_vec(reduction_type, var, next_value):
    if reduction_type == "max":
        return f"{var} = at::vec::maximum({var}, {next_value})"
    elif reduction_type == "min":
        return f"{var} = at::vec::minimum({var}, {next_value})"
    elif reduction_type == "sum":
        return f"{var} += {next_value}"
    elif reduction_type == "prod":
        return f"{var} *= {next_value}"
    elif reduction_type == "xor_sum":
        return f"{var} ^= {next_value}"
    else:
        raise NotImplementedError()


index_value_name_counter = 1


def argmax_argmin_prefix(reduction_type, src_dtype, tmpvar):
    global index_value_name_counter
    struct_name = f"IndexValue_{index_value_name_counter}"
    index_value_name_counter += 1

    # A small annoyance, due to it being a little cumbersome to just throw {} into strings
    prefix = [
        f"struct {struct_name} {{size_t index; {DTYPE_TO_CPP[src_dtype]} value;}};",
        f"{struct_name} {tmpvar}{{0, {reduction_init(reduction_type, src_dtype)}}};",
    ]
    if reduction_type == "argmax":
        prefix.extend(
            [
                f"#pragma omp declare reduction(argmax : struct {struct_name} :\\",
                "    omp_out.value = omp_in.value < omp_out.value ? omp_out.value : omp_in.value,\\",
                "    omp_out.index = omp_in.value < omp_out.value ? omp_out.index : omp_in.index)\\",
                f"\tinitializer(omp_priv = {{0, {reduction_init(reduction_type, src_dtype)}}})",
            ]
        )
    elif reduction_type == "argmin":
        prefix.extend(
            [
                f"#pragma omp declare reduction(argmin : struct {struct_name} :\\",
                "    omp_out.value = omp_in.value > omp_out.value ? omp_out.value : omp_in.value,\\",
                "    omp_out.index = omp_in.value > omp_out.value ? omp_out.index : omp_in.index)\\",
                f"\tinitializer(omp_priv = {{0, {reduction_init(reduction_type, src_dtype)}}})",
            ]
        )
    return prefix


def parallel_num_threads():
    threads = config.cpp.threads
    if threads < 1:
        threads = torch.get_num_threads()
    return threads


@functools.lru_cache()
def stride_at(var: sympy.Symbol, index: sympy.Expr):
    replacement = {var: var + 1}
    new_index = sympy_subs(index, replacement)
    return sympy.simplify(new_index - index)


@functools.lru_cache()
def cpp_prefix():
    path = Path(__file__).parent / "cpp_prefix.h"
    with path.open() as f:
        _, filename = codecache.write(
            f.read(),
            "h",
        )
    return f'#include "{filename}"'


class CppPrinter(ExprPrinter):
    def _print_Integer(self, expr):
        return f"{int(expr)}L"

    def _print_ModularIndexing(self, expr):
        x, div, mod = expr.args
        x = self.paren(self.doprint(x))
        div = self.paren(self.doprint(div))
        mod = self.paren(self.doprint(mod))
        if div != "1":
            x = f"({x} / {div})"
        return f"static_cast<{INDEX_TYPE}>({x}) % static_cast<{INDEX_TYPE}>({mod})"

    def _print_FloorDiv(self, expr):
        x, div = expr.args
        x = self.paren(self.doprint(x))
        div = self.paren(self.doprint(div))
        return f"({x} / {div})"

    def _print_floor(self, expr):
        assert len(expr.args) == 1
        return f"std::floor({self._print(expr.args[0])})"

    def _print_Pow(self, expr):
        # Uses float constants to perform FP div
        base, exp = expr.args
        base = self._print(base)
        if exp == 0.5:
            return f"std::sqrt({base})"
        assert exp.is_integer
        exp = int(exp)
        if exp > 0:
            return "*".join([self.paren(base)] * exp)
        elif exp < 0:
            return "1.0/" + self.paren("*".join([self.paren(base)] * abs(exp)))
        else:  # exp == 0
            return "1"

    def _print_Rational(self, expr):
        # Uses float constants to perform FP div
        if expr.q == 1:
            return f"{expr.p}"
        return f"{expr.p}.0/{expr.q}.0"

    def _print_ceiling(self, expr):
        assert len(expr.args) == 1
        return f"std::ceil({self._print(expr.args[0])})"


cexpr = CppPrinter().doprint


def cexpr_index(index):
    return f"static_cast<{INDEX_TYPE}>({cexpr(index)})"


class RecordOptimizationContext:
    def __init__(self, func_name: str = ""):
        self.func_name = func_name
        self.current_node: torch.fx.Node = None
        self.opt_ctx: OptimizationContext = None

    def __enter__(self):
        assert V.interpreter
        assert V.interpreter.current_node

        self.current_node: torch.fx.Node = V.interpreter.current_node
        if OptimizationContext.key in self.current_node.meta:
            self.opt_ctx = self.current_node.meta[OptimizationContext.key]
        else:
            self.opt_ctx = OptimizationContext()
        self.opt_ctx.ops_name = self.func_name
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert self.current_node
        assert self.opt_ctx
        self.current_node.meta[OptimizationContext.key] = self.opt_ctx

    def get_opt_ctx(self):
        return self.opt_ctx

    def get_fx_node(self):
        assert self.current_node
        return self.current_node


def get_opt_ctx(node: torch.fx.Node) -> OptimizationContext:
    return node.meta.get(OptimizationContext.key, None)


def get_current_node_opt_ctx() -> OptimizationContext:
    assert V.interpreter.current_node
    return get_opt_ctx(V.interpreter.current_node)


class CppVecOverrides(OpOverrides):
    """Map element-wise ops to aten vectorization C++"""

    @staticmethod
    def add(a, b):
        return f"{a} + {b}"

    @staticmethod
    def sub(a, b):
        return f"{a} - {b}"

    @staticmethod
    def mul(a, b):
        return f"{a} * {b}"

    @staticmethod
    def div(a, b):
        return f"{a} / {b}"

    @staticmethod
    def abs(x):
        return f"{x}.abs()"

    @staticmethod
    def sin(x):
        return f"{x}.sin()"

    @staticmethod
    def cos(x):
        return f"{x}.cos()"

    @staticmethod
    def exp(x):
        return f"{x}.exp()"

    @staticmethod
    def exp2(x):
        return f"{x}.exp2()"

    @staticmethod
    def expm1(x):
        # decompose for a better performance
        vec_one = f"decltype({x})(1)"
        return f"{x}.exp() - {vec_one}"

    @staticmethod
    def erf(x):
        return f"{x}.erf()"

    @staticmethod
    def erfc(x):
        return f"{x}.erfc()"

    @staticmethod
    def sqrt(x):
        return f"{x}.sqrt()"

    @staticmethod
    def eq(x, y):
        return f"to_float_mask({x} == {y})"

    @staticmethod
    def ne(x, y):
        return f"to_float_mask({x} != {y})"

    @staticmethod
    def lt(x, y):
        return f"to_float_mask({x} < {y})"

    @staticmethod
    def gt(x, y):
        return f"to_float_mask({x} > {y})"

    @staticmethod
    def le(x, y):
        return f"to_float_mask({x} <= {y})"

    @staticmethod
    def ge(x, y):
        return f"to_float_mask({x} >= {y})"

    @staticmethod
    def and_(x, y):
        return f"{x} & {y}"

    @staticmethod
    def rsqrt(x):
        return f"{x}.rsqrt()"

    @staticmethod
    def pow(a, b):
        return f"{a}.pow({b})"

    @staticmethod
    def log(x):
        return f"{x}.log()"

    @staticmethod
    def round(x):
        return f"{x}.round()"

    @staticmethod
    def floor(x):
        return f"{x}.floor()"

    @staticmethod
    def ceil(x):
        return f"{x}.ceil()"

    @staticmethod
    def trunc(x):
        return f"{x}.trunc()"

    @staticmethod
    def fmod(a, b):
        return f"{a}.fmod({b})"

    @staticmethod
    def lgamma(x):
        return f"{x}.lgamma()"

    @staticmethod
    def logical_and(a, b):
        return f"({a} != 0) & ({b} != 0)"

    @staticmethod
    def logical_or(a, b):
        return f"({a} != 0) | ({b} != 0)"

    @staticmethod
    def tan(a):
        return f"{a}.tan()"

    @staticmethod
    def tanh(a):
        vec_one = f"decltype({a})(1)"
        vec_two = f"decltype({a})(2)"
        vec_minus_two = f"decltype({a})(-2)"
        return f"{vec_two} / ({vec_one} + ({vec_minus_two} * {a}).exp()) - {vec_one}"

    @staticmethod
    def reciprocal(a):
        return f"{a}.reciprocal()"

    @staticmethod
    def atan(x):
        return f"{x}.atan()"

    @staticmethod
    def acos(x):
        return f"{x}.acos()"

    @staticmethod
    def asin(x):
        return f"{x}.asin()"

    @staticmethod
    def cosh(x):
        return f"{x}.cosh()"

    @staticmethod
    def sinh(x):
        return f"{x}.sinh()"

    @staticmethod
    def log10(x):
        return f"{x}.log10()"

    @staticmethod
    def nextafter(x):
        return f"{x}.nextafter()"

    @staticmethod
    def copysign(a, b):
        return f"{a}.copysign({b})"

    @staticmethod
    def atan2(a, b):
        return f"{a}.atan2({b})"

    @staticmethod
    def hypot(a, b):
        return f"{a}.hypot({b})"

    @staticmethod
    def atanh(x):
        # For real x, atanh(x) = 1/2 * log((1+x)/(1-x))
        vec_one = f"decltype({x})(1)"
        vec_one_half = f"decltype({x})(0.5)"
        return f"{vec_one_half} * (({vec_one} + {x})/({vec_one} - {x})).log()"

    @staticmethod
    def asinh(x):
        # For real x, asinh(x) = log(x + sqrt(1 + x**2))
        vec_one = f"decltype({x})(1)"
        return f"({x} + ({vec_one} + {x}*{x}).sqrt()).log()"

    @staticmethod
    def acosh(x):
        # For real x, acosh(x) = log(x + sqrt(x**2 -1))
        vec_one = f"decltype({x})(1)"
        return f"({x} + ({x}*{x} - {vec_one}).sqrt()).log()"

    @staticmethod
    def constant(val, dtype):
        opt_ctx: OptimizationContext = get_current_node_opt_ctx()
        assert opt_ctx
        assert opt_ctx.dtype in [torch.int32, torch.float32, torch.bfloat16]
        if dtype in [torch.bfloat16]:
            assert opt_ctx.is_load_bf16_as_fp32 or opt_ctx.is_bf16_mem_copy
        proposed_dtype = opt_ctx.dtype
        if val == float("inf"):
            assert proposed_dtype == torch.float
            quote = f"std::numeric_limits<{DTYPE_TO_CPP[proposed_dtype]}>::infinity()"
        elif val == float("-inf"):
            assert proposed_dtype == torch.float
            quote = f"-std::numeric_limits<{DTYPE_TO_CPP[proposed_dtype]}>::infinity()"
        elif math.isnan(val):
            quote = f"std::numeric_limits<{DTYPE_TO_CPP[proposed_dtype]}>::quiet_NaN()"
        elif val is True or val is False:
            quote = f"static_cast<{DTYPE_TO_CPP[proposed_dtype]}>({str(val).lower()})"
        else:
            quote = f"static_cast<{DTYPE_TO_CPP[proposed_dtype]}>({repr(val)})"

        return f"at::vec::Vectorized<{DTYPE_TO_CPP[proposed_dtype]}>({quote})"

    @staticmethod
    def relu(x):
        bug = config.cpp.inject_relu_bug_TESTING_ONLY
        if bug == "compile_error":
            return "compile error!"
        elif bug == "runtime_error":
            return f"{x}; throw 1"
        elif bug == "accuracy":
            return f"{x} + decltype({x})(1)"
        elif bug is None:
            return f"at::vec::clamp_min({x}, decltype({x})(0))"
        else:
            raise AssertionError(
                f"unrecognized config cpp.inject_relu_bug_TESTING_ONLY = {bug!r}"
            )

    # TODO: this seems to be dead
    @staticmethod
    def sigmoid(x):
        return f"decltype({x})(1)/(decltype({x})(1) + {x}.neg().exp())"

    @staticmethod
    def neg(x):
        return f"{x}.neg()"

    @staticmethod
    def floordiv(a, b):
        # a and b are integer type
        _t = f"decltype({a})"
        quot = f"{a} / {b}"
        rem = f"{a} % {b}"
        return f"(({a} < {_t}(0)) != ({b} < {_t}(0)) ? ({rem} != {_t}(0) ? {quot} - {_t}(1) : {quot}) : {quot})"

    @staticmethod
    def truncdiv(a, b):
        # a and b are integer type
        return f"{a} / {b}"

    @staticmethod
    def minimum(a, b):
        return f"at::vec::minimum({a}, {b})"

    @staticmethod
    def maximum(a, b):
        return f"at::vec::maximum({a}, {b})"

    @staticmethod
    def square(a):
        return f"{a} * {a}"

    @staticmethod
    def where(a, b, c):
        return f"decltype({b})::blendv({c}, {b}, {a})"

    @staticmethod
    def sign(x):
        code = BracesBuffer()
        # auto tmp5 = tmp4 < 0 ? -1 : 1;
        vec_zero = f"decltype({x})(0)"
        vec_one = f"decltype({x})(1)"
        blendv = f"decltype({x})::blendv({vec_zero}, {vec_one}, {vec_zero} < {x})"
        left = V.kernel.cse.newvar()
        code.writeline(f"auto {left} = {blendv};")

        # auto tmp6 = tmp4 == 0 ? 0 : tmp5;
        blendv = f"decltype({x})::blendv({vec_zero}, {vec_one}, {x} < {vec_zero})"
        right = V.kernel.cse.newvar()
        code.writeline(f"auto {right} = {blendv};")
        result = V.kernel.cse.newvar()
        code.writeline(f"auto {result} = {left} - {right};")
        V.kernel.compute.splice(code)
        return result

    @staticmethod
    def to_dtype(x, dtype):
        assert dtype in [
            torch.bool,
            torch.float,
            torch.bfloat16,
            torch.uint8,
        ], f"{__name__} does not support {dtype}"
        node: torch.fx.Node = V.interpreter.current_node
        assert node
        opt_ctx_x = get_opt_ctx(node.args[1])
        assert opt_ctx_x
        if opt_ctx_x.dtype in (torch.float, torch.float32) and dtype == torch.bool:
            return f"vec_convert_to_mask({x})"
        if opt_ctx_x.dtype == torch.bool and dtype in (torch.float, torch.float32):
            return f"mask_convert_to_float({x})"
        # TODO(jgong5): support conversion for other types
        # currently we only allow load/store torch.uint8 and handle conversion there
        return f"({x})"

    @staticmethod
    def log1p(x):
        bug = config.cpp.inject_log1p_bug_TESTING_ONLY
        if bug == "accuracy":
            return f"{x} + decltype({x})(1)"
        elif bug is None:
            return f"{x}.log1p()"
        else:
            raise AssertionError(
                f"unrecognized config cpp.inject_log1p_bug_TESTING_ONLY = {bug!r}"
            )

    @staticmethod
    def masked(mask, body, other):
        code = BracesBuffer()
        var = V.kernel.cse.newvar()
        code.writeline(f"auto {var} = [&]")
        with V.kernel.swap_buffers(code), code.indent():
            result = body()
            code.writeline(f"return {result};")
        code.writeline(";")
        V.kernel.compute.splice(code)

        if other == float("-inf"):
            other_code = (
                "at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())"
            )
        elif other == float("inf"):
            other_code = (
                "at::vec::Vectorized<float>(std::numeric_limits<float>::infinity())"
            )
        elif math.isnan(other):
            other_code = (
                "at::vec::Vectorized<float>(std::numeric_limits<float>::quiet_NaN())"
            )
        else:
            other_code = f"at::vec::Vectorized<float>({other!r})"
        type = f"decltype({var}())"
        float_mask = f"to_float_mask({mask})"
        return f"{type}::blendv({other_code}, {var}(), {float_mask})"

    @staticmethod
    def index_expr(expr, dtype):
        assert dtype == torch.int64
        opt_ctx: OptimizationContext = get_current_node_opt_ctx()
        assert opt_ctx
        assert opt_ctx.dtype == torch.int32
        assert opt_ctx.is_most_inner_loop_irrevelant
        return f"at::vec::Vectorized<int>(static_cast<int>({cexpr(V.kernel.rename_indexing(expr))}))"


class CppOverrides(OpOverrides):
    """Map element-wise ops to C++"""

    @staticmethod
    def mul(a, b):
        return f"decltype({a})({a} * {b})"

    @staticmethod
    def to_dtype(x, dtype):
        assert dtype in DTYPE_TO_CPP, f"{dtype} missing from {__name__}.DTYPE_TO_CPP"
        return f"static_cast<{DTYPE_TO_CPP[dtype]}>({x})"

    @staticmethod
    def abs(x):
        return f"std::abs({x})"

    @staticmethod
    def sin(x):
        return f"std::sin({x})"

    @staticmethod
    def cos(x):
        return f"std::cos({x})"

    @staticmethod
    def neg(x):
        return f"decltype({x})(-{x})"

    @staticmethod
    def exp(x):
        # return f"Sleef_expf_u10({x})"
        return f"std::exp({x})"

    @staticmethod
    def exp2(x):
        return f"std::exp2({x})"

    @staticmethod
    def expm1(x):
        return f"std::expm1({x})"

    @staticmethod
    def erf(x):
        return f"std::erf({x})"

    @staticmethod
    def erfc(x):
        return f"std::erfc({x})"

    @staticmethod
    def sqrt(x):
        return f"std::sqrt({x})"

    @staticmethod
    def rsqrt(x):
        return f"1 / std::sqrt({x})"

    @staticmethod
    def log1p(x):
        bug = config.cpp.inject_log1p_bug_TESTING_ONLY
        if bug == "accuracy":
            return f"{x} + decltype({x})(1)"
        elif bug is None:
            return f"std::log1p({x})"
        else:
            raise AssertionError(
                f"unrecognized config cpp.inject_log1p_bug_TESTING_ONLY = {bug!r}"
            )

    @staticmethod
    def tan(x):
        return f"std::tan({x})"

    @staticmethod
    def tanh(x):
        return f"std::tanh({x})"

    @staticmethod
    def signbit(x):
        return f"std::signbit({x})"

    @staticmethod
    def pow(a, b):
        return f"std::pow({a}, {b})"

    @staticmethod
    def log(x):
        return f"std::log({x})"

    @staticmethod
    def round(x):
        return f"std::nearbyint({x})"

    @staticmethod
    def floor(x):
        return f"std::floor({x})"

    @staticmethod
    def floordiv(a, b):
        # a and b are integer type
        quot = f"{a} / {b}"
        rem = f"{a} % {b}"
        return f"(({a} < 0) != ({b} < 0) ? ({rem} != 0 ? {quot} - 1 : {quot}) : {quot})"

    @staticmethod
    def ceil(x):
        return f"std::ceil({x})"

    @staticmethod
    def trunc(x):
        return f"std::trunc({x})"

    @staticmethod
    def truncdiv(a, b):
        # a and b are integer type
        return f"{a} / {b}"

    @staticmethod
    def fmod(a, b):
        return f"std::fmod({a}, {b})"

    @staticmethod
    def isinf(x):
        return f"std::isinf({x})"

    @staticmethod
    def isnan(x):
        return f"std::isnan({x})"

    @staticmethod
    def lgamma(x):
        return f"std::lgamma({x})"

    @staticmethod
    def acos(x):
        return f"std::acos({x})"

    @staticmethod
    def acosh(x):
        return f"std::acosh({x})"

    @staticmethod
    def cosh(x):
        return f"std::cosh({x})"

    @staticmethod
    def sinh(x):
        return f"std::sinh({x})"

    @staticmethod
    def asin(x):
        return f"std::asin({x})"

    @staticmethod
    def asinh(x):
        return f"std::asinh({x})"

    @staticmethod
    def atan2(x, y):
        return f"std::atan2({x}, {y})"

    @staticmethod
    def atan(x):
        return f"std::atan({x})"

    @staticmethod
    def atanh(x):
        return f"std::atanh({x})"

    @staticmethod
    def copysign(x, y):
        return f"std::copysign({x}, {y})"

    @staticmethod
    def hypot(x, y):
        return f"std::hypot({x}, {y})"

    @staticmethod
    def log10(x):
        return f"std::log10({x})"

    @staticmethod
    def nextafter(x, y):
        return f"std::nextafter({x}, {y})"

    @staticmethod
    def relu(x):
        bug = config.cpp.inject_relu_bug_TESTING_ONLY
        if bug == "compile_error":
            return "compile error!"
        elif bug == "runtime_error":
            return f"{x}; throw 1"
        elif bug == "accuracy":
            return f"{x} + decltype({x})(1)"
        elif bug is None:
            return f"{x} * ({x}>0)"
        else:
            raise AssertionError(
                f"unrecognized config cpp.inject_relu_bug_TESTING_ONLY = {bug!r}"
            )

    @staticmethod
    def minimum(a, b):
        return f"min_propagate_nan({a}, {b})"

    @staticmethod
    def maximum(a, b):
        return f"max_propagate_nan({a}, {b})"

    @staticmethod
    def where(a, b, c):
        return f"{a} ? {b} : {c}"

    @staticmethod
    def mod(a, b):
        return f"mod({a}, {b})"

    @staticmethod
    def constant(val, dtype):
        if dtype in (torch.float16, torch.bfloat16):
            # Since load promotes all half-precision inputs to float, constants
            # must be promoted as well
            dtype = torch.float32

        if val == float("inf"):
            return f"std::numeric_limits<{DTYPE_TO_CPP[dtype]}>::infinity()"
        elif val == float("-inf"):
            return f"-std::numeric_limits<{DTYPE_TO_CPP[dtype]}>::infinity()"
        elif math.isnan(val):
            return f"std::numeric_limits<{DTYPE_TO_CPP[dtype]}>::quiet_NaN()"
        elif val is True or val is False:
            return ops.to_dtype(str(val).lower(), dtype)
        return ops.to_dtype(repr(val), dtype)

    @staticmethod
    def index_expr(expr, dtype):
        return ops.to_dtype(cexpr(V.kernel.rename_indexing(expr)), dtype)

    @staticmethod
    def masked(mask, body, other):
        code = BracesBuffer()

        # Write masked operation into a lambda
        body_var = V.kernel.cse.newvar()
        code.writeline(f"auto {body_var} = [&]")
        with V.kernel.swap_buffers(code), code.indent():
            result = body()
            code.writeline(f"return {result};")
        code.writeline(";")
        V.kernel.compute.splice(code)

        # Use the lambda's return type as the type of other
        type = f"decltype({body_var}())"

        if other == float("-inf"):
            other_code = f"-std::numeric_limits<{type}>::infinity()"
        elif other == float("inf"):
            other_code = f"std::numeric_limits<{type}>::infinity()"
        elif isinstance(other, bool):
            other_code = f"static_cast<{type}>({str(other).lower()})"
        elif math.isnan(other):
            other_code = f"std::numeric_limits<{type}>::quiet_NaN()"
        else:
            other_code = f"static_cast<{type}>({repr(other)})"

        return f"{mask} ? {body_var}() : {other_code}"

    @staticmethod
    def logical_and(a, b):
        return f"{a} && {b}"

    @staticmethod
    def logical_or(a, b):
        return f"{a} || {b}"

    @staticmethod
    def bitwise_and(x, y):
        return f"decltype({x})({x} & {y})"

    @staticmethod
    def bitwise_or(x, y):
        return f"decltype({x})({x} | {y})"

    @staticmethod
    def bitwise_xor(x, y):
        return f"decltype({x})({x} ^ {y})"

    @staticmethod
    def rand(seed: sympy.Expr, offset: sympy.Expr, dtype):
        return f"static_cast<{DTYPE_TO_CPP[dtype]}>(normalized_rand_cpu({seed}, {offset}));"

    @staticmethod
    def randn(seed: sympy.Expr, offset: sympy.Expr, dtype):
        return f"static_cast<{DTYPE_TO_CPP[dtype]}>(randn_cpu({seed}, {offset}));"

    @staticmethod
    def randint(seed: sympy.Expr, offset: sympy.Expr, dtype):
        return f"static_cast<{DTYPE_TO_CPP[dtype]}>(randint_cpu({seed}, {offset}));"

    @staticmethod
    def sigmoid(x):
        return f"decltype({x})(1) / (decltype({x})(1) + std::exp(-{x}))"

    @staticmethod
    def sign(x):
        code = BracesBuffer()
        # auto tmp5 = tmp4 < 0 ? -1 : 1;
        left = V.kernel.cse.newvar()
        right = V.kernel.cse.newvar()
        result = V.kernel.cse.newvar()
        scalar_zero = f"decltype({x})(0)"
        scalar_one = f"decltype({x})(1)"
        code.writeline(f"auto {left} = {x} > 0 ? {scalar_one} : {scalar_zero};")
        code.writeline(f"auto {right} = {x} < 0 ? {scalar_one} : {scalar_zero};")
        code.writeline(f"auto {result} = {left} - {right};")
        V.kernel.compute.splice(code)
        return result


class CppKernel(Kernel):
    overrides = CppOverrides
    sexpr = cexpr
    newvar_prefix = "auto "
    suffix = ";"

    def __init__(self, args, num_threads):
        super().__init__(args)
        self.call_ranges = None
        self.ranges = None
        self.itervars = None
        self.reduction_depth = None
        self.reduction_prefix = IndentedBuffer()
        self.reduction_suffix = IndentedBuffer()
        self.reduction_var_map = {}
        self.reduction_cse = CSE(self.newvar_prefix, self.suffix, name_prefix="tmp_acc")
        self.preloads = IndentedBuffer()
        self.poststores = IndentedBuffer()
        self.num_threads = num_threads  # num_threads the kernel specialized for

    def scale_index_with_offset(
        self, index: sympy.Expr, scale=1, itervar_idx=-1, offset=0
    ):
        var = self.itervars[itervar_idx]
        replacement = {var: var * scale + offset}
        new_index = sympy_subs(index, replacement)
        return new_index

    def load(self, name: str, index: sympy.Expr):
        var = self.args.input(name)
        index = self.rename_indexing(index)
        line = f"{var}[{cexpr_index(index)}]"
        if V.graph.get_dtype(name) in [torch.float16]:
            line = f"static_cast<float>({line})"
        return self.cse.generate(self.loads, line)

    def store(self, name, index, value, mode=None):
        assert "buf" in name
        var = self.args.output(name)
        index = self.rename_indexing(index)
        if mode is None:
            line = f"{var}[{cexpr_index(index)}] = {value};"
        elif mode == "atomic_add":
            if not config.cpp.dynamic_threads and self.num_threads == 1:
                line = f"{var}[{cexpr_index(index)}] += {value};"
            else:
                line = f"atomic_add(&{var}[{cexpr_index(index)}], {value});"
        else:
            raise NotImplementedError(f"store mode={mode}")
        self.stores.writeline(DeferredLine(name, line))

    def reduction(self, name, dtype, src_dtype, reduction_type, index, value):
        argmax_or_argmin = reduction_type in {"argmax", "argmin"}
        tmpvar = self.reduction_cse.generate(
            self.loads, f"reduction {name} {cexpr_index(index)}", write=False
        )
        index = self.rename_indexing(index)
        self.reduction_var_map[tmpvar] = reduction_type
        if argmax_or_argmin:
            self.reduction_prefix.writelines(
                argmax_argmin_prefix(reduction_type, src_dtype, tmpvar)
            )
            compare_op = "<" if reduction_type == "argmax" else ">"
            self.stores.writelines(
                [
                    f"if ({tmpvar}.value {compare_op} {value}) {{",
                    f"    {tmpvar}.index = {self.itervars[-1]}; {tmpvar}.value = {value};",
                    "}",
                ],
            )
        else:
            if dtype in (torch.float16, torch.bfloat16):
                self.reduction_prefix.writeline(
                    f"float {tmpvar} = {reduction_init(reduction_type, dtype)};"
                )
            else:
                self.reduction_prefix.writeline(
                    f"{DTYPE_TO_CPP[dtype]} {tmpvar} = {reduction_init(reduction_type, dtype)};"
                )
            self.stores.writeline(
                f"{reduction_combine(reduction_type, tmpvar, value)};"
            )

        if name not in V.graph.removed_buffers:
            var = self.args.output(name)
            member_name = ".index" if argmax_or_argmin else ""
            self.reduction_suffix.writeline(
                DeferredLine(
                    name, f"{var}[{cexpr_index(index)}] = {tmpvar}{member_name};"
                )
            )
        self.cse.store_cache[name] = tmpvar

    def set_ranges(self, lengths, reduction_lengths):
        if self.call_ranges:
            assert self.call_ranges == tuple(lengths) + tuple(
                reduction_lengths
            ), f"{self.call_ranges} == {tuple(lengths)} + {tuple(reduction_lengths)}"
            assert self.reduction_depth == len(lengths)
        else:
            self.call_ranges = tuple(lengths) + tuple(reduction_lengths)
            self.ranges = [self.rename_indexing(x) for x in self.call_ranges]
            self.itervars = [sympy_symbol(f"i{n}") for n in range(len(self.ranges))]
            self.reduction_depth = len(lengths)
        return (
            self.itervars[: self.reduction_depth],
            self.itervars[self.reduction_depth :],
        )

    def size_hint(self):
        return V.graph.sizevars.size_hint(sympy_product(self.call_ranges))

    def codegen_loops_impl(self, loop_nest, code, worksharing):
        threads = parallel_num_threads()
        par_depth = self.decide_parallel_depth(
            self.call_ranges[: loop_nest.max_parallel_depth()], threads
        )
        with contextlib.ExitStack() as stack:
            if par_depth:
                if loop_nest.is_reduction_only():
                    # need to close the worksharing scope to define reduction vars outside it
                    worksharing.close()
                else:
                    worksharing.parallel(threads)
                loop_nest.mark_parallel(par_depth)
            elif threads > 1:
                if worksharing.single():
                    stack.enter_context(code.indent())

            def gen_kernel(kernel):
                with contextlib.ExitStack() as stack:
                    assert kernel
                    if hasattr(kernel, "codegen_inner_loops"):
                        code.splice(kernel.preloads)
                        kernel.codegen_inner_loops(code)
                        stack.enter_context(code.indent())
                    code.splice(kernel.loads)
                    code.splice(kernel.compute)
                    code.splice(kernel.stores)
                if hasattr(kernel, "codegen_inner_loops"):
                    code.splice(kernel.poststores)

            def get_reduction_code_buffer(loops, is_suffix=True):
                for loop in loops:
                    for kernel in loop.get_kernels():
                        if is_suffix:
                            return kernel.reduction_suffix
                        else:
                            return kernel.reduction_prefix
                return None

            def gen_loops(loops: List[LoopLevel], in_reduction=False):
                with contextlib.ExitStack() as stack_outer:
                    if loops:
                        loop = loops[0]
                        if loop.is_reduction() and not in_reduction:
                            reduction_prefix = get_reduction_code_buffer(
                                loops, is_suffix=False
                            )
                            if reduction_prefix:
                                stack_outer.enter_context(code.indent())
                            code.splice(reduction_prefix)
                        if loop_nest.is_reduction_only() and loop.parallel:
                            worksharing.parallel(threads)

                    for loop in loops:
                        gen_loop(loop, in_reduction)

                    if loops:
                        loop = loops[0]
                        if loop_nest.is_reduction_only() and loop.parallel:
                            worksharing.close()
                        if loop.is_reduction() and not in_reduction:
                            code.splice(
                                get_reduction_code_buffer(loops, is_suffix=True)
                            )

            def gen_loop(loop: LoopLevel, in_reduction=False):
                with contextlib.ExitStack() as stack:
                    loop_lines = loop.lines()
                    if loop_lines is None:
                        return
                    code.writelines(loop_lines)
                    stack.enter_context(code.indent())
                    # generate inner loops or loop body
                    if loop.inner:
                        gen_loops(loop.inner, loop.is_reduction())
                    else:
                        kernels = loop.get_kernels()
                        assert len(kernels) == 1
                        gen_kernel(kernels[0])

            stack.enter_context(code.indent())
            if loop_nest.root:
                gen_loops(loop_nest.root)
            else:
                gen_kernel(loop_nest.kernel)

    def codegen_loops(self, code, worksharing):
        loop_nest = LoopNestWithSplit.build(self)
        self.codegen_loops_impl(loop_nest, code, worksharing)

    def decide_parallel_depth(self, ranges, threads):
        seq = self.size_hint()
        par = 1
        depth = 0
        for expr in ranges:
            hint = V.graph.sizevars.size_hint(expr)
            if par >= 2 * threads or par == threads:
                break
            if seq // threads < config.cpp.min_chunk_size:
                # not enough work
                break
            depth += 1
            par *= hint
            seq /= hint
        # if we assume thread number is dynamic, make sure we
        # have at least one parallel scope and let OMP runtime
        # to manage the serial vs. parallel.
        if config.cpp.dynamic_threads and depth == 0 and len(ranges) > 0:
            depth = 1
        return depth

    @contextlib.contextmanager
    def write_to_suffix(self):
        prior = (self.loads, self.compute, self.stores, self.cse)
        self.loads = IndentedBuffer()
        self.compute = IndentedBuffer()
        self.stores = IndentedBuffer()
        self.cse = self.cse.clone()
        yield
        self.reduction_suffix.splice(self.loads)
        self.reduction_suffix.splice(self.compute)
        self.reduction_suffix.splice(self.stores)
        (self.loads, self.compute, self.stores, self.cse) = prior


class CppVecKernel(CppKernel):
    overrides = CppVecOverrides

    def __init__(self, args, num_threads, tiling_factor=0, tiling_idx=-1):
        super().__init__(args, num_threads)
        assert codecache.pick_vec_isa()
        if tiling_factor == 0:
            tiling_factor = codecache.pick_vec_isa().nelements()
        self.tiling_factor = tiling_factor
        self.tiling_idx = tiling_idx
        self.reduction_omp_dec: Dict[str, str] = {}
        metrics.generated_cpp_vec_kernel_count += 1

    def load(self, name: str, index: sympy.Expr):
        opt_ctx: OptimizationContext = get_current_node_opt_ctx()
        var = self.args.input(name)
        index = self.rename_indexing(index)
        dtype = V.graph.get_dtype(name)
        tiling_var = self.itervars[self.tiling_idx]
        is_broadcast = not index.has(tiling_var)
        is_mask = dtype in [torch.bool, torch.uint8]
        non_contiguous = (
            not is_broadcast
            and stride_at(tiling_var, index) != 1
            or "tmp" in f"{index}"
        )
        var_expr = (
            f"{var}[{cexpr_index(index)}]"
            if is_broadcast
            else f"{var} + {cexpr_index(index)}"
        )
        loadbuf = "tmpbuf" if non_contiguous else var_expr
        if is_broadcast:
            if is_mask:
                loadbuf = f"flag_to_float_scalar({loadbuf})"
            line = f"at::vec::Vectorized<float>(static_cast<float>({loadbuf}))"
        elif dtype in [torch.uint8] and opt_ctx.is_load_uint8_as_float:
            line = f"at::vec::load_uint8_as_float({var_expr})"
        elif is_mask:
            line = f"flag_to_float_vec({loadbuf})"
        elif dtype in [torch.bfloat16]:
            if opt_ctx.is_load_bf16_as_fp32:
                line = f"load_bf16_as_float({loadbuf})"
            else:
                assert opt_ctx.is_bf16_mem_copy
                line = f"at::vec::Vectorized<bfloat16>::loadu({loadbuf}, {self.tiling_factor})"
        else:
            line = f"at::vec::Vectorized<float>::loadu({loadbuf})"
        if non_contiguous:
            tmpbuftype = "float" if is_mask else f"{DTYPE_TO_CPP[dtype]}"
            tmpbufsize = f"{self.tiling_factor}"
            if dtype in [torch.bfloat16]:
                tmpbufsize += " * 2"
            tmpbufdeclare = f"__at_align__ {tmpbuftype} tmpbuf[{tmpbufsize}];"
            inner = sympy.symbols(f"{tiling_var}_inner")
            new_index = self.scale_index_with_offset(
                index, itervar_idx=self.tiling_idx, offset=inner
            )
            tmpbufdefine = (
                f"for (long {inner} = 0; {inner} < {self.tiling_factor}; {inner}++) "
            )
            rhs = f"{var}[{cexpr_index(new_index)}]"
            if is_mask:
                rhs = f"flag_to_float_scalar({rhs})"
            tmpbufdefine += f"tmpbuf[{inner}] = {rhs};"
            line = f"([&]() {{ {tmpbufdeclare} {tmpbufdefine} return {line}; }})()"

        return self.cse.generate(self.loads, line)

    def store(self, name, index, value, mode=None):
        assert "buf" in name
        assert mode is None
        opt_ctx: OptimizationContext = get_current_node_opt_ctx()
        var = self.args.output(name)
        index = self.rename_indexing(index)
        tiling_var = self.itervars[self.tiling_idx]
        assert index.has(tiling_var)
        var_expr = f"{var} + {cexpr_index(index)}"
        dtype = V.graph.get_dtype(name)
        non_contiguous = stride_at(tiling_var, index) != 1 or "tmp" in f"{index}"
        if non_contiguous:
            var_expr = "tmpbuf"
        if V.graph.get_dtype(name) in [torch.bfloat16]:
            if opt_ctx.is_store_fp32_as_bf16:
                line = f"store_float_as_bf16({var_expr}, {value});"
            else:
                assert opt_ctx.is_bf16_mem_copy
                line = f"{value}.store({var_expr}, {self.tiling_factor});"
        elif (V.graph.get_dtype(name) in [torch.uint8]) and (
            opt_ctx.is_store_float_as_uint8
        ):
            # TODO(Leslie): Optimize the implementation of store_float_as_uint8
            # * Pattern match of quantization op in the loop body.
            # * Skip the explicit saturation and clamp inside store_float_as_uint8.
            line = f"at::vec::store_float_as_uint8({value}, {var_expr});"
        else:
            line = f"{value}.store({var_expr});"
        if non_contiguous:
            inner = sympy.symbols(f"{tiling_var}_inner")
            new_index = self.scale_index_with_offset(
                index, itervar_idx=self.tiling_idx, offset=inner
            )
            tmp_bufsize = (
                f"{self.tiling_factor}*sizeof(float)/sizeof({DTYPE_TO_CPP[dtype]})"
            )
            line = (
                f"{{ __at_align__ {DTYPE_TO_CPP[dtype]} tmpbuf[{tmp_bufsize}]; {line} "
                f"for (long {inner} = 0; {inner} < {self.tiling_factor}; {inner}++) "
                f"{var}[{cexpr_index(new_index)}] = tmpbuf[{inner}]; }}"
            )
        self.stores.writeline(DeferredLine(name, line))

    def reduction(self, name, dtype, src_dtype, reduction_type, index, value):
        assert reduction_type in {"max", "min", "sum", "prod", "xor_sum"}
        assert dtype == torch.float
        assert src_dtype == torch.float
        reduce_map = {"max": "maximum", "min": "minimum"}

        vec_ns = "at::vec"
        vec = f"{vec_ns}::Vectorized<{DTYPE_TO_CPP[dtype]}>"

        if reduction_type not in self.reduction_omp_dec:
            vec_reduc_prefix = "#pragma omp declare reduction("
            vec_reduc_prefix += f"{RTYPE_TO_CPP[reduction_type]}:{vec}:"
            if reduction_type == "sum":
                vec_reduc_prefix += "omp_out += omp_in"
            elif reduction_type == "prod":
                vec_reduc_prefix += "omp_out *= omp_in"
            elif reduction_type == "xor_sum":
                vec_reduc_prefix += "omp_out ^= omp_in"
            else:
                vec_reduc_prefix += (
                    f"omp_out = {vec_ns}::{reduce_map[reduction_type]}(omp_out, omp_in)"
                )
            vec_reduc_prefix += ")"
            vec_reduc_prefix += " initializer("
            vec_reduc_prefix += "omp_priv={{"
            vec_reduc_prefix += f"{reduction_init(reduction_type, dtype)}"
            vec_reduc_prefix += "}})"
            self.reduction_omp_dec[reduction_type] = RTYPE_TO_CPP[reduction_type]
            self.reduction_prefix.writeline(vec_reduc_prefix)

        tmpvar = self.reduction_cse.generate(
            self.loads, f"reduction {name} {cexpr_index(index)}", write=False
        )
        tmpvar_vec = f"{tmpvar}_vec"

        index = self.rename_indexing(index)
        self.reduction_var_map[tmpvar_vec] = reduction_type
        self.reduction_prefix.writeline(
            f"{DTYPE_TO_CPP[dtype]} {tmpvar} = {reduction_init(reduction_type, dtype)};"
        )
        self.reduction_prefix.writeline(
            f"auto {tmpvar_vec} = at::vec::Vectorized<{DTYPE_TO_CPP[dtype]}>({tmpvar});"
        )
        self.stores.writeline(
            f"{reduction_combine_vec(reduction_type, tmpvar_vec, value)};"
        )

        if self.tiling_idx >= self.reduction_depth:
            # Horizontal reduction
            reduce_all_body = "{"
            if reduction_type == "sum":
                reduce_all_body += "return x + y;"
            elif reduction_type == "prod":
                reduce_all_body += "return x * y;"
            elif reduction_type == "xor_sum":
                reduce_all_body += "return x ^ y;"
            else:
                reduce_all_body += (
                    f"return {vec_ns}::{reduce_map[reduction_type]}(x, y);"
                )
            reduce_all_body += "}"
            vec_reduce_all_func = f"{vec_ns}::vec_reduce_all<{DTYPE_TO_CPP[dtype]}>"
            next_value = f"{vec_reduce_all_func}([]({vec}& x, {vec}&y) {reduce_all_body}, {tmpvar_vec})"
            self.reduction_suffix.writeline(
                DeferredLine(
                    name, f"{reduction_combine(reduction_type, tmpvar, next_value)};"
                )
            )

        if name not in V.graph.removed_buffers:
            var = self.args.output(name)
            if self.tiling_idx >= self.reduction_depth:
                # Horizontal reduction
                self.reduction_suffix.writeline(
                    DeferredLine(name, f"{var}[{cexpr_index(index)}] = {tmpvar};")
                )
            else:
                # Vertical reduction
                self.reduction_suffix.writeline(
                    DeferredLine(
                        name, f"{tmpvar_vec}.store({var} + {cexpr_index(index)});"
                    )
                )

        self.cse.store_cache[name] = tmpvar


class CppTile2DKernel(CppVecKernel):
    """
    A vector kernel that handles the 2d tiles with the tile size defined in `tiling_factor` on
    the inner-most loop level and one of the outer loop level (`outer_tiling_idx`). When the data
    tile is accessed in a contiguous way from the outer loop axis, a transposition is applied on the
    tile to make the access contiguous from the inner-most loop axis. Then, the same vectorization
    logic from its parent `CppVecKernel` is leveraged for load/store/compute. The transposed tile load
    and store are generated into kernel.preloads and kernel.poststores buffers.

    The loop structure looks like below:
    for ...
      for i_outer ...
        for ...
          for inner_most ...
            // generated by CppTile2DKernel
            float tmp0[16*16]; at::vec::transpose_mxn<...>(tmp0, in_ptr0 + ..., ...); // into kernel.preloads
            float tmp1[16*16]; // into kernel.preloads
            for i_inner ... { // the kernel inner loop
              vectorized loads/compute/stores (e.g., load tmp0, store tmp1) // into kernel.loads/compute/stores
            }
            at::vec::transpose_mxn(out_ptr0 + ..., tmp1, ...) // into kernel.poststores
          for inner_most ... (tail)
            // generated by CppVecKernel
            ...
      for i_outer ... (tail)
        for ...
          for ...
            // generated by CppKernel
            ...
    """

    def __init__(self, args, num_threads, tiling_factor, tiling_indices):
        super().__init__(args, num_threads, tiling_factor, tiling_indices[1])
        self.tiling_indices = tiling_indices

    def inner_itervar(self):
        return sympy.symbols(f"{self.itervars[self.outer_idx]}_inner")

    def need_vec_transpose(self, index):
        return stride_at(self.itervars[self.outer_idx], index) == 1 and index.has(
            self.itervars[self.tiling_idx]
        )

    def gen_transposed_tile_load_store(self, name, var, index, is_store):
        # transposed tile load/store outside the kernel inner loop
        dtype = V.graph.get_dtype(name)
        factor = self.tiling_factor
        src = f"{var} + {cexpr_index(index)}"
        dst = "__place_holder__"
        ld_src = f"{cexpr_index(stride_at(self.itervars[self.tiling_idx], index))}"
        ld_dst = f"{factor}"
        if is_store:
            src, dst = dst, src
            ld_src, ld_dst = ld_dst, ld_src

        need_define = True
        load_or_store = f"at::vec::transpose_mxn<{DTYPE_TO_CPP[dtype]},{factor},{factor}>({src}, {ld_src}, {dst}, {ld_dst});"
        if is_store:
            tile_var = self.cse.newvar()
        elif load_or_store not in self.cse.cache:
            tile_var = self.cse.generate(self.preloads, load_or_store, write=False)
        else:
            need_define = False
            tile_var = self.cse.cache[load_or_store]

        if need_define:
            define_line = f"{DTYPE_TO_CPP[dtype]} {tile_var}[{factor}*{factor}] __attribute__ ((aligned ({factor})));"
            self.preloads.writeline(define_line)

        load_or_store = load_or_store.replace("__place_holder__", str(tile_var))
        if is_store:
            self.poststores.writeline(DeferredLine(name, load_or_store))
        else:
            self.preloads.writeline(load_or_store)

        return tile_var

    def load(self, name: str, index: sympy.Expr):
        opt_ctx: OptimizationContext = get_current_node_opt_ctx()
        var = self.args.input(name)
        index = self.rename_indexing(index)

        inner = self.inner_itervar()
        if self.need_vec_transpose(index):
            tile_var = self.gen_transposed_tile_load_store(
                name, var, index, is_store=False
            )
            # vector load inside the kernel inner loop
            loadbuf = f"{tile_var} + {cexpr_index(inner * self.tiling_factor)}"
            if V.graph.get_dtype(name) in [torch.bfloat16]:
                if opt_ctx.is_load_bf16_as_fp32:
                    line = f"load_bf16_as_float({loadbuf})"
                else:
                    assert opt_ctx.is_bf16_mem_copy
                    line = f"at::vec::Vectorized<bfloat16>::loadu({loadbuf}, {self.tiling_factor})"
            else:
                line = f"at::vec::Vectorized<float>::loadu({loadbuf})"
            return self.cse.generate(self.loads, line)
        else:
            new_index = self.scale_index_with_offset(
                index,
                itervar_idx=self.outer_idx,
                offset=inner,
            )
            return super().load(name, new_index)

    def store(self, name, index, value, mode=None):
        assert "buf" in name
        opt_ctx: OptimizationContext = get_current_node_opt_ctx()
        var = self.args.output(name)

        inner = self.inner_itervar()
        index = self.rename_indexing(index)
        assert mode is None
        if self.need_vec_transpose(index):
            tile_var = self.gen_transposed_tile_load_store(
                name, var, index, is_store=True
            )
            # vector store inside the kernel inner loop
            storebuf = f"{tile_var} + {cexpr_index(inner * self.tiling_factor)}"
            if V.graph.get_dtype(name) in [torch.bfloat16]:
                if opt_ctx.is_store_fp32_as_bf16:
                    line = f"store_float_as_bf16({storebuf}, {value})"
                else:
                    assert opt_ctx.is_bf16_mem_copy
                    line = f"{value}.store({storebuf}, {self.tiling_factor})"
            else:
                line = f"{value}.store({storebuf});"
            self.stores.writeline(DeferredLine(name, line))
        else:
            new_index = self.scale_index_with_offset(
                index,
                itervar_idx=self.outer_idx,
                offset=inner,
            )
            super().store(name, new_index, value, mode)

    def codegen_inner_loops(self, code):
        inner = self.inner_itervar()
        code.writeline(
            f"for (long {inner} = 0; {inner} < {self.tiling_factor}; {inner}++)"
        )

    def set_ranges(self, group, reduction_group):
        vars = super().set_ranges(group, reduction_group)
        # do vertical reduction as the tail loop
        self.outer_idx, self.tiling_idx = (
            self.tiling_indices
            if self.tiling_indices[1] < self.reduction_depth
            else reversed(self.tiling_indices)
        )
        return vars


class CppVecKernelChecker(CppVecKernel):
    def __init__(self, args, num_threads, tiling_factor, tiling_idx=-1):
        super().__init__(args, num_threads, tiling_factor, tiling_idx)

        # Since this kernel is only for checker but does not generate any
        # code, so we need to decrease the kernel count.
        metrics.generated_kernel_count -= 1
        metrics.generated_cpp_vec_kernel_count -= 1

        # Used to record the graph wrapper code as the wrapper_code status could be
        # changed during graph run.
        self._orig_wrapper_code = None

        self.simd_vec = True
        self.fast_vec_list = []
        for k, v in CppVecOverrides.__dict__.items():
            if isinstance(v, staticmethod):
                self.fast_vec_list.append(k)
        self.exit_stack = contextlib.ExitStack()

        # Cache all the load result
        self.load_supported_dtypes: list[torch.dtype] = [
            torch.float,
            torch.bfloat16,
            torch.bool,
            torch.uint8,
        ]
        self.store_supported_dtypes: list[torch.dtype] = [
            torch.float,
            torch.bfloat16,
            torch.uint8,
        ]
        # Cache the dtypes of the store operation. If the store is mixing dtypes, the
        # vectorization would not support it as it is hard to determine the vec dtype
        self.store_dtypes: list[torch.dtype] = []
        # The dtype is used for vectorization
        self.vec_dtype: torch.dtype = torch.float32

    def disable_vec(self, msg=None):
        if schedule_log.isEnabledFor(logging.DEBUG):
            schedule_log.debug("Disabled vectorization: %s", msg)
        self.simd_vec = False

    def could_vec(self, name: str, index: sympy.Expr):
        assert self.itervars is not None
        return len(self.itervars) > 0

    def is_mask(self, name: str, users: Dict[torch.fx.Node, None]):
        load_type = V.graph.get_dtype(name)
        if load_type == torch.bool:
            return all(user.target in ("where", "masked") for user in users.keys())
        elif load_type == torch.uint8:
            """
            If the load value is torch.uint8, then we only support the loaded
            value is as the mask.
            """
            if not all(
                user.target == "to_dtype" and user.args[-1] == torch.bool
                for user in users.keys()
            ):
                return False

            for to_dtype_node in users.keys():
                assert to_dtype_node.target == "to_dtype"
                if not all(
                    user.target in ("where", "masked")
                    for user in to_dtype_node.users.keys()
                ):
                    return False
            return True
        else:
            return False

    def can_load_bf16_as_fp32(self, input_node: torch.fx.Node):
        assert input_node.target in ["load", "constant"]
        load_type = (
            V.graph.get_dtype(input_node.args[1])
            if input_node.target == "load"
            else input_node.args[-1]
        )
        if load_type not in [torch.bfloat16]:
            return False

        if not all(
            user.target == "to_dtype" and user.args[-1] == torch.float
            for user in input_node.users
        ):
            return False

        return True

    def can_store_fp32_as_bf16(self, store_var: str, value_node: torch.fx.Node):
        load_type = V.graph.get_dtype(store_var)
        if load_type not in [torch.bfloat16]:
            return False

        if value_node.target == "to_dtype" and value_node.args[-1] == torch.bfloat16:
            return True

        return False

    def is_load_uint8_as_float(self, name: str, users: Dict[torch.fx.Node, None]):
        """
        Check:
        1. load_type is torch.uint8
        2. has 1 user node of target to_dtype
        3. dtype of to_dtype is torch.float
        """
        load_type = V.graph.get_dtype(name)
        if load_type is not torch.uint8:
            return False
        if len(users) == 1:
            user = list(users)[0]
            if (user.target == "to_dtype") and (user.args[-1] == torch.float):
                return True
            return False
        return False

    def can_store_fp32_as_uint8(self, store_var: str, value_node: torch.fx.Node):
        """
        Check:
        1. store_type is torch.uint8
        2. value_node is of target to_dtype
        3. dtype of to_dtype node is torch.uint8
        """
        store_type = V.graph.get_dtype(store_var)
        if store_type not in [torch.uint8]:
            return False
        if value_node.target == "to_dtype" and value_node.args[-1] == torch.uint8:
            return True

        return False

    def is_load_integer_scalar_tensor(self, name: str, index: sympy.Expr):
        load_dtype = V.graph.get_dtype(name)
        buffer = V.graph.get_buffer(name)
        return (
            load_dtype in [torch.int32, torch.int64]
            and isinstance(buffer, TensorBox)
            and isinstance(buffer.data, StorageBox)
            and (len(buffer.data.layout.size) == 0)
            and (index == 0)
        )

    def load(self, name: str, index: sympy.Expr):
        with RecordOptimizationContext(__name__) as node_ctx:
            load_dtype = V.graph.get_dtype(name)
            opt_ctx: OptimizationContext = node_ctx.get_opt_ctx()
            assert opt_ctx
            opt_ctx.dtype = load_dtype
            opt_ctx.is_load_as_mask = self.is_mask(name, node_ctx.get_fx_node().users)
            opt_ctx.is_load_uint8_as_float = self.is_load_uint8_as_float(
                name, node_ctx.get_fx_node().users
            )

            var = self.cse.newvar()

            if load_dtype in [torch.bool, torch.uint8] and not (
                opt_ctx.is_load_as_mask or opt_ctx.is_load_uint8_as_float
            ):
                if not opt_ctx.is_load_as_mask:
                    self.disable_vec(f"{load_dtype} not loaded as mask")
                elif not opt_ctx.is_load_uint8_as_float:
                    self.disable_vec(f"{load_dtype} not loaded as float")
                return var

            if (
                load_dtype not in self.load_supported_dtypes
            ) and not self.is_load_integer_scalar_tensor(name, index):
                self.disable_vec(f"{load_dtype} not supported by load")
                return var

            if load_dtype in [torch.bfloat16]:
                opt_ctx.is_load_bf16_as_fp32 = self.can_load_bf16_as_fp32(
                    node_ctx.get_fx_node()
                )
                if not (opt_ctx.is_load_bf16_as_fp32 or opt_ctx.is_bf16_mem_copy):
                    self.disable_vec("bfloat16 not legalized as float on load")
                    return var

            index = self.rename_indexing(index)
            if self.simd_vec and not self.could_vec(name, index):
                self.disable_vec(f"not a loop: {index}")
            return var

    def store(self, name, index, value, mode=None):
        with RecordOptimizationContext(__name__) as node_ctx:
            store_dtype = V.graph.get_dtype(name)

            opt_ctx: OptimizationContext = node_ctx.get_opt_ctx()
            assert opt_ctx
            opt_ctx.dtype = store_dtype

            store_dtype = torch.float if store_dtype == torch.float32 else store_dtype
            self.store_dtypes.append(store_dtype)
            if store_dtype not in self.store_supported_dtypes:
                self.disable_vec(f"{store_dtype} not supported by store")
                return self.simd_vec

            if store_dtype in [torch.bfloat16]:
                value_node = node_ctx.get_fx_node().all_input_nodes[-1]
                opt_ctx.is_store_fp32_as_bf16 = self.can_store_fp32_as_bf16(
                    name, value_node
                )
                if not (opt_ctx.is_store_fp32_as_bf16 or opt_ctx.is_bf16_mem_copy):
                    self.disable_vec("bfloat16 not legalized as float on store")
                    return self.simd_vec
            elif store_dtype in [torch.uint8]:
                value_node = node_ctx.get_fx_node().all_input_nodes[-1]
                opt_ctx.is_store_float_as_uint8 = self.can_store_fp32_as_uint8(
                    name, value_node
                )
                if not opt_ctx.is_store_float_as_uint8:
                    self.disable_vec("not support store float32 as uint8")
                    return self.simd_vec

            assert "buf" in name
            index = self.rename_indexing(index)

            if mode:
                self.disable_vec(f"store mode: {mode}")
                return self.simd_vec

            if self.simd_vec and not self.could_vec(name, index):
                self.disable_vec(f"not a loop: {index}")
            return self.simd_vec

    def reduction(self, name, dtype, src_dtype, reduction_type, index, value):
        if (
            dtype == torch.float
            and src_dtype == torch.float
            and reduction_type in ["max", "min", "sum", "prod", "xor_sum"]
        ):
            pass
        else:
            self.disable_vec(
                f"reduction: dtype {dtype}, src_dtype {src_dtype}, reduction_type {reduction_type}"
            )
        return self.simd_vec

    def is_supported_cmp(self, node: torch.fx.Node):
        def get_node_dtype(node):
            if type(node) == torch.fx.Node:
                opt_ctx: OptimizationContext = get_current_node_opt_ctx()
                return opt_ctx.dtype if opt_ctx else None
            else:
                return None

        def get_cmp_dtypes(node: torch.fx.Node):
            return get_node_dtype(node.args[-2]), get_node_dtype(node.args[-1])

        assert len(node.args) >= 2
        # cmp(x, y): y is a magic value like x >= 1
        if type(node.args[-1]) in [int, float]:
            return True
        # cmp(x, y): x is a magic value like 1 >= y
        if type(node.args[-2]) in [int, float]:
            return False

        left_dtype, right_dtype = get_cmp_dtypes(node)
        if left_dtype is None or right_dtype is None:
            # TODO(Eikan): To record, deduce and propagate the data type of every expression.
            return True
        else:
            return left_dtype == right_dtype

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert self._orig_wrapper_code is not None
        # Restore the wrapper_code
        V.graph.wrapper_code = self._orig_wrapper_code
        self.exit_stack.__exit__(exc_type, exc_val, exc_tb)

    def __enter__(self):
        # Record the graph wrapper code. The wrapper_code status could be
        # changed during graph run. Regarding this checker, we also need to
        # run the graph but we don't expect to change any status that would
        # impact the code generation. Hence, we record the graph wrapper code
        # and replace it with a dummy wrapper_code and then restore to the
        # original one as long as the checker is finished.
        self._orig_wrapper_code = V.graph.wrapper_code
        V.graph.wrapper_code = WrapperCodeGen()

        class VecCheckerProxy:
            bin_cmp_ops = ["eq", "ne", "le", "ge", "lt", "gt"]

            @staticmethod
            def _bin_cmp_op(x, y):
                current_node: torch.fx.Node = V.interpreter.current_node
                if not self.is_supported_cmp(current_node):
                    self.disable_vec(f"binary comparison op: {current_node}")
                return self.simd_vec

            @staticmethod
            def __getattr__(name):
                def inner(*args, **kwargs):
                    if name in VecCheckerProxy.bin_cmp_ops:
                        return VecCheckerProxy._bin_cmp_op(args, kwargs)

                    if name not in self.fast_vec_list:
                        self.disable_vec(f"op: {name}")
                    return self.simd_vec

                return inner

            @staticmethod
            def load(name: str, index: sympy.Expr):
                return self.load(name, index)

            @staticmethod
            def store(name, index, value, mode=None):
                return self.store(name, index, value, mode=mode)

            @staticmethod
            def reduction(name, dtype, src_dtype, reduction_type, index, value):
                return self.reduction(
                    name, dtype, src_dtype, reduction_type, index, value
                )

            @staticmethod
            def constant(val, dtype):
                with RecordOptimizationContext(__name__) as node_ctx:
                    opt_ctx: OptimizationContext = node_ctx.get_opt_ctx()
                    assert opt_ctx
                    opt_ctx.dtype = dtype
                    i32_iinfo = numpy.iinfo(numpy.int32)
                    if (
                        dtype == torch.int64
                        and val <= i32_iinfo.max
                        and val >= i32_iinfo.min
                    ):
                        opt_ctx.dtype = torch.int32

                    f32_iinfo = numpy.finfo(numpy.float32)
                    if dtype == torch.double:
                        if (
                            (val <= f32_iinfo.max and val >= f32_iinfo.min)
                            or (val == numpy.inf)
                            or (val == -numpy.inf)
                        ):
                            opt_ctx.dtype = torch.float32

                    supported_dtypes = [torch.float32, torch.int32, torch.bfloat16]

                    if opt_ctx.dtype not in supported_dtypes or (
                        opt_ctx.dtype == torch.int32
                        and not all(
                            user.target in VecCheckerProxy.bin_cmp_ops
                            for user in node_ctx.current_node.users
                        )
                    ):
                        self.disable_vec(f"constant dtype: {opt_ctx.dtype}")

                    if opt_ctx.dtype in [torch.bfloat16]:
                        if self.can_load_bf16_as_fp32(node_ctx.get_fx_node()):
                            opt_ctx.is_load_bf16_as_fp32 = True
                            opt_ctx.dtype = torch.float
                        elif not opt_ctx.is_bf16_mem_copy:
                            self.disable_vec(
                                "bfloat16 not legalized as float in constant"
                            )

                    return val

            @staticmethod
            def index_expr(expr, dtype):
                current_node: torch.fx.Node = V.interpreter.current_node

                assert len(self.ranges) == len(self.itervars)
                if not len(self.ranges) or not all(
                    not isinstance(range, sympy.Expr) or sympy.simplify(range).is_number
                    for range in self.ranges
                ):
                    # if the range value is sympy.Expr, we might could not deduce the accurate loop interval.
                    self.disable_vec(f"index_expr: {expr}, dtype {dtype}")
                    return self.cse.newvar()

                def mod_indexing_rep(x, y, z):
                    if z.is_constant():
                        return x / y

                    # never really happens, we'll bail on optimizing
                    return (x / y) % z

                def indexing_div_rep(x, y):
                    return x / y

                with RecordOptimizationContext(__name__) as node_ctx:
                    assert len(self.ranges) == len(self.itervars)

                    opt_ctx: OptimizationContext = node_ctx.get_opt_ctx()
                    assert opt_ctx
                    max_expr = expr.replace(
                        ir.ModularIndexing, mod_indexing_rep
                    ).replace(ir.FloorDiv, indexing_div_rep)
                    min_expr = max_expr
                    for idx in range(len(self.ranges)):
                        max_expr = sympy.maximum(
                            max_expr,
                            self.itervars[idx],
                            sympy.Interval(0, self.ranges[idx]),
                        )
                        min_expr = sympy.minimum(
                            min_expr,
                            self.itervars[idx],
                            sympy.Interval(0, self.ranges[idx]),
                        )
                    i32_iinfo = numpy.iinfo(numpy.int32)
                    if (
                        dtype == torch.int64
                        and max_expr.is_number
                        and min_expr.is_number
                        and max_expr <= i32_iinfo.max
                        and min_expr >= i32_iinfo.min
                        and all(
                            user.target in VecCheckerProxy.bin_cmp_ops
                            for user in node_ctx.current_node.users
                        )
                    ):
                        opt_ctx.dtype = torch.int32
                    else:
                        opt_ctx.dtype = dtype
                        self.disable_vec(f"index_expr: {expr}, dtype {dtype}")

                    tiling_var = self.itervars[self.tiling_idx]
                    tiling_var_irrelevant = not expr.has(tiling_var)
                    if not tiling_var_irrelevant:
                        self.disable_vec(
                            f"index_expr (tiling var relevant): {expr}, dtype {dtype}"
                        )
                    opt_ctx.is_most_inner_loop_irrevelant = tiling_var_irrelevant
                    tmp_var = self.cse.newvar()
                    return tmp_var

            @staticmethod
            def indirect_indexing(index_var, size):
                return sympy.Symbol(str(index_var))

            @staticmethod
            def masked(mask, body, other):
                body()
                return self.cse.newvar()

            @staticmethod
            def to_dtype(x, dtype):
                with RecordOptimizationContext(__name__) as node_ctx:
                    opt_ctx: OptimizationContext = node_ctx.get_opt_ctx()
                    assert opt_ctx
                    opt_ctx.dtype = dtype

                    cur_node = node_ctx.get_fx_node()
                    input_value: torch.fx.Node = cur_node.all_input_nodes[1]
                    if dtype == torch.float:
                        if input_value.target in ["load", "constant"]:
                            # Support masked_load for BF16. Because the legalization will
                            # insert to_dtype to convert the BF16 input to FP32.
                            dtype = (
                                V.graph.get_dtype(input_value.args[1])
                                if input_value.target == "load"
                                else input_value.args[-1]
                            )
                            if dtype in [torch.bfloat16]:
                                opt_ctx.is_load_bf16_as_fp32 = True
                            elif (dtype == torch.uint8) and (
                                input_value.target == "load"
                            ):
                                # 1. doing uint8 to float.
                                # 2. the previous node of uint8 to float is load.
                                opt_ctx.is_load_uint8_as_float = True
                            elif dtype == torch.float:
                                pass
                            elif (
                                dtype in [torch.int32, torch.int64]
                                and input_value.target == "load"
                            ):
                                buffer = V.graph.get_buffer(input_value.args[1])
                                # Check if load of a scalar tensor of integer
                                if not (
                                    isinstance(buffer, TensorBox)
                                    and isinstance(buffer.data, StorageBox)
                                    and len(buffer.data.layout.size) == 0
                                ):
                                    self.disable_vec(f"to_dtype: dtype {dtype}")
                            else:
                                self.disable_vec(f"to_dtype: dtype {dtype}")
                    elif dtype == torch.bfloat16:
                        if not all(usr.target == "store" for usr in cur_node.users):
                            self.disable_vec(
                                "to_dtype: bfloat16 expecting users are all stores"
                            )
                            return x

                        store_names = [usr.args[1] for usr in cur_node.users]
                        if not all(
                            V.graph.get_dtype(name) in [torch.bfloat16]
                            for name in store_names
                        ):
                            self.disable_vec(
                                "to_dtype: expecting all stores into bfloat16"
                            )
                            return x

                        opt_ctx.is_store_fp32_as_bf16 = True
                    elif dtype == torch.bool:
                        pass
                    elif dtype == torch.uint8:
                        opt_ctx.is_store_float_as_uint8 = all(
                            usr.target in ["store"] for usr in cur_node.users
                        )
                        if not opt_ctx.is_store_float_as_uint8:
                            self.disable_vec(f"to_dtype: dtype {dtype}")
                    else:
                        self.disable_vec(f"to_dtype: dtype {dtype}")
                    return x

        self.exit_stack.enter_context(V.set_ops_handler(VecCheckerProxy()))
        self.exit_stack.enter_context(V.set_kernel_handler(self))
        return self


class CppKernelProxy(CppKernel):
    def __init__(self, kernel_group):
        super().__init__(kernel_group.args, kernel_group.ws.num_threads)
        self.kernel_group = kernel_group
        self.loop_nest = None
        self.call_ranges = None
        self.picked_vec_isa: codecache.VecISA = codecache.pick_vec_isa()

    def data_type_propagation(self, nodes):
        for _node in nodes:
            assert isinstance(_node, SchedulerNode)
            data_type_propagation(_node)

    def legalize_bf16(self, nodes):
        def add_to_dtype(sub_graph: torch.fx.Graph):
            def is_bf16_mem_copy(node: torch.fx.Node):
                if node.target in ["load", "constant"]:
                    bf16_mem_copy = all(
                        usr.target == "store"
                        and V.graph.get_dtype(usr.args[1]) == torch.bfloat16
                        for usr in node.users
                    )
                elif node.target == "store":
                    stored_node = node.args[3]
                    bf16_mem_copy = is_bf16_mem_copy(stored_node)
                else:
                    bf16_mem_copy = False
                if bf16_mem_copy:
                    opt_ctx = OptimizationContext()
                    opt_ctx.is_bf16_mem_copy = bf16_mem_copy
                    node.meta[OptimizationContext.key] = opt_ctx
                return bf16_mem_copy

            for node in sub_graph.nodes:
                _node: torch.fx.Node = node
                if _node.target in ["load", "constant"]:
                    assert len(_node.args) == 3
                    if is_bf16_mem_copy(node):
                        continue
                    ops = _node.args[0]
                    # If the node is constant, the last arg is dtype
                    load_dtype = (
                        V.graph.get_dtype(_node.args[1])
                        if _node.target == "load"
                        else _node.args[-1]
                    )

                    if load_dtype == torch.bfloat16:
                        with sub_graph.inserting_after(_node):
                            to_type_node = sub_graph.call_method(
                                "to_dtype", args=(ops, _node, torch.float)
                            )
                            to_type_node_args = to_type_node.args
                            _node.replace_all_uses_with(to_type_node)
                            to_type_node.args = to_type_node_args
                            metrics.cpp_to_dtype_count += 1
                elif _node.target == "store":
                    if is_bf16_mem_copy(_node):
                        continue
                    ops, store_var, _, value_var, _ = _node.args
                    store_dtype = V.graph.get_dtype(store_var)
                    if store_dtype == torch.bfloat16:
                        with sub_graph.inserting_before(_node):
                            to_type_node = sub_graph.call_method(
                                "to_dtype", args=(ops, value_var, torch.bfloat16)
                            )
                            _node.replace_input_with(value_var, to_type_node)
                            metrics.cpp_to_dtype_count += 1
                elif _node.target == "reduction":
                    (
                        ops,
                        name,
                        dtype,
                        src_dtype,
                        reduction_type,
                        index,
                        value,
                    ) = _node.args
                    if src_dtype == torch.bfloat16:
                        # Since we always convert the load/store value to float if the tensor is bfloat16.
                        # Therefore, the reduction should never work with bfloat16 value. Hence, we update
                        # the bfloat16 reduction by
                        #     1) updating the src_dtype to float
                        # and 2) updating the dtype to float if it is bfloat16.
                        assert dtype in [torch.float, torch.bfloat16, torch.int64]
                        _node.args = (
                            ops,
                            name,
                            torch.float if dtype == torch.bfloat16 else dtype,
                            torch.float,
                            reduction_type,
                            index,
                            value,
                        )
                elif _node.target == "to_dtype" and _node.args[-1] in [torch.bfloat16]:
                    (ops, x, _) = _node.args
                    from_load = _node.all_input_nodes[-1].target == "load"
                    to_store = all(usr.target == "store" for usr in _node.users)
                    # The legalization always loads the BF16 tensor as FP32 for computation and converts
                    # back to BF16 after the computation. Hence, there should be no computation w/ BF16.
                    # Therefore, we update the to_dtype by replacing the bf16 dtype with fp32.
                    if not (from_load or to_store):
                        _node.args = (ops, x, torch.float)
                else:
                    pass

            def eliminate_to_dtype(sub_graph: torch.fx.Graph):
                def _eliminate_duplicate_to_node(sub_graph: torch.fx.Graph):
                    # Eliminate the redundant to_dtype node. Let's consider a pattern as follows:
                    #   graph():
                    #     %to_dtype1 = call_method[target=to_dtype](args = (%ops, %input, torch.float), kwargs = {})
                    #     %to_dtype2 = call_method[target=to_dtype](args = (%ops, %to_dtype1, torch.float), kwargs = {})
                    # Regarding the first to_dtype, it is redundant because the second to_type also converts to the
                    # torch.float. Hence, we remove the first to_type
                    def _used_by_to(to_node: torch.fx.Node):
                        return all(usr.target == "to_dtype" for usr in to_node.users)

                    all_to_nodes = [
                        node for node in sub_graph.nodes if node.target == "to_dtype"
                    ]
                    all_to_nodes_and_users = [
                        {node: node.users} for node in all_to_nodes if _used_by_to(node)
                    ]
                    for node_users in all_to_nodes_and_users:
                        for node, users in node_users.items():
                            if all(usr.args[-1] == node.args[-1] for usr in users):
                                val_node = node.all_input_nodes[-1]
                                node.replace_all_uses_with(val_node)
                                sub_graph.erase_node(node)

                    # For debug mode, the graph of LoopBody will attach a new GraphModule as
                    # owning_module for debugging while the release mode will not. The lint will
                    # check whether the graph has owning_module to decide if it needs to check
                    # call_module. LoopBody might contain get_index as a module call. But it
                    # is just a function. Hence, it cannot pass the lint check for debug mode.
                    # We bypass the check if the owning_module is None. Eventually, we should call
                    # get_index via call_function but not call_module.
                    if sub_graph.owning_module is None:
                        sub_graph.lint()

                _eliminate_duplicate_to_node(sub_graph)

            eliminate_to_dtype(sub_graph)

        def _legalize_bf16(loop_body: ir.LoopBody):
            sub_blocks = [loop_body.root_block] + list(loop_body.subblocks.values())
            for sub_block in sub_blocks:
                add_to_dtype(sub_block.graph)

        for _node in nodes:
            assert isinstance(_node, SchedulerNode)
            node: SchedulerNode = _node
            if isinstance(node._body, ir.LoopBody):
                body: ir.LoopBody = node._body
                _legalize_bf16(body)

    def codegen_nodes(self, nodes):
        # Legalize BF16 node by adding to_dtype explicitly
        self.legalize_bf16(nodes)
        self.data_type_propagation(nodes)

        kernel_group = self.kernel_group
        _, (group, reduction_group) = max(
            nodes, key=lambda x: int(x.is_reduction())
        ).group

        self.set_ranges(group, reduction_group)

        def codegen_kernel(cls, *args):
            with kernel_group.new_kernel(cls, *args) as kernel:
                run(kernel)

                # Ugly hack to maintain the metrics kernel count since
                # we only count in CppKernelProxy, not those contained in it
                metrics.generated_kernel_count -= 1

                return kernel

        def run(kernel):
            vars, reduction_vars = kernel.set_ranges(group, reduction_group)
            in_suffix = False
            for node in nodes:
                if node.group[1] in [
                    (group, reduction_group),
                    (group + reduction_group, ()),
                ]:
                    assert not in_suffix
                    node.run(vars, reduction_vars)
                else:
                    in_suffix = True
                    assert node.group[1] == (
                        group,
                        (),
                    ), f"unexpected group: {node.group[1]} != {group}, {reduction_group}"
                    # we can fuse in some extra pointwise into the suffix
                    with kernel.write_to_suffix():
                        node.run(vars, ())

        scalar_kernel = codegen_kernel(CppKernel)
        self.loop_nest = LoopNestWithSplit.build(scalar_kernel)

        if not self.picked_vec_isa:
            return

        def select_tiling_indices():
            all_index = []
            for node in nodes:
                rw = dependencies.extract_read_writes(node._body, *node._sizes)
                all_index += [dep.index for dep in itertools.chain(rw.reads, rw.writes)]
            contig_vars = set()
            contig_vars_list = []
            non_contig_stride_const = set()
            non_contig_stride_other = set()
            for index in all_index:
                for var in index.free_symbols:
                    if not re.search(r"^d\d+$", var.name):
                        continue
                    stride = stride_at(var, index)
                    if stride == 1:
                        contig_vars.add(int(var.name[1:]))
                        contig_vars_list.append(int(var.name[1:]))
                    elif all(s.name.startswith("s") for s in stride.free_symbols):
                        non_contig_stride_const.add(int(var.name[1:]))
                    else:
                        non_contig_stride_other.add(int(var.name[1:]))
            contig_only = (
                contig_vars - non_contig_stride_const - non_contig_stride_other
            )
            if len(contig_vars) == 0:
                # no contiguous vars
                return [len(self.itervars) - 1]
            if contig_only:
                return sorted(contig_only)[-1:]
            contig_and_const_stride = (
                contig_vars & non_contig_stride_const
            ) - non_contig_stride_other
            contig_vars_sorted = sorted(contig_vars)
            if (
                len(contig_vars_sorted) == 2
                and contig_vars_sorted[-1] in contig_and_const_stride
                and contig_vars_sorted[-1] == len(self.itervars) - 1
            ):
                return contig_vars_sorted
            return sorted(contig_vars_sorted, key=lambda i: contig_vars_list.count(i))[
                -1:
            ]

        def select_tiling():
            # TODO(jgong5): support alternative tiling factors and data types
            tiling_factor = self.picked_vec_isa.nelements(dtype=torch.float)
            tiling_indices = select_tiling_indices()
            if tiling_indices:
                with CppVecKernelChecker(
                    deepcopy(self.kernel_group.args),
                    parallel_num_threads(),
                    tiling_factor,
                    tiling_indices[-1],
                ) as vec_checker:
                    run(vec_checker)
                if vec_checker.simd_vec:
                    if len(tiling_indices) == 1:
                        return [tiling_factor], tiling_indices
                    if len(tiling_indices) == 2:
                        return [tiling_factor, tiling_factor], tiling_indices
            return [], []

        # Kernels share the same global contexts like V.graph.wrapper_code, V.kernel.args.
        # But the generated scalar kernel has updated these global contexts. Hence, the other kernels
        # should not do this again to avoid context conflict. By now, we only control the
        # config.inplace_buffers. In the future, we could maintain more contexts.
        with torch._inductor.config.patch(inplace_buffers=False):
            tiling_factors, tiling_indices = select_tiling()
            assert len(tiling_factors) == len(tiling_indices)
            if len(tiling_indices) == 1:
                main_loop, tail_loop = self.loop_nest.split_with_tiling(
                    tiling_indices[0], factor=tiling_factors[0]
                )
                main_loop.set_kernel(
                    codegen_kernel(CppVecKernel, tiling_factors[0], tiling_indices[0])
                )
                tail_loop.set_kernel(scalar_kernel)
                main_loop.simd_vec = True
                tail_loop.simd_omp = True
                # We chop the loop into two cubes by the nelements - main loop and tail loop.
                # Regarding the main loop, it is straightforward that it could be vectorized with
                # nelements. But for the tail loop, it still could be vectorized. For example,
                # if the nelements is 8(256bits), then the tail loop still could be vectorized
                # as 4(128bits).
                tail_loop.simd_nelements = tiling_factors[0] // 2
            elif len(tiling_indices) == 2:
                assert (
                    tiling_indices[1] == len(self.itervars) - 1
                    and tiling_factors[0] == tiling_factors[1]
                )
                outer_main_loop, outer_tail_loop = self.loop_nest.split_with_tiling(
                    tiling_indices[0], factor=tiling_factors[0]
                )
                outer_tail_loop.set_kernel(scalar_kernel)
                inner_main_loop, inner_tail_loop = outer_main_loop.split_with_tiling(
                    tiling_indices[1] - tiling_indices[0], factor=tiling_factors[0]
                )
                inner_main_loop.set_kernel(
                    codegen_kernel(CppTile2DKernel, tiling_factors[0], tiling_indices)
                )
                inner_tail_loop.set_kernel(
                    codegen_kernel(CppVecKernel, tiling_factors[0], tiling_indices[0])
                )

    def codegen_loops(self, code, worksharing):
        self.codegen_loops_impl(self.loop_nest, code, worksharing)


class CppScheduling:
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.get_kernel_group()

    def group_fn(self, sizes):
        return tuple(tuple(map(V.graph.sizevars.simplify, s)) for s in sizes)

    def get_kernel_group(self):
        from .wrapper import CppWrapperCodeGen

        if isinstance(V.graph.wrapper_code, CppWrapperCodeGen):
            self.kernel_group = CppWrapperKernelGroup()
        else:
            self.kernel_group = KernelGroup()

    @staticmethod
    def can_fuse_horizontal(node1, node2):
        _, (vars1, reduce1) = node1.group
        _, (vars2, reduce2) = node2.group
        if vars1 == vars2 and reduce1 == reduce2:
            return True
        if reduce1 == () and vars1 == vars2 + reduce2:
            return True
        # TODO(jansel): allow fusion pointwise (vars1, ()) suffix?
        return False

    @classmethod
    def can_fuse_vertical(cls, node1, node2):
        return cls.can_fuse_horizontal(node1, node2) and not node1.is_reduction()

    def codegen_nodes(self, nodes):
        """
        Turn an set of pre-fused nodes into a C++ kernel.
        """
        kernel_group = self.kernel_group

        cpp_kernel_proxy = CppKernelProxy(kernel_group)
        cpp_kernel_proxy.codegen_nodes(nodes)

        kernel_group.finalize_kernel(cpp_kernel_proxy, nodes)

    def codegen_sync(self):
        pass

    def flush(self):
        self.kernel_group.codegen_define_and_call(V.graph.wrapper_code)
        self.get_kernel_group()


class KernelGroup:
    def __init__(self):
        super().__init__()
        self.args = KernelArgs()
        self.loops_code = BracesBuffer()
        self.ws = WorkSharing(self.loops_code)
        self.stack = contextlib.ExitStack()
        self.stack.enter_context(self.ws)
        self.scheduled_nodes = []

    def new_kernel(self, cls, *args):
        return cls(self.args, parallel_num_threads(), *args)

    def finalize_kernel(self, new_kernel, nodes):
        self.scheduled_nodes += nodes
        code = self.loops_code
        ws = self.ws
        new_kernel.codegen_loops(code, ws)

    def codegen_define_and_call(self, wrapper):
        self.stack.close()
        if not self.scheduled_nodes:
            return

        fused_name = (
            get_fused_kernel_name(self.scheduled_nodes, config.cpp.descriptive_names)
            if config.cpp.descriptive_names
            else ""
        )
        kernel_name = "_".join(["cpp", fused_name, wrapper.next_kernel_suffix()])
        arg_defs, call_args, arg_types = self.args.cpp_argdefs()
        arg_defs = ",\n".ljust(25).join(arg_defs)
        arg_types = ",".join(arg_types)
        code = BracesBuffer()
        # TODO: support kernel profile on other platforms
        enable_kernel_profile = (
            config.cpp.enable_kernel_profile and sys.platform == "linux"
        )
        if enable_kernel_profile:
            code.writelines(["#include <ATen/record_function.h>"])
        kernel_decl_name = kernel_name if V.graph.cpp_wrapper else "kernel"
        code.writeline(cpp_prefix())

        code.writeline(f'extern "C" void {kernel_decl_name}({arg_defs})')
        with code.indent():
            if enable_kernel_profile:
                graph_id = V.graph.graph_id
                prefix = "graph_" + str(graph_id) + "_" if graph_id is not None else ""
                code.writelines(
                    [
                        f'RECORD_FUNCTION("{prefix + kernel_name}", c10::ArrayRef<c10::IValue>({{}}));'
                    ]
                )
            for old, new in self.args.aliases():
                code.writeline(f"auto {old} = {new};")
            code.splice(self.loops_code)

        codecache_def = IndentedBuffer()
        if not V.graph.cpp_wrapper:
            codecache_def.writeline("async_compile.cpp('''")
        codecache_def.splice(code)
        if not V.graph.cpp_wrapper:
            codecache_def.writeline("''')")

        codecache_str = codecache_def.getvalue()
        # TODO(voz): Ostensibly, we should not need this. But there are cases where C++ codegen does
        # not use BracesBuffer, so we have no good indicator of a C++ buffer atm.
        codecache_str = codecache_str.replace("#pragma CMT", "//")
        wrapper.define_kernel(kernel_name, codecache_str, cpp=True)
        # generate the code to call this
        wrapper.generate_kernel_call(kernel_name, call_args, cpp=True)


class CppWrapperKernelGroup(KernelGroup):
    def __init__(self):
        super().__init__()
        self.args = CppWrapperKernelArgs()


class WorkSharing:
    def __init__(self, code):
        self.code = code
        self.in_parallel = False
        self.num_threads = None
        self.stack = contextlib.ExitStack()

    def parallel(self, threads):
        if self.in_parallel and threads != self.num_threads:
            # wrong number of threads
            self.close()
        if not self.in_parallel:
            self.num_threads = threads
            self.in_parallel = True
            if config.cpp.dynamic_threads:
                self.code.writeline("#pragma omp parallel")
            else:
                self.code.writeline(f"#pragma omp parallel num_threads({threads})")
            self.stack.enter_context(self.code.indent())

    def single(self):
        if self.in_parallel:
            self.code.writeline("#pragma omp single")
        return self.in_parallel

    def close(self):
        self.stack.close()
        self.in_parallel = False

    def __enter__(self):
        self.stack.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stack.__exit__(exc_type, exc_val, exc_tb)


@dataclasses.dataclass
class LoopLevel:
    var: sympy.Expr = None
    size: sympy.Expr = None
    offset: sympy.Expr = sympy.Integer(0)
    steps: sympy.Expr = sympy.Integer(1)
    parallel: int = 0
    simd_omp: bool = False
    picked_vec_isa: codecache.VecISA = codecache.pick_vec_isa()
    simd_nelements: int = picked_vec_isa.nelements() if picked_vec_isa else 0
    simd_vec: bool = False
    collapsed: bool = False
    reduction_var_map: Dict[str, str] = None
    parent: "LoopLevel" = None
    # the next inner level of the loop, empty if it is inner-most
    # contains >1 LoopLevel if the inner level of loop is split
    inner: List["LoopLevel"] = dataclasses.field(default_factory=list)
    # kernel assigned to this loop level, only valid when it is a leaf
    kernel: CppKernel = None

    def get_kernels(self) -> List[CppKernel]:
        """Get all kernel objects under this loop level"""
        if self.kernel:
            return [self.kernel]
        kernels = []
        for loop in self.inner:
            kernels += loop.get_kernels()
        return kernels

    def set_kernel(self, kernel: CppKernel):
        """
        Set the kernel under this loop level. No split is allowed under
        this loop level.
        """
        if not self.inner:
            self.kernel = kernel
            loop = self
            if loop.is_reduction():
                loop.reduction_var_map = kernel.reduction_var_map.copy()
                loop = loop.parent
                while loop is not None and loop.is_reduction():
                    loop.reduction_var_map.update(kernel.reduction_var_map)
                    loop = loop.parent
            return
        assert len(self.inner) == 1
        self.inner[0].set_kernel(kernel)

    def get_loops_at(self, depth) -> List["LoopLevel"]:
        if depth == 0:
            return [self]
        else:
            loops = []
            for loop in self.inner:
                loops += loop.get_loops_at(depth - 1)
            return loops

    def is_reduction(self):
        return bool(self.reduction_var_map)

    def split_with_tiling(self, depth, factor):
        def clone_inner():
            inner = []
            if self.inner:
                for loop in self.inner:
                    inner.append(loop.clone())
            return inner

        def do_split_with_tiling():
            sympy_factor = sympy.Integer(factor)

            offset = ir.FloorDiv(self.size, sympy_factor) * sympy_factor
            main_loop = LoopLevel(self.var, offset)
            main_loop.steps = sympy_factor
            main_loop.parallel = self.parallel
            main_loop.collapsed = False
            main_loop.reduction_var_map = self.reduction_var_map
            main_loop.inner = clone_inner()
            if main_loop.inner:
                for loop in main_loop.inner:
                    loop.parent = main_loop

            tail_loop = LoopLevel(self.var, self.size)
            tail_loop.offset = offset
            tail_loop.parallel = self.parallel
            tail_loop.collapsed = False
            tail_loop.reduction_var_map = self.reduction_var_map
            tail_loop.inner = clone_inner()
            if tail_loop.inner:
                for loop in tail_loop.inner:
                    loop.parent = tail_loop

            return main_loop, tail_loop

        if depth == 0:
            main_loop, tail_loop = do_split_with_tiling()
            parent = self.parent
            if parent:
                parent.inner = [main_loop, tail_loop]
                main_loop.parent = parent
                tail_loop.parent = parent
            return main_loop, tail_loop
        else:
            assert len(self.inner) == 1
            return self.inner[0].split_with_tiling(depth - 1, factor)

    def clone(self):
        loop = copy(self)
        loop.inner = []
        if self.inner:
            for inner_loop in self.inner:
                inner_loop_clone = inner_loop.clone()
                inner_loop_clone.parent = loop
                loop.inner.append(inner_loop_clone)
        loop.kernel = deepcopy(self.kernel)
        return loop

    def lines(self):
        offset_expr = cexpr_index(self.offset)
        size_expr = cexpr_index(self.size)
        if config.cpp.no_redundant_loops and offset_expr == size_expr:
            return None
        if self.reduction_var_map:
            reduction = " " + " ".join(
                f"reduction({RTYPE_TO_CPP[rtype]}:{var})"
                for var, rtype in self.reduction_var_map.items()
            )
        else:
            reduction = ""
        simd = (
            f"simd simdlen({self.simd_nelements}) "
            if self.simd_omp and self.simd_nelements > 1
            else ""
        )
        if self.parallel:
            # TODO(jansel): look into chunk size and other schedules
            line1 = f"#pragma omp for{reduction} "
            if self.parallel > 1:
                line1 += f" collapse({self.parallel})"
            if self.simd_omp:
                line1 = line1.replace(" for ", f" for {simd}")
        elif self.simd_vec:
            line1 = ""
        elif self.simd_omp:
            line1 = f"#pragma omp {simd}{reduction}"
        elif not self.reduction_var_map and codecache.is_gcc():
            line1 = "#pragma GCC ivdep"
        else:
            line1 = ""
        offset_str = f"{INDEX_TYPE} {self.var}={offset_expr}"
        size_str = f"{self.var}<{size_expr}"
        steps_str = f"{self.var}+={cexpr_index(self.steps)}"
        line2 = f"for({offset_str}; {size_str}; {steps_str})"
        if self.collapsed or not line1:
            return [line2]
        return [line1, line2]


@dataclasses.dataclass
class LoopNestWithSplit:
    """
    A loop-nest like structure but with some loop level split along
    the loop range into the main tiling loop and the tail. It is built
    with the `build` method as a loop nest and then split with
    `split_with_tiling` at some depth.

    A typical case is for vectorization where we typically split at the inner-most
    loop level. A more complicated case is 2D tiling where we split at
    both inner-most and outer levels.
    """

    root: List[LoopLevel] = None
    kernel: CppKernel = None

    @staticmethod
    def build(kernel: CppKernel):
        """Build a LoopNest with the given `kernel` as the leaf"""
        itervars = kernel.itervars
        ranges = kernel.ranges
        reduction_depth = kernel.reduction_depth

        root: List[LoopLevel] = []
        levels: List[LoopLevel] = root
        loop: LoopLevel = None
        for loop_idx, (var, size) in enumerate(zip(itervars, ranges)):
            loop = LoopLevel(var, size, parent=loop)
            if loop_idx >= reduction_depth:
                loop.reduction_var_map = kernel.reduction_var_map.copy()
            levels.append(loop)
            levels = loop.inner
        loop_nest = LoopNestWithSplit(root, len(itervars))
        if loop:
            loop.kernel = kernel
        else:
            loop_nest.kernel = kernel
        return loop_nest

    def __bool__(self):
        return bool(self.root)

    def get_loops_at(self, depth) -> List[LoopLevel]:
        """Get all the loop levels at the given `depth` (most outer loop has depth 0)"""
        loops = []
        for loop in self.root:
            loops += loop.get_loops_at(depth)
        return loops

    @cache_on_self
    def max_parallel_depth(self):
        """
        Maximal allowed depth for parallelism:
        1) Levels without splitting and
        2) All reduction or non-reduction levels
        When the loop is split at the top level, the max depth is 1.
        """
        max_depth = 0
        loops = self.root
        if len(loops) > 1:
            return 1
        is_reduction = loops[0].is_reduction() if loops else False
        while len(loops) == 1 and loops[0].is_reduction() == is_reduction:
            max_depth += 1
            loops = loops[0].inner
        return max_depth

    def is_reduction_only(self):
        """
        Whether all the loops are for reduction. Reduction loops
        are always the inner most ones.
        """
        return self.root and self.root[0].is_reduction()

    def mark_parallel(self, par_depth):
        assert (
            par_depth <= self.max_parallel_depth()
        ), "Parallel depth cannot exceed the maximal allowed parallel depth"
        loops = self.root
        for loop in loops:
            loop.parallel = par_depth
        for i in range(1, par_depth):
            loops = loops[0].inner
            loops[0].collapsed = True

    def split_with_tiling(self, depth, factor):
        """
        Split the loop into main and tail loops at given `depth` so that the range
        of the main loop has range `floor_div(range, factor) * factor` and
        the tail loop handles the remainder. The main loop is tiled
        according to the `factor`.
        """
        loops = self.get_loops_at(depth)
        assert len(loops) == 1
        split_loops = loops[0].split_with_tiling(0, factor)
        if depth == 0:
            self.root = split_loops
        return split_loops
