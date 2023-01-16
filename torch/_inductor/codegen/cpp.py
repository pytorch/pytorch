import contextlib
import dataclasses
import functools
import math
import sys
from copy import copy, deepcopy
from pathlib import Path
from typing import Dict, List
from unittest.mock import patch

import sympy

import torch
from torch._prims_common import is_float_dtype

from .. import codecache, config, ir, metrics
from ..codegen.wrapper import WrapperCodeGen
from ..utils import cache_on_self, sympy_product, sympy_subs, sympy_symbol
from ..virtualized import ops, V
from .common import (
    BracesBuffer,
    CppWrapperKernelArgs,
    DeferredIndentedBuffer,
    ExprPrinter,
    IndentedBuffer,
    Kernel,
    KernelArgs,
    OpOverrides,
)

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
    torch.float32: "at::ScalarType::Float",
    torch.float64: "at::ScalarType::Double",
    torch.float16: "at::ScalarType::Half",
    torch.int64: "at::ScalarType::Long",
    torch.int32: "at::ScalarType::Int",
    torch.int16: "at::ScalarType::Short",
    torch.int8: "at::ScalarType::Char",
    torch.uint8: "at::ScalarType::Byte",
    torch.bool: "at::ScalarType::Bool",
    torch.bfloat16: "at::ScalarType::BFloat16",
}

INDEX_TYPE = "long"

RTYPE_TO_CPP = {
    "sum": "+",
    "min": "min",
    "max": "max",
    "argmin": "argmin",
    "argmax": "argmax",
    "any": "||",
}


def reduction_init(reduction_type, dtype):
    if reduction_type in ("sum", "any"):
        return 0
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
    if reduction_type == "any":
        return f"{var} = {var} || {next_value}"
    return f"{var} = std::{reduction_type}({var}, {next_value})"


def reduction_combine_vec(reduction_type, var, next_value):
    if reduction_type == "max":
        return f"{var} = at::vec::maximum({var}, {next_value})"
    elif reduction_type == "min":
        return f"{var} = at::vec::minimum({var}, {next_value})"
    elif reduction_type == "sum":
        return f"{var} += {next_value}"
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


def float16_reduction_prefix(rtype):
    # TODO: This user-defined reduction uses float16 accumulation for sum. To reduce numerical
    # errors, float32 accumulation should be used instead.
    assert rtype in (
        "sum",
        "any",
    ), f"float16 user-defined reduction only supports 'sum' and 'any' but got {rtype}"
    prefix = [
        f"#pragma omp declare reduction({RTYPE_TO_CPP[rtype]}:{DTYPE_TO_CPP[torch.float16]}:"
        + f"omp_out = omp_out {RTYPE_TO_CPP[rtype]} omp_in)"
    ]
    return prefix


def parallel_num_threads():
    threads = config.cpp.threads
    if threads < 1:
        threads = torch.get_num_threads()
    return threads


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
    def _print_ModularIndexing(self, expr):
        x, div, mod = expr.args
        x = self.paren(self.doprint(x))
        div = self.paren(self.doprint(div))
        mod = self.paren(self.doprint(mod))
        if div != "1":
            x = f"({x} / {div})"
        return f"{x} % {mod}"

    def _print_IndexingDiv(self, expr):
        x, div = expr.args
        x = self.paren(self.doprint(x))
        div = self.paren(self.doprint(div))
        return f"({x} / {div})"


cexpr = CppPrinter().doprint


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
    def erf(x):
        return f"{x}.erf()"

    @staticmethod
    def sqrt(x):
        return f"{x}.sqrt()"

    @staticmethod
    def eq(x, y):
        return f"{x} == {y}"

    @staticmethod
    def ne(x, y):
        return f"{x} != {y}"

    @staticmethod
    def lt(x, y):
        return f"{x} < {y}"

    @staticmethod
    def gt(x, y):
        return f"{x} > {y}"

    @staticmethod
    def le(x, y):
        return f"{x} <= {y}"

    @staticmethod
    def ge(x, y):
        return f"{x} >= {y}"

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
        return f"{a} && {b}"

    @staticmethod
    def logical_or(a, b):
        return f"{a} || {b}"

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
    def constant(val, dtype):
        if val == float("inf"):
            quote = f"std::numeric_limits<{DTYPE_TO_CPP[dtype]}>::infinity()"
        elif val == float("-inf"):
            quote = f"-std::numeric_limits<{DTYPE_TO_CPP[dtype]}>::infinity()"
        elif math.isnan(val):
            quote = f"std::numeric_limits<{DTYPE_TO_CPP[dtype]}>::quiet_NaN()"
        elif val is True or val is False:
            quote = f"static_cast<{DTYPE_TO_CPP[dtype]}>({str(val).lower()})"
        else:
            quote = f"static_cast<{DTYPE_TO_CPP[dtype]}>({repr(val)})"
        return f"at::vec::Vectorized<{DTYPE_TO_CPP[dtype]}>({quote})"

    @staticmethod
    def relu(x):
        return f"at::vec::clamp_min({x}, decltype({x})(0))"

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
        return f"{a}.pow(2)"

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
        assert dtype in [torch.bool], f"{__name__} does not support {dtype}"
        return f"({x})"

    @staticmethod
    def expm1(x):
        return f"{x}.expm1()"

    @staticmethod
    def log1p(x):
        return f"{x}.log1p()"


class CppOverrides(OpOverrides):
    """Map element-wise ops to C++"""

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
    def exp(x):
        # return f"Sleef_expf_u10({x})"
        return f"std::exp({x})"

    @staticmethod
    def erf(x):
        return f"std::erf({x})"

    @staticmethod
    def sqrt(x):
        return f"std::sqrt({x})"

    @staticmethod
    def rsqrt(x):
        return f"1 / std::sqrt({x})"

    @staticmethod
    def log1p(x):
        return f"std::log1p({x})"

    @staticmethod
    def expm1(x):
        return f"std::expm1({x})"

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
    def relu(x):
        return f"{x} * ({x}>0)"

    @staticmethod
    def minimum(a, b):
        return f"({b} != {b}) ? {b} : std::min({a}, {b})"

    @staticmethod
    def maximum(a, b):
        return f"({b} != {b}) ? {b} : std::max({a}, {b})"

    @staticmethod
    def where(a, b, c):
        return f"{a} ? {b} : {c}"

    @staticmethod
    def mod(a, b):
        return f"mod({a}, {b})"

    @staticmethod
    def constant(val, dtype):
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
        var = V.kernel.cse.newvar()
        if other == float("-inf"):
            code.writeline(f"float {var} = -std::numeric_limits<float>::infinity();")
        elif other == float("inf"):
            code.writeline(f"float {var} = std::numeric_limits<float>::infinity();")
        elif isinstance(other, bool):
            if other:
                code.writeline(f"auto {var} = true;")
            else:
                code.writeline(f"auto {var} = false;")
        elif isinstance(other, float):
            code.writeline(f"float {var} = {other};")
        else:
            code.writeline(f"auto {var} = {other!r};")
        code.writeline(f"if({mask})")
        with V.kernel.swap_buffers(code), code.indent():
            result = body()
            code.writeline(f"{var} = {result};")
        V.kernel.compute.splice(code)
        return var

    @staticmethod
    def logical_and(a, b):
        return f"{a} && {b}"

    @staticmethod
    def logical_or(a, b):
        return f"{a} || {b}"

    @staticmethod
    def rand(seed: sympy.Expr, offset: sympy.Expr, dtype):
        return f"static_cast<{DTYPE_TO_CPP[dtype]}>(normalized_rand_cpu({seed}, {offset}));"

    @staticmethod
    def randn(seed: sympy.Expr, offset: sympy.Expr, dtype):
        return f"static_cast<{DTYPE_TO_CPP[dtype]}>(randn_cpu({seed}, {offset}));"

    @staticmethod
    def sigmoid(x):
        x = ops.exp(f"-{x}")
        return f"1 / (1 + {x})"

    @staticmethod
    def sign(x):
        code = BracesBuffer()
        # auto tmp5 = tmp4 < 0 ? -1 : 1;
        left = V.kernel.cse.newvar()
        right = V.kernel.cse.newvar()
        result = V.kernel.cse.newvar()
        code.writeline(f"auto {left} = {x} > 0 ? 1 : 0;")
        code.writeline(f"auto {right} = {x} < 0 ? 1 : 0;")
        code.writeline(f"auto {result} = {left} - {right};")
        V.kernel.compute.splice(code)
        return result


class CppKernel(Kernel):
    overrides = CppOverrides
    sexpr = cexpr
    newvar_prefix = "auto "
    suffix = ";"

    def __init__(self, args, num_threads):
        super(CppKernel, self).__init__(args)
        self.call_ranges = None
        self.ranges = None
        self.itervars = None
        self.reduction_depth = None
        self.reduction_prefix = IndentedBuffer()
        self.reduction_suffix = DeferredIndentedBuffer()
        self.reduction_var_map = {}
        self.preloads = IndentedBuffer()
        self.poststores = DeferredIndentedBuffer()
        self.num_threads = num_threads  # num_threads the kernel specialized for

    def scale_index_with_offset(
        self, index: sympy.Expr, scale, itervar_idx=-1, offset=0
    ):
        expanded_index = sympy.expand(index)
        var = self.itervars[itervar_idx]
        replacement = {var: var * scale + offset}
        new_index = sympy_subs(expanded_index, replacement)
        return new_index

    def load(self, name: str, index: sympy.Expr):
        var = self.args.input(name)
        index = self.rename_indexing(index)
        line = f"{var}[{cexpr(index)}]"
        if V.graph.get_dtype(name) in (torch.float16, torch.bfloat16):
            line = f"static_cast<float>({line})"
        return self.cse.generate(self.loads, line)

    def store(self, name, index, value, mode=None):
        assert "buf" in name
        var = self.args.output(name)
        index = self.rename_indexing(index)
        if mode is None:
            line = f"{var}[{cexpr(index)}] = {value};"
        elif mode == "atomic_add":
            if not config.cpp.dynamic_threads and self.num_threads == 1:
                line = f"{var}[{cexpr(index)}] += {value};"
            else:
                line = f"atomic_add(&{var}[{cexpr(index)}], {value});"
        else:
            raise NotImplementedError(f"store mode={mode}")
        self.stores.writeline(name, line)

    def reduction(self, name, dtype, src_dtype, reduction_type, index, value):
        argmax_or_argmin = reduction_type in {"argmax", "argmin"}
        tmpvar = self.cse.generate(
            self.loads, f"reduction {name} {cexpr(index)}", write=False
        )
        index = self.rename_indexing(index)
        self.reduction_var_map[tmpvar] = reduction_type
        if argmax_or_argmin:
            self.reduction_prefix.writelines(
                argmax_argmin_prefix(reduction_type, src_dtype, tmpvar)
            )
            compare_op = "<" if reduction_type == "argmax" else ">"
            self.stores.writelines(
                None,
                [
                    f"if ({tmpvar}.value {compare_op} {value}) {{",
                    f"    {tmpvar}.index = {self.itervars[-1]}; {tmpvar}.value = {value};",
                    "}",
                ],
            )
        else:
            if dtype == torch.float16:
                self.reduction_prefix.writelines(
                    float16_reduction_prefix(reduction_type)
                )
            self.reduction_prefix.writeline(
                f"{DTYPE_TO_CPP[dtype]} {tmpvar} = {reduction_init(reduction_type, dtype)};"
            )
            self.stores.writeline(
                None, f"{reduction_combine(reduction_type, tmpvar, value)};"
            )

        if name not in V.graph.removed_buffers:
            var = self.args.output(name)
            member_name = ".index" if argmax_or_argmin else ""
            self.reduction_suffix.writeline(
                name, f"{var}[{cexpr(index)}] = {tmpvar}{member_name};"
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

            def gen_loops(loops: List[LoopLevel], in_reduction=False):
                with contextlib.ExitStack() as stack_outer:
                    if loops:
                        loop = loops[0]
                        if loop.is_reduction() and not in_reduction:
                            kernels = loop.get_kernels()
                            assert kernels
                            # TODO(jgong5): should gen prefix for all kernels.
                            # currently, Vec kernel generates prefix for both
                            # vector and scalar kernels.
                            if kernels[0].reduction_prefix:
                                stack_outer.enter_context(code.indent())
                            code.splice(kernels[0].reduction_prefix)
                        if loop_nest.is_reduction_only() and loop.parallel:
                            worksharing.parallel(threads)

                    for loop in loops:
                        gen_loop(loop, in_reduction)

                    if loops:
                        if loop_nest.is_reduction_only() and loop.parallel:
                            worksharing.close()
                        for loop in loops:
                            if loop.is_reduction() and not in_reduction:
                                kernels = loop.get_kernels()
                                for kernel in kernels:
                                    code.splice(kernel.reduction_suffix)

            def gen_loop(loop: LoopLevel, in_reduction=False):
                with contextlib.ExitStack() as stack:
                    code.writelines(loop.lines())
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
        self.stores = DeferredIndentedBuffer()
        self.cse = self.cse.clone()
        yield
        self.reduction_suffix.splice(self.loads)
        self.reduction_suffix.splice(self.compute)
        self.reduction_suffix.splice(self.stores)
        (self.loads, self.compute, self.stores, self.cse) = prior


class CppVecKernel(CppKernel):
    overrides = CppVecOverrides

    def __init__(self, args, num_threads, tiling_factor=0):
        super(CppVecKernel, self).__init__(args, num_threads)
        assert codecache.pick_vec_isa()
        if tiling_factor == 0:
            tiling_factor = codecache.pick_vec_isa().nelements()
        self.tiling_factor = tiling_factor
        self.reduction_omp_dec: Dict[str, str] = {}
        self.var_vec_buf_map: Dict[str, str] = {}
        metrics.generated_cpp_vec_kernel_count += 1

    def stride_at(self, var: sympy.Symbol, index: sympy.Expr):
        replacement = {var: var + 1}
        new_index = sympy_subs(index, replacement)
        return sympy.simplify(new_index - index)

    def is_stride1_at(self, var: sympy.Symbol, index: sympy.Expr):
        return self.stride_at(var, index) == 1

    def is_invariant_under(self, var: sympy.Symbol, index: sympy.Expr):
        expanded_index = sympy.expand(index)
        return not expanded_index.has(var)

    def load(self, name: str, index: sympy.Expr):
        var = self.args.input(name)
        index = self.rename_indexing(index)

        expanded_index = sympy.expand(index)
        new_index = self.scale_index_with_offset(index, self.tiling_factor)

        if expanded_index == new_index:
            line = f"at::vec::Vectorized<float>({var}[{cexpr(index)}])"
        else:
            if V.graph.get_dtype(name) in [torch.bool, torch.uint8]:
                nelements = codecache.pick_vec_isa().nelements()
                if var not in self.var_vec_buf_map:
                    self.var_vec_buf_map[var] = f"g_tmp_buffer_{var}"
                    self.loads.writeline(
                        f"float {self.var_vec_buf_map[var]}[{nelements}] = {{0}};"
                    )
                self.loads.writeline(
                    f"flag_to_float({var} + {cexpr(new_index)}, {self.var_vec_buf_map[var]}, {nelements});"
                )
                line = f"at::vec::Vectorized<float>::loadu({self.var_vec_buf_map[var]})"
            else:
                line = f"at::vec::Vectorized<float>::loadu({var} + {cexpr(new_index)})"

        return self.cse.generate(self.loads, line)

    def store(self, name, index, value, mode=None):
        assert "buf" in name
        var = self.args.output(name)
        index = self.rename_indexing(index)
        assert mode is None

        expanded_index = sympy.expand(index)
        new_index = self.scale_index_with_offset(index, self.tiling_factor)
        assert new_index != expanded_index
        line = f"{value}.store({var} + {cexpr(new_index)});"
        self.stores.writeline(name, line)

    def reduction(self, name, dtype, src_dtype, reduction_type, index, value):
        assert reduction_type in {"max", "min", "sum"}
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

        tmpvar = self.cse.generate(
            self.loads, f"reduction {name} {cexpr(index)}", write=False
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
            None, f"{reduction_combine_vec(reduction_type, tmpvar_vec, value)};"
        )

        reduce_all_body = "{"
        if reduction_type == "sum":
            reduce_all_body += "return x + y;"
        else:
            reduce_all_body += f"return {vec_ns}::{reduce_map[reduction_type]}(x, y);"
        reduce_all_body += "}"
        vec_reduce_all_func = f"{vec_ns}::vec_reduce_all<{DTYPE_TO_CPP[dtype]}>"
        next_value = f"{vec_reduce_all_func}([]({vec}& x, {vec}&y) {reduce_all_body}, {tmpvar_vec})"
        self.reduction_suffix.writeline(
            name,
            f"{reduction_combine(reduction_type, tmpvar, next_value)};",
        )
        # NOTE(jgong5): we do not generate the real stores here with the assumption that
        # the scalar kernel that handles the loop tail would be generated and generates
        # the stores there.
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
            // generated by CppTile2DTailKernel
            ...
      for i_outer ... (tail)
        for ...
          for ...
            // generated by CppKernel
            ...
    """

    def __init__(self, args, num_threads, tiling_factor, outer_tiling_idx):
        super().__init__(args, num_threads, tiling_factor)
        self.outer_tiling_idx = outer_tiling_idx

    def inner_itervar(self):
        return sympy.symbols(f"{self.itervars[self.outer_tiling_idx]}_inner")

    def need_vec_transpose(self, index):
        return self.is_stride1_at(
            self.itervars[self.outer_tiling_idx], index
        ) and not self.is_invariant_under(self.itervars[-1], index)

    def gen_transposed_tile_load_store(self, name, var, index, is_store):
        # transposed tile load/store outside the kernel inner loop
        factor = self.tiling_factor
        new_index = self.scale_index_with_offset(index, factor, itervar_idx=-1)
        new_index = self.scale_index_with_offset(
            new_index, factor, itervar_idx=self.outer_tiling_idx
        )

        src = f"{var} + {cexpr(new_index)}"
        dst = "__place_holder__"
        ld_src = f"{cexpr(self.stride_at(self.itervars[-1], index))}"
        ld_dst = f"{factor}"
        if is_store:
            src, dst = dst, src
            ld_src, ld_dst = ld_dst, ld_src

        need_define = True
        load_or_store = f"at::vec::transpose_mxn<float,{factor},{factor}>({src}, {ld_src}, {dst}, {ld_dst});"
        if is_store:
            tile_var = self.cse.newvar()
        elif load_or_store not in self.cse.cache:
            tile_var = self.cse.generate(self.preloads, load_or_store, write=False)
        else:
            need_define = False
            tile_var = self.cse.cache[load_or_store]

        if need_define:
            define_line = f"float {tile_var}[{factor}*{factor}] __attribute__ ((aligned ({factor})));"
            self.preloads.writeline(define_line)

        load_or_store = load_or_store.replace("__place_holder__", str(tile_var))
        if is_store:
            self.poststores.writeline(name, load_or_store)
        else:
            self.preloads.writeline(load_or_store)

        return tile_var

    def load(self, name: str, index: sympy.Expr):
        var = self.args.input(name)
        index = self.rename_indexing(index)

        inner = self.inner_itervar()
        expanded_index = sympy.expand(index)
        if self.need_vec_transpose(expanded_index):
            tile_var = self.gen_transposed_tile_load_store(
                name, var, expanded_index, is_store=False
            )
            # vector load inside the kernel inner loop
            line = f"at::vec::Vectorized<float>::loadu({tile_var} + {cexpr(inner * self.tiling_factor)})"
            return self.cse.generate(self.loads, line)
        else:
            new_index = self.scale_index_with_offset(
                expanded_index,
                self.tiling_factor,
                itervar_idx=self.outer_tiling_idx,
                offset=inner,
            )
            return super().load(name, new_index)

    def store(self, name, index, value, mode=None):
        assert "buf" in name
        var = self.args.output(name)

        inner = self.inner_itervar()
        index = self.rename_indexing(index)
        assert mode is None
        # TODO(jgong5): assert the index is an affine expression on the itervars in concern
        expanded_index = sympy.expand(index)
        if self.need_vec_transpose(expanded_index):
            tile_var = self.gen_transposed_tile_load_store(
                name, var, expanded_index, is_store=True
            )
            # vector store inside the kernel inner loop
            line = f"{value}.store({tile_var} + {cexpr(inner * self.tiling_factor)});"
            self.stores.writeline(name, line)
        else:
            new_index = self.scale_index_with_offset(
                expanded_index,
                self.tiling_factor,
                itervar_idx=self.outer_tiling_idx,
                offset=inner,
            )
            super().store(name, new_index, value, mode)

    def codegen_inner_loops(self, code):
        inner = self.inner_itervar()
        code.writeline(
            f"for (long {inner} = 0; {inner} < {self.tiling_factor}; {inner}++)"
        )


class CppTile2DTailKernel(CppKernel):
    """
    A scalar kernel that handles the tail of inner-most loop split from a 2d tiling. The tile of the outer
    loop axis is handled with a kernel inner loop (see method `codegen_inner_loops`).
    """

    def __init__(self, args, num_threads, tiling_factor, outer_tiling_idx):
        super().__init__(args, num_threads)
        self.outer_tiling_idx = outer_tiling_idx
        self.tiling_factor = tiling_factor

    def inner_itervar(self):
        return sympy.symbols(f"{self.itervars[self.outer_tiling_idx]}_inner")

    def transform_index(self, index):
        index = self.rename_indexing(index)
        expanded_index = sympy.expand(index)
        new_index = self.scale_index_with_offset(
            expanded_index,
            self.tiling_factor,
            itervar_idx=self.outer_tiling_idx,
            offset=self.inner_itervar(),
        )
        return new_index

    def load(self, name: str, index: sympy.Expr):
        new_index = self.transform_index(index)
        return super().load(name, new_index)

    def store(self, name, index, value, mode=None):
        assert "buf" in name
        var = self.args.output(name)
        assert mode is None
        new_index = self.transform_index(index)
        super().store(name, new_index, value, mode)

    def codegen_inner_loops(self, code):
        inner = self.inner_itervar()
        code.writeline(
            f"for (long {inner} = 0; {inner} < {self.tiling_factor}; {inner}++)"
        )


class CppVecKernelChecker(CppVecKernel):
    def __init__(self, args, num_threads, tiling_factor):
        super(CppVecKernelChecker, self).__init__(args, num_threads, tiling_factor)

        # Since this kernel is only for checker but does not genreate any
        # code, so we need to decrease the kernel count.
        metrics.generated_kernel_count -= 1
        metrics.generated_cpp_vec_kernel_count -= 1

        # Used to recorde the graph wrapper code as the wrapper_code status could be
        # changed during graph run.
        self._orig_wrapper_code = None

        self.simd_vec = True
        self.fast_vec_list = []
        for k, v in CppVecOverrides.__dict__.items():
            if isinstance(v, staticmethod):
                self.fast_vec_list.append(k)
        self.exit_stack = contextlib.ExitStack()

    def could_vec(self, name: str, index: sympy.Expr):
        assert self.itervars is not None
        # Not a loop
        if len(self.itervars) == 0:
            return False

        most_inner_var = self.itervars[-1]
        return self.is_invariant_under(most_inner_var, index) or self.is_stride1_at(
            most_inner_var, index
        )

    def load(self, name: str, index: sympy.Expr):
        if not V.graph.get_dtype(name) in [
            torch.float,
            torch.float32,
            torch.bool,
            torch.uint8,
        ]:
            self.simd_vec = False
            return self.simd_vec

        index = self.rename_indexing(index)
        self.simd_vec = self.simd_vec and self.could_vec(name, index)
        return self.simd_vec

    def store(self, name, index, value, mode=None):
        if not V.graph.get_dtype(name) in [torch.float, torch.float32]:
            self.simd_vec = False
            return self.simd_vec

        assert "buf" in name
        index = self.rename_indexing(index)

        if mode:
            self.simd_vec = False
            return False

        self.simd_vec = self.simd_vec and self.could_vec(name, index)
        return self.simd_vec

    def reduction(self, name, dtype, src_dtype, reduction_type, index, value):
        if (
            dtype == torch.float
            and src_dtype == torch.float
            and reduction_type in ["max", "min", "sum"]
        ):
            pass
        else:
            self.simd_vec = False
        return self.simd_vec

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert self._orig_wrapper_code is not None
        # Restore the wrapper_code
        V.graph.wrapper_code = self._orig_wrapper_code
        self.exit_stack.__exit__(exc_type, exc_val, exc_tb)

    def __enter__(self):
        # Recorde the graph wrapper code. The wrapper_code status could be
        # changed during graph run. Regarding this checker, we also need to
        # run the graph but we don't expect to change any status that would
        # impact the code generation. Hence, we record the graph wapper code
        # and replace it with a dummy warpper_code and then restore to the
        # original one as long as the checker is finished.
        self._orig_wrapper_code = V.graph.wrapper_code
        V.graph.wrapper_code = WrapperCodeGen()

        class VecCheckerProxy:
            @staticmethod
            def __getattr__(name):
                def inner(*args, **kwargs):
                    if not (name in self.fast_vec_list):
                        self.simd_vec = False
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
                supported_dtype = (torch.float32, torch.int32)
                is_supported_dtype = dtype in (supported_dtype)
                if not is_supported_dtype:
                    self.simd_vec = False
                return is_supported_dtype

            @staticmethod
            def index_expr(expr, dtype):
                self.simd_vec = False
                tmp_var = self.cse.newvar()
                return tmp_var

            @staticmethod
            def indirect_indexing(index_var):
                self.simd_vec = False
                return sympy.Symbol(str(index_var))

            @staticmethod
            def masked(mask, body, other):
                tmp_var = self.cse.newvar()
                return tmp_var

            @staticmethod
            def to_dtype(x, dtype):
                if dtype != torch.bool:
                    self.simd_vec = False
                return x

        self.exit_stack.enter_context(V.set_ops_handler(VecCheckerProxy()))
        self.exit_stack.enter_context(V.set_kernel_handler(self))
        return self


class CppTile2DKernelChecker(CppVecKernelChecker):
    """
    Currently, we only address the situations with following constraints.
    1. There exists one and only one fp32 load/store with outer loop var having contiguous buffer accesses.
    2. When a load/store doesn't have contiguous access in an outer loop var, the access should be
       vectorizable from the inner-most dim.
    3. No reduction.
    """

    def __init__(self, args, num_threads, tiling_factor):
        super().__init__(args, num_threads, tiling_factor)
        self.can_tile2d = True
        self.outer_tiling_idx = -1

    def check_can_tile2d(self, name: str, index: sympy.Expr):
        if not self.can_tile2d:
            return
        # check contiguity from any of the outer loops
        has_stride1 = False
        for loop_idx, itervar in enumerate(self.itervars[:-1]):
            if self.is_stride1_at(itervar, index):
                # only support 2d tile now
                if V.graph.get_dtype(name) not in [torch.float, torch.float32] or (
                    self.outer_tiling_idx >= 0 and self.outer_tiling_idx != loop_idx
                ):
                    self.can_tile2d = False
                    return
                else:
                    self.outer_tiling_idx = loop_idx
                has_stride1 = True
        if not has_stride1 and not self.could_vec(name, index):
            self.can_tile2d = False
        return self.can_tile2d

    def load(self, name: str, index: sympy.Expr):
        if not V.graph.get_dtype(name) in [
            torch.float,
            torch.float32,
            torch.bool,
            torch.uint8,
        ]:
            self.can_tile2d = False
            return self.can_tile2d
        index = self.rename_indexing(index)
        return self.check_can_tile2d(name, index)

    def store(self, name, index, value, mode=None):
        if not V.graph.get_dtype(name) in [
            torch.float,
            torch.float32,
        ]:
            self.can_tile2d = False
            return self.can_tile2d
        index = self.rename_indexing(index)
        return self.check_can_tile2d(name, index)

    def reduction(self, name, dtype, src_dtype, reduction_type, index, value):
        self.can_tile2d = False
        return self.can_tile2d

    def __exit__(self, exc_type, exc_val, exc_tb):
        super().__exit__(exc_type, exc_val, exc_tb)
        if not self.simd_vec or self.outer_tiling_idx < 0:
            self.can_tile2d = False


class CppKernelProxy(CppKernel):
    def __init__(self, kernel_group):
        super(CppKernelProxy, self).__init__(
            kernel_group.args, kernel_group.ws.num_threads
        )
        self.kernel_group = kernel_group
        self.loop_nest = None
        self.call_ranges = None
        self.picked_vec_isa: codecache.VecISA = codecache.pick_vec_isa()

    def codegen_nodes(self, nodes):
        kernel_group = self.kernel_group
        _, (group, reduction_group) = max(
            nodes, key=lambda x: int(x.is_reduction())
        ).group

        def codegen_kernel(cls, *args):
            with kernel_group.new_kernel(cls, *args) as kernel:
                run(kernel)

                # Ugly hack to maitain the metrics kernel count since
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
        inner_most_idx = len(scalar_kernel.itervars) - 1
        self.call_ranges = scalar_kernel.call_ranges
        self.loop_nest = LoopNestWithSplit.build(scalar_kernel)

        if not self.picked_vec_isa:
            return

        # TODO(jgong5): support alternative tiling factors and data types
        tiling_factor = self.picked_vec_isa.nelements(dtype=torch.float)

        # Kernels share the same global contexts like V.graph.wrapper_code, V.kernel.args.
        # But the generated scalar kernel has updated these global contexts. Hence, the other kernels
        # should not do this again to avoid context conflict. By now, we only control the
        # config.inplace_buffers. In the future, we could maintain more contexts.
        with patch.object(torch._inductor.config, "inplace_buffers", False):

            with CppVecKernelChecker(
                deepcopy(self.kernel_group.args), parallel_num_threads(), tiling_factor
            ) as vec_checker:
                run(vec_checker)

            with CppTile2DKernelChecker(
                deepcopy(self.kernel_group.args), parallel_num_threads(), tiling_factor
            ) as tile2d_checker:
                run(tile2d_checker)

            if vec_checker.simd_vec:
                main_loop, tail_loop = self.loop_nest.split_with_tiling(
                    inner_most_idx, factor=tiling_factor
                )
                main_loop.set_kernel(codegen_kernel(CppVecKernel, tiling_factor))
                tail_loop.set_kernel(scalar_kernel)
                main_loop.simd_vec = True
                tail_loop.simd_omp = True
                # We chop the loop into two cubes by the nelements - main loop and tail loop.
                # Regarding the main loop, it is straightforward that it could be vectorized with
                # nelements. But for the tail loop, it still could be vectorized. For example,
                # if the nelements is 8(256bits), then the tail loop still could be vectorized
                # as 4(128bits).
                tail_loop.simd_nelements = tiling_factor // 2
            elif tile2d_checker.can_tile2d:
                outer_tiling_idx = tile2d_checker.outer_tiling_idx
                assert outer_tiling_idx < inner_most_idx
                outer_main_loop, outer_tail_loop = self.loop_nest.split_with_tiling(
                    outer_tiling_idx, factor=tiling_factor
                )
                outer_tail_loop.set_kernel(scalar_kernel)
                inner_main_loop, inner_tail_loop = outer_main_loop.split_with_tiling(
                    inner_most_idx - outer_tiling_idx, factor=tiling_factor
                )
                inner_main_loop.set_kernel(
                    codegen_kernel(CppTile2DKernel, tiling_factor, outer_tiling_idx)
                )
                inner_tail_loop.set_kernel(
                    codegen_kernel(CppTile2DTailKernel, tiling_factor, outer_tiling_idx)
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

        kernel_group.finalize_kernel(cpp_kernel_proxy, None)

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
        self.count = 0

    def new_kernel(self, cls, *args):
        return cls(self.args, parallel_num_threads(), *args)

    def finalize_kernel(self, new_kernel, scheduler):
        self.count += 1
        code = self.loops_code
        ws = self.ws
        new_kernel.codegen_loops(code, ws)

    def codegen_define_and_call(self, wrapper):
        self.stack.close()
        if self.count == 0:
            return

        kernel_name = "kernel_cpp_" + wrapper.next_kernel_suffix()
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
        code.writelines([cpp_prefix(), "" f'extern "C" void kernel({arg_defs})'])
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
        codecache_def.writeline("async_compile.cpp('''")
        codecache_def.splice(code)
        codecache_def.writeline("''')")

        codecache_str = codecache_def.getvalue()
        # TODO(voz): Ostensibly, we should not need this. But there are cases where C++ codegen does
        # not use BracesBuffer, so we have no good indicator of a C++ buffer atm.
        codecache_str = codecache_str.replace("#pragma CMT", "//")
        wrapper.define_kernel(kernel_name, codecache_str)
        wrapper.load_kernel(kernel_name, code, arg_types)
        # generate the code to call this
        wrapper.generate_kernel_call(kernel_name, call_args)


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

            main_loop_range = ir.IndexingDiv(self.size, sympy_factor)
            main_loop = LoopLevel(self.var, main_loop_range)
            main_loop.parallel = self.parallel
            main_loop.collapsed = False
            main_loop.reduction_var_map = self.reduction_var_map
            main_loop.inner = clone_inner()
            if main_loop.inner:
                for loop in main_loop.inner:
                    loop.parent = main_loop

            offset = main_loop_range * sympy_factor
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
        if self.inner:
            for idx, inner_loop in enumerate(self.inner):
                loop.inner[idx] = inner_loop.clone()
                loop.inner[idx].parent = loop
        loop.kernel = deepcopy(self.kernel)
        return loop

    def lines(self):
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
        line2 = f"for({INDEX_TYPE} {self.var}={cexpr(self.offset)}; {self.var}<{cexpr(self.size)}; {self.var}+={cexpr(self.steps)})"
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
