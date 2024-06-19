# mypy: allow-untyped-defs
from __future__ import annotations

import itertools
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TYPE_CHECKING, Union

import sympy

import torch
import torch._logging
from ..._prims_common import is_integer_dtype
from ...utils._sympy.symbol import symbol_is_type, SymT
from ...utils._sympy.value_ranges import ValueRanges
from .. import config, ir
from ..codecache import HalideCodeCache
from ..metrics import is_metric_table_enabled, log_kernel_metadata

from ..runtime.hints import HalideInputSpec, HalideMeta, ReductionHint
from ..utils import (
    get_bounds_index_expr,
    get_kernel_metadata,
    parallel_num_threads,
    sympy_dot,
    sympy_index_symbol,
    sympy_subs,
)
from ..virtualized import _ops as ops, OpsHandler, V
from .common import (
    BackendFeature,
    CSEVariable,
    DeferredLine,
    IndentedBuffer,
    OpOverrides,
    PythonPrinter,
    SizeArg,
)
from .cpp import DTYPE_TO_CPP
from .cpp_utils import cexpr
from .simd import constant_repr, IterationRangesEntry, SIMDKernel, SIMDScheduling

if TYPE_CHECKING:
    from ..ops_handler import ReductionType, StoreMode

log = logging.getLogger(__name__)


def halide_constant(val):
    if isinstance(val, int) and not (-2147483648 <= val <= 2147483647):
        info = torch.iinfo(torch.int64)
        if val == info.min:
            return "hl.Int(64).min()"
        if val == info.max:
            return "hl.Int(64).max()"
        return f"hl.i64({val!r})"
    if isinstance(val, float):
        return f"hl.f64({constant_repr(val)})"
    return repr(val)


class Unsupported(RuntimeError):
    def __init__(self, thing):
        super().__init__(f"halide backend does not support: {thing}")


class HalidePrinter(PythonPrinter):
    @staticmethod
    def cast_index(expr):
        return f"hl.cast({V.kernel.index_dtype}, {expr})"

    @staticmethod
    def cast_float(expr):
        return f"hl.cast(hl.Float(32), {expr})"

    def _print_floor(self, expr):
        assert len(expr.args) == 1
        return self.cast_index(f"hl.floor({self._print(expr.args[0])})")

    def _print_Trunc(self, expr):
        assert len(expr.args) == 1
        return self.cast_index(f"hl.trunc({self._print(expr.args[0])})")

    def _print_ceiling(self, expr):
        assert len(expr.args) == 1
        return self.cast_index(f"hl.ceil({self._print(expr.args[0])})")

    def _helper_sqrt(self, expr):
        return f"hl.sqrt({self.cast_float(self._print(expr))})"

    def _print_Where(self, expr):
        c = self.doprint(expr.args[0])
        p = self.doprint(expr.args[1])
        q = self.doprint(expr.args[2])
        return f"hl.select({c}, {p}, {q})"

    def _print_Min(self, expr):
        if len(expr.args) == 1:
            return self._print(expr.args[0])

        mid = len(expr.args) // 2
        a = self._print(sympy.Min(*expr.args[:mid]))
        b = self._print(sympy.Min(*expr.args[mid:]))
        return f"hl.min({a}, {b})"

    def _print_Max(self, expr):
        if len(expr.args) == 1:
            return self._print(expr.args[0])

        mid = len(expr.args) // 2
        a = self._print(sympy.Max(*expr.args[:mid]))
        b = self._print(sympy.Max(*expr.args[mid:]))

        return f"hl.max({a}, {b})"

    def _print_Abs(self, expr):
        assert len(expr.args) == 1
        return self.cast_index(f"hl.abs({self._print(expr.args[0])})")

    def _print_OpaqueUnaryFn_cos(self, expr):
        assert len(expr.args) == 1
        return f"hl.cos(({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_cosh(self, expr):
        assert len(expr.args) == 1
        return f"hl.cosh(({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_acos(self, expr):
        assert len(expr.args) == 1
        return f"hl.acos(({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_sin(self, expr):
        assert len(expr.args) == 1
        return f"hl.sin(({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_sinh(self, expr):
        assert len(expr.args) == 1
        return f"hl.sinh(({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_asin(self, expr):
        assert len(expr.args) == 1
        return f"hl.asin(({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_tan(self, expr):
        assert len(expr.args) == 1
        return f"hl.tan(({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_tanh(self, expr):
        assert len(expr.args) == 1
        return f"hl.tanh(({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_atan(self, expr):
        assert len(expr.args) == 1
        return f"hl.atan(({self._print(expr.args[0])})"

    def _print_FloorDiv(self, expr):
        if expr.is_integer:
            return super()._print_FloorDiv(expr)

        x, div = expr.args
        x = self.cast_float(self.paren(self.doprint(x)))
        div = self.cast_float(self.paren(self.doprint(div)))
        return self.cast_index(f"hl.floor({x} / {div})")

    def _print_Round(self, expr):
        assert len(expr.args) == 1
        return self.cast_index(f"hl.round({self._print(expr.args[0])})")

    _print_RoundToInt = _print_Round

    def _print_IntTrueDiv(self, expr):
        a, b = expr.args
        # force a cast to float
        return f"({a}) / ({b}+hl.f32(0))"

    def _print_RoundDecimal(self, expr):
        val, n = expr.args
        val = self._print(val)
        n = int(n)
        return f"hl.f32({10.**(-n)!r})*hl.round(({val})*hl.f32({10.**n!r}))"


texpr = HalidePrinter().doprint
pexpr = PythonPrinter().doprint


_halide_type = {
    torch.bool: "hl.Bool()",
    torch.float16: "hl.Float(16)",
    torch.float32: "hl.Float(32)",
    torch.float64: "hl.Float(64)",
    torch.int8: "hl.Int(8)",
    torch.int16: "hl.Int(16)",
    torch.int32: "hl.Int(32)",
    torch.int64: "hl.Int(64)",
    torch.uint8: "hl.UInt(8)",
    torch.uint16: "hl.UInt(16)",
    torch.uint32: "hl.UInt(32)",
    torch.uint64: "hl.UInt(64)",
}


def halide_type(dtype):
    if dtype == torch.bfloat16:
        raise Unsupported("torch.bfloat16")
    return _halide_type[dtype]


def halide_acc_type(dtype):
    if is_integer_dtype(dtype) and dtype.is_signed and dtype != torch.int64:
        dtype = torch.int32
    if dtype in (torch.float16, torch.bfloat16):
        dtype = torch.float32
    return halide_type(dtype)


class HalideOverrides(OpOverrides):
    @staticmethod
    def to_dtype(x, dtype: torch.dtype, src_dtype: Optional[torch.dtype] = None):
        if dtype == torch.bool:
            return f"({x} != 0)"
        return f"hl.cast({halide_type(dtype)}, {x})"

    @staticmethod
    def to_dtype_bitcast(x, dtype: torch.dtype, src_dtype: torch.dtype):
        if src_dtype in (torch.float16, torch.bfloat16):
            x = f"hl.cast({halide_type(src_dtype)}, {x})"  # body compute is upcast to fp32
        line = f"hl.reinterpret({halide_type(dtype)}, {x})"
        if dtype in (torch.float16, torch.bfloat16):
            line = f"hl.cast(hl.Float(32), {line})"
        return line

    @classmethod
    def constant(cls, value, dtype):
        return cls.to_dtype(halide_constant(value), dtype)

    @staticmethod
    def abs(x):
        return f"hl.abs({x})"

    @staticmethod
    def exp(x):
        return f"hl.fast_exp(hl.cast(hl.Float(32), {x})) if {x.name}.type().bits() <= 32 else hl.exp({x})"

    @staticmethod
    def libdevice_exp(x):
        return f"hl.exp({x})"  # higher precision that ops.exp

    @staticmethod
    def sqrt(x):
        return f"hl.sqrt({x})"

    @staticmethod
    def minimum(a, b):
        # return f"hl.min({a}, {b})"  <== handles nan wrong
        return f"hl.select(({a}<{b})|hl.is_nan({a}), {a}, {b}) if {a.name}.type().is_float() else hl.min({a}, {b})"

    @staticmethod
    def maximum(a, b):
        # return f"hl.max({a}, {b})"  <== handles nan wrong
        return f"hl.select(({a}>{b})|hl.is_nan({a}), {a}, {b}) if {a.name}.type().is_float() else hl.max({a}, {b})"

    @staticmethod
    def where(a, b, c):
        return f"hl.select({a}, {b}, hl.cast({b.name}.type(), {c}))"

    @staticmethod
    def cos(x):
        return f"hl.cos({x})"

    @staticmethod
    def sin(x):
        return f"hl.sin({x})"

    @staticmethod
    def lgamma(x):
        raise Unsupported("lgamma")

    @staticmethod
    def erf(x):
        return f"hl.erf({x})"

    @staticmethod
    def cosh(x):
        return f"hl.cosh({x})"

    @staticmethod
    def sinh(x):
        return f"hl.sinh({x})"

    @staticmethod
    def acos(x):
        return f"hl.acos({x})"

    @staticmethod
    def acosh(x):
        return f"hl.acosh({x})"

    @staticmethod
    def asin(x):
        return f"hl.asin({x})"

    @staticmethod
    def asinh(x):
        return f"hl.asinh({x})"

    @staticmethod
    def atan2(x, y):
        return f"hl.atan2({x}, {y})"

    @staticmethod
    def atan(x):
        return f"hl.atan({x})"

    @staticmethod
    def atanh(x):
        return f"hl.atanh({x})"

    @staticmethod
    def copysign(x, y):
        raise Unsupported("copysign")

    @staticmethod
    def erfinv(x):
        raise Unsupported("erfinv")

    @staticmethod
    def hypot(x, y):
        return f"hl.hypot({x}, {y})"

    @staticmethod
    def nextafter(x, y):
        raise Unsupported("nextafter")

    @staticmethod
    def logical_and(a, b):
        return f"{a} & {b}"

    @staticmethod
    def logical_not(a):
        return f"{a} == 0"

    @staticmethod
    def logical_or(a, b):
        return f"{a} | {b}"

    @staticmethod
    def logical_xor(a, b):
        return f"({a} ^ {b})"

    @staticmethod
    def bitwise_and(a, b):
        return f"{a} & {b}"

    @staticmethod
    def bitwise_not(a):
        return f"~{a}"

    @staticmethod
    def bitwise_or(a, b):
        return f"{a} | {b}"

    @staticmethod
    def bitwise_xor(a, b):
        return f"{a} ^ {b}"

    @staticmethod
    def bitwise_left_shift(a, b):
        return f"{a} << {b}"

    @staticmethod
    def bitwise_right_shift(a, b):
        return f"{a} >> {b}"

    @staticmethod
    def rand(seed, offset):
        raise Unsupported("rand")

    @staticmethod
    def randn(seed, offset):
        raise Unsupported("rand")

    @staticmethod
    def randint64(seed, offset, low, high):
        raise Unsupported("rand")

    @staticmethod
    def load_seed(name, offset):
        raise Unsupported("rand")

    @staticmethod
    def rsqrt(x):
        # return f"hl.fast_inverse_sqrt({x})"  <== accuracy issues
        return f"1./hl.sqrt({x})"

    @staticmethod
    def tan(x):
        return f"hl.tan({x})"

    @staticmethod
    def tanh(x):
        return f"hl.tanh({x})"

    @staticmethod
    def signbit(x):
        return f"(hl.reinterpret(hl.UInt(32), hl.cast(hl.Float(32), {x})) >> 31) != 0"

    @staticmethod
    def fmod(a, b):
        # TODO(jansel): find a better way to do this, builtin % has wrong sign
        return f"{a} - hl.trunc({a}/{b})*{b}"

    @staticmethod
    def pow(a, b):
        return f"hl.pow({a}, {b})"  # hl.fast_pow fails accuracy

    @staticmethod
    def log(x):
        return f"hl.log({x})"  # hl.fast_log fails accuracy

    @staticmethod
    def isinf(x):
        return f"hl.is_inf({x})"

    @staticmethod
    def isnan(x):
        return f"hl.is_nan({x})"

    @staticmethod
    def round(x):
        return f"hl.round({x})"

    @staticmethod
    def floor(x):
        return f"hl.floor({x})"

    @staticmethod
    def int_truediv(a, b):
        return f"({a}) / ({b} + hl.f32(0))"

    @staticmethod
    def floordiv(a, b):
        # TODO(jansel): find a better ways to do this, the select-based trick from triton.py didn't work
        return (
            f"hl.floor(hl.cast(hl.Float(max(32, {a.name}.type().bits())), {a}) / {b})"
        )

    @classmethod
    def sign(cls, x):
        left = ops.to_dtype(ops.lt("0", x), torch.int8)
        right = ops.to_dtype(ops.lt(x, "0"), torch.int8)
        sub = ops.sub(left, right)
        return f"hl.cast({x.name}.type(), {sub})"

    @staticmethod
    def trunc(x):
        return f"hl.trunc({x})"

    @staticmethod
    def truncdiv(a, b):
        # this causes crashes with floating point exception, see test_div_zero_dim_cpu
        # return f"hl.div_round_to_zero({a}, {b})"
        return (
            f"hl.trunc(hl.cast(hl.Float(max(32, {a.name}.type().bits())), {a}) / {b})"
        )

    @staticmethod
    def ceil(x):
        return f"hl.ceil({x})"

    @staticmethod
    def relu(x):
        return f"hl.max({x}, 0)"

    @classmethod
    def index_expr(cls, expr, dtype):
        index = V.kernel.prepare_indexing(expr)
        var = V.kernel.genfunc(
            V.kernel.index_to_str(index),
            V.kernel.used_dims_from_index(index),
            bounds=get_bounds_index_expr(expr),
        )
        if dtype not in {torch.int32, torch.int64}:
            return ops.to_dtype(var, dtype)
        return var

    @classmethod
    def indirect_indexing(cls, index_var, size, check=True):
        # TODO(jansel): Halide only supports 32-bit indexing, we should error on overflow
        index_var = ops.to_dtype(index_var, torch.int32)
        index_var = ops.halide_clamp(index_var, size, check)
        return sympy_index_symbol(str(index_var))

    @classmethod
    def halide_clamp(cls, value, size, check):
        end = V.kernel.kexpr(V.kernel.rename_indexing(size) - 1)
        if not isinstance(size, (int, sympy.Integer)):
            end = f"hl.cast({value.name}.type(), {end})"
        # Skip unsafe_promise_clamped to workaround: https://github.com/halide/Halide/issues/8261#issuecomment-2148835692
        # return f"hl.unsafe_promise_clamped({value}, 0, {end})"
        return f"hl.clamp({value}, 0, {end})"

    @staticmethod
    def masked(mask, body, other):
        with V.kernel.mask_loads(mask, other) as new_mask:
            result = body()

        if result.bounds.is_bool:
            other = bool(other)

        # Take dtype from result to prevent accidental promotion
        other = V.kernel.genfunc(
            f"hl.cast({result.name}.type(), {halide_constant(other)})",
            [],
            bounds=ValueRanges.wrap(other),
        )
        # TODO(jansel): look into removing the where in the same places triton does
        return ops.where(new_mask, result, other)


# Use mypy to check protocol implemented correctly
def _typecheck_HalideOverrides(h: HalideOverrides) -> OpsHandler[str]:
    return h


class HalideCSEVariable(CSEVariable):
    undefined_re = re.compile(r"\b(tmp\d+)\[\?\]")

    def __init__(self, name, bounds: ValueRanges[Any]):
        super().__init__(name, bounds)
        self.used_dims: Optional[List[str]] = None

    def update_on_args(self, name, args, kwargs):
        used = set(self.used_dims or ())
        for arg in itertools.chain(args, kwargs.values()):
            if isinstance(arg, HalideCSEVariable):
                assert arg.used_dims is not None, (name, arg, args)
                used.update(arg.used_dims)
        self.used_dims = [t.name for t in V.kernel.range_trees if t.name in used]
        assert len(self.used_dims) == len(used)

    def index_str(self, dims):
        if len(dims) == 0:
            return self.name
        # Reversed since Halide is column major
        return f"{self.name}[{', '.join(map(str, reversed(dims)))}]"

    def __str__(self):
        if self.used_dims is None:
            # This will get recomputed and replaced in codegen_kernel()
            return f"{self.name}[?]"
        return self.index_str(self.used_dims)

    def with_dom(self, suffix):
        assert self.used_dims is not None
        return self.index_str([f"{d}_{suffix}" for d in self.used_dims])

    def reduction_str(self):
        assert self.used_dims is not None
        dims = [*self.used_dims]
        assert dims[-1] == "rindex"
        dims[-1] = "rdom"
        return self.index_str(dims)


class HalideKernel(SIMDKernel):
    overrides = HalideOverrides  # type: ignore[assignment]
    kexpr: Callable[[sympy.Expr], str] = texpr

    def __init__(
        self,
        *groups,
        index_dtype: str,
        mutations: Optional[Set[str]] = None,
        pid_cache=None,
        reduction_hint=ReductionHint.DEFAULT,
        disable_persistent_reduction=False,
    ):
        super().__init__(
            *groups,
            index_dtype=index_dtype,
            mutations=mutations,
            reduction_hint=reduction_hint,
            pid_cache=pid_cache,
            disable_persistent_reduction=disable_persistent_reduction,
        )
        # For halide, we just write directly to the body
        self.compute = self.body
        self.loads = self.body
        self.stores = self.body
        self.indexing_code_dom = IndentedBuffer()
        self.needs_dom_indexing = self.inside_reduction
        self.has_reduction = self.inside_reduction
        self.store_buffer_dimensions: Dict[str, List[sympy.Expr]] = {}

    def create_cse_var(self, name, bounds=None):
        self.body.writeline(f"{name} = hl.Func({name!r})")
        return HalideCSEVariable(name, bounds)

    def codegen_iteration_ranges_entry(self, entry: IterationRangesEntry):
        expr = self.rename_indexing(entry.expr)
        self.indexing_code.writeline(f"{entry.name} = {self.kexpr(expr)}")

        if self.has_reduction:
            # idom includes iteration ranges of the numel of inputs
            expr_idom = sympy_subs(
                expr,
                {
                    tree.symbol(): sympy_index_symbol(f"{tree.name}_idom")
                    for tree in self.range_trees
                },
            )
            self.indexing_code_dom.writeline(
                f"{entry.name}_idom = {self.kexpr(expr_idom)}"
            )

        if entry.prefix != "r":
            # idom includes iteration ranges of the numel of outputs (which is different for reductions)
            expr_idom = sympy_subs(
                expr,
                {
                    tree.symbol(): sympy_index_symbol(f"{tree.name}_odom")
                    for tree in self.range_trees
                },
            )
            self.indexing_code_dom.writeline(
                f"{entry.name}_odom = {self.kexpr(expr_idom)}"
            )

    def used_dims_from_index(self, index: sympy.Expr):
        """Detect which range trees are used to populate HalideCSEVariable.used_dims"""
        used_dims = set()
        for sym in index.free_symbols:
            assert isinstance(sym, sympy.Symbol)
            if symbol_is_type(sym, SymT.TMP):
                # indirect indexing
                cse_var = self.lookup_cse_var(sym.name)
                assert (
                    isinstance(cse_var, HalideCSEVariable)
                    and cse_var.used_dims is not None
                )
                used_dims.update(cse_var.used_dims)
            elif symbol_is_type(
                sym, (SymT.UNBACKED_INT, SymT.SIZE, SymT.PRECOMPUTED_SIZE, SymT.INDEX)
            ):
                pass
            else:
                # sym is one of xN, yN or rN
                assert symbol_is_type(
                    sym, (SymT.RINDEX, SymT.XBLOCK, SymT.YBLOCK)
                ), sym.name
                used_dims.add(f"{sym.name[0]}index")

        ordered = [tree.name for tree in self.range_trees if tree.name in used_dims]
        assert len(ordered) == len(used_dims)
        return ordered

    def load(self, name: str, index: sympy.Expr):
        """Codegen a load from an InputBuffer"""
        var = self.args.input(name)
        index = self.prepare_indexing(index)
        index_str = self.index_to_str(index)
        if self.is_indirect_indexing(index) or self._load_mask:
            # Halide doesn't have a great way to do masked loads
            var = f"hl.BoundaryConditions.constant_exterior({var}, 0)"
        line = f"{var}[{index_str}]"
        dtype = V.graph.get_dtype(name)
        if dtype in (torch.float16, torch.bfloat16):
            line = f"hl.cast(hl.Float(32), {line})"
        return self.genfunc(line, self.used_dims_from_index(index))

    def index_to_dom(self, index: sympy.Expr, suffix: str):
        """Replace xindex => xindex_dom, x0 => x0_dom, etc for update-style indexing"""
        replacements: Dict[sympy.Expr, Any] = {}
        for sym in index.free_symbols:
            assert isinstance(sym, sympy.Symbol)
            if symbol_is_type(sym, SymT.TMP):
                # indirect indexing
                cse_var = self.lookup_cse_var(sym.name)
                assert isinstance(cse_var, HalideCSEVariable)
                replacements[sym] = sympy.Symbol(cse_var.with_dom(suffix))
            elif symbol_is_type(
                sym, (SymT.UNBACKED_INT, SymT.SIZE, SymT.PRECOMPUTED_SIZE, SymT.INDEX)
            ):
                pass
            else:
                # sym is one of xN, yN or rN
                assert symbol_is_type(
                    sym, (SymT.RINDEX, SymT.XBLOCK, SymT.YBLOCK)
                ), sym.name
                replacements[sym] = sympy.Symbol(f"{sym.name}_{suffix}")
        return sympy_subs(index, replacements)

    def lookup_cse_var(self, name: str):
        return self.cse.varname_map[re.sub(r"\[.*", "", name)]

    def determine_store_indexing(
        self, name: str, index: sympy.Expr, value: HalideCSEVariable, var: str, mode
    ):
        """
        Halide requires the initial definition of an output to be done with a plain Var(),
        while subsequent updates can use Expr().  For us index may be an Expr. This function
        tries to make the output index a var, and if that fails switches to the more flexible
        hl.RDom()+update codegen.
        """
        assert value.used_dims is not None
        assert var not in self.store_buffer_dimensions
        eq = V.graph.sizevars.statically_known_equals

        if index == 0 and eq(self.halide_buffer_numel(name), 1) and mode is None:
            # 1-element case
            index_str = "hl.Var()"  # halide requires storage dst to be a Var
            value_str = value.index_str([0 for _ in value.used_dims])
            return index_str, value_str

        var_ranges = self.var_ranges()
        range_trees = self.active_range_trees()
        numel = self.halide_buffer_numel(name)

        if (
            isinstance(index, sympy.Symbol)
            and index in var_ranges
            and eq(var_ranges[index], numel)
            and mode is None
        ):
            value_str = str(value)
            index_str = self.index_to_str(index)
            return index_str, value_str

        try:
            value_index, dim_sizes, index_vars = self.match_strides_to_dimensions(
                index, var_ranges, range_trees, f"{var}_i", mode
            )
        except NotImplementedError:
            pass
        else:
            self.store_buffer_dimensions[var] = dim_sizes
            for v in index_vars:
                self.body.writeline(
                    DeferredLine(name, f"{v.name} = hl.Var({v.name!r})")
                )
            index_str = ", ".join(v.name for v in index_vars)
            value_str = value.index_str([value_index[d[0]] for d in value.used_dims])
            return index_str, value_str

        self.needs_dom_indexing = True
        # Fall back to using RDom-style store
        self.body.writeline(
            DeferredLine(name, f"{var}[hl.Var()] = hl.undef({var}.type())")
        )
        suffix = "idom" if self.inside_reduction else "odom"
        value_str = value.with_dom(suffix)
        index_str = self.index_to_str(self.index_to_dom(index, suffix))
        return index_str, value_str

    def match_strides_to_dimensions(
        self, index, var_ranges, range_trees, varname, mode
    ):
        """Best effort conversion of 1D indexing into N-D indexing"""
        if mode is not None:
            raise NotImplementedError  # atomic_add
        eq = V.graph.sizevars.statically_known_equals
        used_vars = set(index.free_symbols)
        var_ranges = {s: v for s, v in var_ranges.items() if s in used_vars}
        strides = V.graph.sizevars.stride_vars(index, var_ranges)
        if not strides or not eq(sympy_dot(var_ranges, strides), index):
            raise NotImplementedError  # complex or indirect indexing

        tree_numels = {t.prefix: sympy.Integer(1) for t in range_trees}
        prefix_to_tree = {t.prefix: t for t in range_trees}
        expected_stride = sympy.Integer(1)
        new_lengths = []
        new_index = {t.prefix: sympy.Integer(0) for t in range_trees}
        new_vars: List[sympy.Symbol] = []
        for stride, (v, length) in sorted(
            zip(strides, var_ranges.items()),
            key=lambda x: V.graph.sizevars.size_hint(x[0], fallback=float("inf")),  # type: ignore[arg-type]
        ):
            if not eq(expected_stride, stride):
                raise NotImplementedError  # gaps in indexing or unbacked symints
            prefix = v.name[0]
            if prefix_to_tree[prefix].lookup(tree_numels[prefix], length) != v:
                raise NotImplementedError  # output reordering
            new_var = sympy.Symbol(f"{varname}{len(new_vars)}")
            new_vars.append(new_var)
            new_lengths.append(length)
            new_index[prefix] += tree_numels[prefix] * new_var
            tree_numels[prefix] *= length
            expected_stride *= length
        return new_index, new_lengths, new_vars

    def store(
        self, name: str, index: sympy.Expr, value: CSEVariable, mode: StoreMode = None
    ) -> None:
        """Codegen a store to an OutputBuffer"""
        var = self.args.output(name)
        index = self.prepare_indexing(index)
        assert isinstance(value, HalideCSEVariable)
        index_str, value_str = self.determine_store_indexing(
            name, index, value, var, mode
        )

        if self.is_indirect_indexing(index):
            # Workaround "Buffer out_ptr0 may be accessed in an unbounded way"
            # TODO(jansel): we should error here rather than writing to the first/last element
            index_str = f"hl.clamp({index_str}, 0, {self.kexpr(self.halide_buffer_numel(name) - 1)})"

        if mode is None:
            line = f"{var}[{index_str}] = hl.cast({var}.type(), {value_str})"
        elif mode == "atomic_add":
            line = f"{var}[{index_str}] += {value_str}"
        else:
            raise NotImplementedError(f"store mode={mode}")
        self.body.writeline(DeferredLine(name, line))

    def reduction(
        self,
        dtype: torch.dtype,
        src_dtype: torch.dtype,
        reduction_type: ReductionType,
        value: Union[CSEVariable, Tuple[CSEVariable, ...]],
    ) -> Union[CSEVariable, Tuple[CSEVariable, ...]]:
        """Codegen a reduction operation"""
        assert self.inside_reduction
        assert not self._load_mask
        assert isinstance(value, HalideCSEVariable) and value.used_dims is not None

        cache_key = (src_dtype, reduction_type, value)
        if cache_key in self.cse.reduction_cache:
            return self.cse.reduction_cache[cache_key]

        acc_type = halide_acc_type(dtype)
        result_var = self.newfunc(
            [
                tree.name
                for tree in self.range_trees[:-1]
                if tree.name in value.used_dims
            ]
        )

        default = ir.Reduction.default_accumulator(reduction_type, src_dtype)

        if value.used_dims[-1] != "rindex":
            value = self.genfunc(f"{value}", [*value.used_dims, "rindex"])
        value_str = value.reduction_str()

        if reduction_type in ("argmax", "argmin"):
            self.body.writeline(
                f"{result_var} = hl.{reduction_type}(rdom, {value_str})[0]"
            )
        elif reduction_type in ("sum", "prod", "min", "max", "any"):
            fn = {
                "sum": "sum",
                "prod": "product",
                "min": "minimum",
                "max": "maximum",
                "any": "maximum",
            }[reduction_type]
            self.body.writeline(f"{result_var} = hl.{fn}(rdom, {value_str})")
        elif reduction_type == "xor_sum":
            result_var_init = result_var
            if not result_var.used_dims:  # need a fake dim
                result_var_init = result_var.index_str([self.range_trees[0].name])
                result_var.used_dims = ["0"]
            self.body.writeline(
                f"{result_var_init} = hl.cast({acc_type}, {halide_constant(default)})"
            )
            self.body.writeline(f"{result_var} = {result_var} ^ {value_str}")
        elif reduction_type == "welford_reduce":
            # TODO(jansel): implement welford_reduce without fallback
            result_var = self.welford_reduce_fallback(dtype, value)
        else:
            raise Unsupported(reduction_type)

        self.cse.reduction_cache[cache_key] = result_var
        return result_var

    def genfunc(
        self, line, used_dims, *, bounds=ValueRanges.unknown()
    ) -> HalideCSEVariable:
        var = self.cse.generate(self.body, line, bounds=bounds)
        assert isinstance(var, HalideCSEVariable)
        var.used_dims = used_dims
        return var

    def newfunc(self, used_dims) -> HalideCSEVariable:
        var = self.cse.newvar()
        assert isinstance(var, HalideCSEVariable)
        var.used_dims = used_dims
        return var

    def halide_buffer_numel(self, name: str):
        """
        We map all tensors to 1D buffers in Halide since Halide has trouble representing some strides that PyTorch
        supports.  If there are gaps in the underlying layout the numel we pass to Halide includes the gaps while
        PyTorch's numel excludes them.
        """
        return V.graph.get_buffer(name).get_layout().storage_size()

    def halide_argdefs(self):
        """
        Halide requires scalar inputs before outputs, so need to reorder args.
        """

        def arg_order(arg_tuple):
            call_str, arg = arg_tuple
            if isinstance(arg, SizeArg):
                return 1  # this would normally be at the end, move it to middle
            elif "out_ptr" in arg.name:
                return 2
            else:
                assert "in_ptr" in arg.name
                return 0

        _, a, b, _ = self.args.python_argdefs()
        return sorted(zip(a, b), key=arg_order)

    def halide_kernel_meta(self) -> HalideMeta:
        """Compute metadata required by codecache.py"""
        argtypes = []
        for _, arg in self.halide_argdefs():
            if isinstance(arg, SizeArg):
                shape = None
                dtype = "long"
            else:
                if arg.name in self.store_buffer_dimensions and "out" in arg.name:
                    shape = [
                        cexpr(self.rename_indexing(x))
                        for x in self.store_buffer_dimensions[arg.name]
                    ]
                    assert shape
                else:
                    shape = [
                        cexpr(
                            self.rename_indexing(self.halide_buffer_numel(arg.buffer))
                        )
                    ] or ["1"]
                dtype = f"{DTYPE_TO_CPP[arg.dtype]}*"
            argtypes.append(
                HalideInputSpec(
                    dtype,
                    arg.name,
                    shape,
                )
            )
        target = ["host", "strict_float"]
        # TODO(jansel): for cuda want target="host-cuda-cuda_capability_86-user_context"
        if not config.halide.asserts:
            target.append("no_asserts")
        if "64" in self.index_dtype:
            # TODO(jansel): it is unclear if this does anything, since input sizes are still int32
            target.append("large_buffers")

        return HalideMeta(
            argtypes,
            target="-".join(target),
            scheduler="Mullapudi2016",
            scheduler_flags={
                "parallelism": parallel_num_threads(),
                "last_level_cache_size": HalideCodeCache.cpu_cache_size(),
            },
        )

    def codegen_kernel(self, name=None):
        """Called at the end to generate a final kernel string"""
        if self.args.inplace_buffers:
            raise Unsupported("inplace_buffers")
        meta = self.halide_kernel_meta()  # ensure needed args are added early
        code = IndentedBuffer()
        code.splice(
            """
            import halide as hl

            @hl.generator(name="kernel")
            class Kernel:
        """,
            strip=True,
        )
        code.do_indent()
        for _, arg in self.halide_argdefs():
            if isinstance(arg, SizeArg):
                code.writeline(f"{arg.name} = hl.InputScalar({self.index_dtype})")
            else:
                assert arg.buffer, arg
                argcls = "hl.OutputBuffer" if "out" in arg.name else "hl.InputBuffer"
                argtype = halide_type(arg.dtype)
                ndim = len(self.store_buffer_dimensions.get(arg.name, (0,)))
                code.writeline(f"{arg.name} = {argcls}({argtype}, {ndim})")
        code.splice(
            """
            def generate(g):
        """
        )
        code.do_indent()
        for _, arg in self.halide_argdefs():
            code.writeline(f"{arg.name} = g.{arg.name}")
        for old, new in self.args.aliases():
            code.writeline(f"{old} = {new}")

        dom_size = {}
        for tree in self.active_range_trees(reorder=True):
            code.writeline(f"{tree.name} = hl.Var({tree.name!r})")
            length = self.kexpr(self.rename_indexing(tree.numel))
            dom_size[tree.name] = f"hl.Range(0, {length})"
        assert len(dom_size) <= 3
        code.splice(self.indexing_code)

        if self.inside_reduction:
            sizes = [*dom_size.values()]
            code.writeline(f"idom = hl.RDom([{', '.join(sizes)}])")
            code.writeline(f"odom = hl.RDom([{', '.join(sizes[:-1])}])")
            code.writeline(f"rdom = hl.RDom([{sizes[-1]}])")
            for name, xyz in zip(dom_size.keys(), "xyz"):
                code.writeline(f"{name}_idom = idom.{xyz}")
                if name[0] != "r":
                    code.writeline(f"{name}_odom = odom.{xyz}")
        elif self.needs_dom_indexing:
            code.writeline(f"odom = hl.RDom([{', '.join(dom_size.values())}])")
            for name, xyz in zip(dom_size.keys(), "xyz"):
                code.writeline(f"{name}_odom = odom.{xyz}")

        if self.needs_dom_indexing:
            code.splice(self.indexing_code_dom)

        def update_index(m):
            var = self.cse.varname_map[m.group(1)]
            assert var.used_dims is not None, var
            if var.used_dims:
                return str(var)
            else:
                return var.name  # a constant doesn't need to be wrapped in func

        for line in self.body._lines:
            if isinstance(line, str):
                # fill in missing indices
                line = HalideCSEVariable.undefined_re.sub(update_index, line)
            code.writeline(line)
        code.writeline("")
        code.writeline("assert g.using_autoscheduler()")

        for _, arg in self.halide_argdefs():
            # fallback=1 below because halide requires buffers to be at least as large as the estimates
            # This causes crashes if our estimate is greater than the vector length
            # https://github.com/halide/Halide/issues/3103
            if isinstance(arg, SizeArg):
                hint = V.graph.sizevars.size_hint(arg.expr, fallback=1)
                code.writeline(f"{arg.name}.set_estimate({hint})")
            else:
                if arg.name in self.store_buffer_dimensions and "out" in arg.name:
                    hints = V.graph.sizevars.size_hints(
                        self.store_buffer_dimensions[arg.name], fallback=1
                    )
                else:
                    hints = V.graph.sizevars.size_hints(
                        [V.graph.get_numel(arg.buffer)], fallback=1
                    )
                range_hints = [f"hl.Range(0, {hint})" for hint in hints]
                code.writeline(f"{arg.name}.set_estimates([{', '.join(range_hints)}])")

        code.do_unindent(2)
        code.splice(
            f"""
            if __name__ == "__main__":
                hl.main()
            else:
                hl.load_plugin({HalideCodeCache.find_libautoschedule(meta.scheduler)!r})
                target = hl.Target({meta.target!r})
                autoscheduler = hl.AutoschedulerParams({meta.scheduler!r}, {meta.scheduler_flags!r})
                with hl.GeneratorContext(target, autoscheduler):
                    gen = Kernel()
                    pipeline = gen._build_pipeline()
                    # gen.compile_to_callable() does not run the autoscheduler
                    pipeline.apply_autoscheduler(target, autoscheduler)
                    kernel = pipeline.compile_to_callable([
                            gen._get_input_parameter(a.name)._to_argument()
                            for a in gen._get_arginfos()
                            if a.dir == hl.ArgInfoDirection.Input
                        ], target)
            """
        )
        return code.getvalue()

    def call_kernel(self, name: str, node=None):
        """Codegen a call to this kernel"""
        wrapper = V.graph.wrapper_code
        call_args = [f"{n}" for n, _ in self.halide_argdefs()]
        assert V.graph.scheduler.current_device is not None
        current_device = V.graph.scheduler.current_device
        assert current_device.type == "cpu", "TODO"
        wrapper.generate_kernel_call(
            name,
            call_args,
            cuda=False,  # grid/stream is handled internally in halide
        )

    def generate_assert(self, check):
        return False  # TODO(jansel): support asserts

    def check_bounds(
        self, expr: sympy.Expr, size: sympy.Expr, lower: bool, upper: bool
    ):
        pass  # TODO(jansel): support asserts


class HalideScheduling(SIMDScheduling):
    int32_type = "hl.Int(32)"
    # TODO(jansel): Halide doesn't actually support 64 bit indexing...
    int64_type = "hl.Int(64)"
    kernel_type = HalideKernel

    @classmethod
    def get_backend_features(cls, device: torch.device):
        result = dict.fromkeys(
            [
                BackendFeature.TUPLE_REDUCTION,
            ]
        )
        return result

    def define_kernel(self, src_code, node_schedule, kernel):
        """Codegen kernel definition to go in output wrapper code"""
        wrapper = V.graph.wrapper_code
        if src_code in wrapper.src_to_kernel:
            kernel_name = wrapper.src_to_kernel[src_code]
        else:
            kernel_name = f"halide_kernel_{wrapper.next_kernel_suffix()}"
            wrapper.src_to_kernel[src_code] = kernel_name
            wrapper.add_import_once(
                "from torch._inductor.runtime.hints import HalideMeta, HalideInputSpec"
            )

            compile_wrapper = IndentedBuffer()
            compile_wrapper.writeline(
                f"async_compile.halide({kernel.halide_kernel_meta()!r}, '''"
            )
            compile_wrapper.splice(src_code, strip=True)
            compile_wrapper.writeline("''')")

            origins, detailed_origins = get_kernel_metadata(node_schedule, wrapper)
            metadata_comment = f"{origins}\n{detailed_origins}"
            wrapper.define_kernel(
                kernel_name, compile_wrapper.getvalue(), metadata_comment
            )
            if is_metric_table_enabled("kernel_metadata"):
                log_kernel_metadata(kernel_name, "", src_code)

        return kernel_name
