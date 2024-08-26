# mypy: allow-untyped-defs
from __future__ import annotations

import dataclasses
import functools
import itertools
import logging
import re
from collections import defaultdict
from math import inf
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    TYPE_CHECKING,
    Union,
)

import sympy

import torch
import torch._logging

from ..._prims_common import is_integer_dtype
from ...utils._sympy.functions import FloorDiv, ModularIndexing
from ...utils._sympy.symbol import symbol_is_type, SymT
from ...utils._sympy.value_ranges import ValueRanges
from .. import config, ir
from ..codecache import HalideCodeCache
from ..ir import get_reduction_combine_fn
from ..metrics import is_metric_table_enabled, log_kernel_metadata
from ..ops_handler import AddParenHandler, MockHandler
from ..runtime.hints import HalideInputSpec, HalideMeta, ReductionHint
from ..utils import (
    get_bounds_index_expr,
    get_kernel_metadata,
    parallel_num_threads,
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
    TensorArg,
)
from .cpp import DTYPE_TO_CPP
from .cpp_utils import cexpr
from .simd import constant_repr, SIMDKernel, SIMDScheduling


if TYPE_CHECKING:
    from torch.utils._ordered_set import OrderedSet

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
    def __init__(self, thing) -> None:
        super().__init__(f"halide backend does not support: {thing}")


class HalidePrinter(PythonPrinter):
    @staticmethod
    def cast_index(expr):
        return f"hl.cast({V.kernel.index_dtype}, {expr})"

    @staticmethod
    def cast_float(expr):
        return f"hl.cast(hl.Float(32), {expr})"

    def _print_Float(self, expr):
        return f"hl.f32({expr})"

    def _print_ToFloat(self, expr):
        assert len(expr.args) == 1
        return f"hl.f32({self._print(expr.args[0])})"

    def _print_floor(self, expr):
        assert len(expr.args) == 1
        return self.cast_index(f"hl.floor({self._print(expr.args[0])})")

    def _print_Trunc(self, expr):
        assert len(expr.args) == 1
        return self.cast_index(f"hl.trunc({self._print(expr.args[0])})")

    _print_TruncToInt = _print_Trunc

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
    torch.bfloat16: "hl.BFloat(16)",
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
    return _halide_type[dtype]


def halide_acc_type(dtype):
    if is_integer_dtype(dtype) and dtype.is_signed and dtype != torch.int64:
        dtype = torch.int32
    if dtype in (torch.float16, torch.bfloat16):
        dtype = torch.float32
    return halide_type(dtype)


class HalideOverrides(OpOverrides):
    @staticmethod
    def to_dtype(
        x,
        dtype: torch.dtype,
        src_dtype: Optional[torch.dtype] = None,
        use_compute_types=True,
    ):
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
        if not hasattr(x, "name"):
            return f"hl.exp({x})"
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
        if not hasattr(a, "name"):
            return f"hl.min({a}, {b})"
        b = f"hl.cast({a.name}.type(), {b})"
        return f"hl.select(({a}<{b})|hl.is_nan({a}), {a}, {b}) if {a.name}.type().is_float() else hl.min({a}, {b})"

    @staticmethod
    def maximum(a, b):
        # return f"hl.max({a}, {b})"  <== handles nan wrong
        if not hasattr(a, "name"):
            return f"hl.max({a}, {b})"
        b = f"hl.cast({a.name}.type(), {b})"
        return f"hl.select(({a}>{b})|hl.is_nan({a}), {a}, {b}) if {a.name}.type().is_float() else hl.max({a}, {b})"

    @staticmethod
    def where(a, b, c):
        if hasattr(b, "name"):
            c = f"hl.cast({b.name}.type(), {c})"
        return f"hl.select({a}, {b}, {c})"

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
        return f"halide_helpers.rand({seed}, {offset})"

    @staticmethod
    def randn(seed, offset):
        return f"halide_helpers.randn({seed}, {offset})"

    @staticmethod
    def randint64(seed, offset, low, high):
        return f"halide_helpers.randint64({seed}, {offset}, {low}, {high})"

    @staticmethod
    def load_seed(name, offset):
        return f"{ops.load(name, 0)} + {V.kernel.args.seed_offset('load_seed_offset', offset)}"

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
        # workaround https://github.com/halide/Halide/issues/8309
        return f"hl.is_inf(hl.cast(hl.Float(32), {x}))"

    @staticmethod
    def isnan(x):
        # workaround https://github.com/halide/Halide/issues/8309
        return f"hl.is_nan(hl.cast(hl.Float(32), {x}))"

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
    def indirect_indexing(cls, index_var, size, check=True, wrap_neg=True):
        # TODO(jansel): Halide only supports 32-bit indexing, we should error on overflow
        index_var = ops.to_dtype(index_var, torch.int32)
        index_var = ops.halide_clamp(index_var, size, check)
        index_var.indirect_indexing_size = size
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

    def __init__(self, name, bounds: ValueRanges[Any]) -> None:
        super().__init__(name, bounds)
        self.used_dims: Optional[List[sympy.Symbol]] = None

    def update_on_args(self, name, args, kwargs):
        used = set(self.used_dims or ())
        for arg in itertools.chain(args, kwargs.values()):
            if isinstance(arg, HalideCSEVariable):
                assert arg.used_dims is not None, (name, arg, args)
                used.update(arg.used_dims)
        self.used_dims = V.kernel.sort_used_dims(used)

    def index_str(self, dims):
        if len(dims) == 0:
            return f"{self.name}[()]"
        # Reversed since Halide is column major
        return f"{self.name}[{', '.join(map(str, dims))}]"

    def __str__(self) -> str:
        if self.used_dims is None:
            # This will get recomputed and replaced in codegen_kernel()
            return f"{self.name}[?]"
        return self.index_str(self.used_dims)

    def subs_str(self, replacements):
        assert self.used_dims is not None and all(
            isinstance(x, sympy.Expr) for x in self.used_dims
        )
        return self.index_str([replacements.get(n, n) for n in self.used_dims])


@dataclasses.dataclass
class DimensionInfo:
    expr: Optional[sympy.Expr]
    size: sympy.Expr
    stride: sympy.Expr

    def __init__(self, expr, size, stride) -> None:
        super().__init__()
        if V.graph.sizevars.statically_known_lt(stride, 0):
            stride = -stride
            expr = -expr
        self.expr = expr
        self.size = size
        self.stride = stride

    def index_str(self, replacements=None, zero_vars=False):
        assert self.expr is not None
        expr = self.expr
        if zero_vars and expr == 0:
            return "hl.Var()"
        if replacements:
            replacements = {**replacements}
            for sym in expr.free_symbols:
                if symbol_is_type(sym, SymT.TMP):
                    assert isinstance(sym, sympy.Symbol)
                    var = V.kernel.lookup_cse_var(sym.name)
                    assert isinstance(var, HalideCSEVariable)
                    replacements[sym] = sympy_index_symbol(var.subs_str(replacements))
            expr = sympy_subs(expr, replacements)
        return V.kernel.index_to_str(expr)


def eq(left, right):
    if V.graph.sizevars.statically_known_equals(left, right):
        return True
    try:
        a = V.graph.sizevars.size_hint(left)
        b = V.graph.sizevars.size_hint(right)
    except TypeError:  # unbacked symints
        return False
    if a == b:
        V.graph.sizevars.guard_equals(left, right)
    return a == b


def lt(left, right):
    if V.graph.sizevars.statically_known_lt(left, right):
        return True
    try:
        a = V.graph.sizevars.size_hint(left)
        b = V.graph.sizevars.size_hint(right)
    except TypeError:  # unbacked symints
        gcd = sympy.gcd(left, right)
        if gcd == left:
            return left != right
        return False
    if a < b:
        V.graph.sizevars.guard_lt(left, right)
    return a < b


class HalideKernel(SIMDKernel):
    overrides = HalideOverrides  # type: ignore[assignment]
    kexpr: Callable[[sympy.Expr], str] = texpr

    def __init__(
        self,
        *groups,
        index_dtype: str,
        mutations: Optional[OrderedSet[str]] = None,
        pid_cache=None,
        reduction_hint=ReductionHint.DEFAULT,
        override_persistent_reduction=None,
    ) -> None:
        super().__init__(
            *groups,
            index_dtype=index_dtype,
            mutations=mutations,
            reduction_hint=reduction_hint,
            pid_cache=pid_cache,
            override_persistent_reduction=override_persistent_reduction,
        )
        # For halide, we just write directly to the body
        self.compute = self.body
        self.loads = self.body
        self.stores = self.body
        self.indexing_code_dom = IndentedBuffer()
        self.needs_dom_indexing = self.inside_reduction
        self.has_reduction = self.inside_reduction
        self.buffer_dimensions: Dict[str, List[DimensionInfo]] = {}
        self.buffer_offsets: Dict[str, sympy.Expr] = {}
        # {h0: size1, h1: size2, ...}
        self.halide_vars: Dict[sympy.Symbol, sympy.Expr] = {}
        # {x0: h0, x1: h1+10*h2, ...}
        self.index_replacements: Dict[sympy.Expr, sympy.Expr] = {}
        # {h1: hr1, ...}
        self.reduction_renames: Dict[sympy.Symbol, sympy.Symbol] = {}
        # {"i": {h0: hi0}, "o": ...}
        self.dom_renames: Dict[str, Dict[sympy.Symbol, sympy.Symbol]] = {}
        # {"in_ptr0": ["in_ptr0_view0"], ...}
        self.buffer_aliases: Dict[str, List[str]] = defaultdict(list)
        self.has_indirect_indexing = False

    def create_cse_var(self, name, bounds=None):
        self.body.writeline(f"{name} = hl.Func({name!r})")
        return HalideCSEVariable(name, bounds)

    def finalize_indexing(self, indices: Sequence[sympy.Expr]):
        """
        Hook called right before codegen with every index that will be
        used in the fused kernel.

        This populates self.halide_vars/index_replacements/reduction_renames which is an alternate indexing
        scheme that avoids using divide and modulus.  Instead of xindex/yindex/rindex
        we base indexing on a larger number of vars whose product combines to those.

        This function populates self.halide_vars, self.index_replacements, and self.reduction_renames
        """
        assert not (
            self.index_replacements or self.halide_vars or self.reduction_renames
        )
        size_hint = functools.partial(V.graph.sizevars.size_hint, fallback=inf)  # type: ignore[arg-type]
        indices = dict.fromkeys(map(super().prepare_indexing, indices))
        all_used_symbols = set()
        sym_to_node = {
            n.symbol(): n
            for n in itertools.chain.from_iterable(
                [tree.nodes.values() for tree in self.range_trees]
            )
        }

        def simplify(expr):
            return sympy.simplify(
                V.graph.sizevars.remove_precomputed_replacements(expr)
            )

        def visit_modular_indexing(base, divisor, modulus):
            if base in sym_to_node:
                node = sym_to_node[base]
                all_used_symbols.add(
                    node.root.lookup(
                        node.divisor * divisor,
                        V.graph.sizevars.evaluate_min(
                            modulus, FloorDiv(node.length, divisor)
                        ),
                    ).symbol()
                )

        def visit_floor_div(base, divisor):
            if base in sym_to_node:
                node = sym_to_node[base]
                all_used_symbols.add(
                    node.root.lookup(
                        node.divisor * divisor,
                        FloorDiv(node.length, divisor),
                    ).symbol()
                )

        # first figure out all_used_symbols to do dead symbol elimination
        for index in indices:
            if index.has(ModularIndexing):
                index.replace(
                    ModularIndexing(
                        sympy.Wild("base"),
                        sympy.Wild("divisor"),
                        sympy.Wild("modulus"),
                    ),
                    visit_modular_indexing,
                )
            if index.has(FloorDiv):
                index.replace(
                    FloorDiv(
                        sympy.Wild("base"),
                        sympy.Wild("divisor"),
                    ),
                    visit_floor_div,
                )
            all_used_symbols.update(super().prepare_indexing(index).free_symbols)

        self.has_indirect_indexing = any(
            symbol_is_type(sym, SymT.INDIRECT) for sym in all_used_symbols
        )

        had_fallback = False
        for tree in reversed(self.range_trees):
            nodes = [n for n in tree.nodes.values() if n.symbol() in all_used_symbols]
            nodes.sort(key=lambda n: size_hint(n.divisor))
            if not nodes:
                nodes.append(tree.lookup(1, tree.numel))
            handled_count = 0
            divisor = sympy.Integer(1)
            added_sym_size = []
            # decide on a minimal set of symbols and put them in self.halide_vars
            while handled_count < len(nodes) and not eq(tree.numel, divisor):
                sizes_to_add = [
                    simplify(n.length) for n in nodes if eq(n.divisor, divisor)
                ]
                handled_count += len(sizes_to_add)
                assert sizes_to_add, nodes
                end = divisor * functools.reduce(
                    V.graph.sizevars.evaluate_max, sizes_to_add
                )
                sizes_to_add.extend(
                    [
                        simplify(n.divisor / divisor)
                        for n in nodes
                        if lt(divisor, n.divisor) and lt(n.divisor, end)
                    ]
                )
                while sizes_to_add:
                    next_size = functools.reduce(sympy.gcd, sizes_to_add)
                    if eq(next_size, 1):
                        # sizes share no common factors, e.g [2, 21, 42, 441, 889056]
                        # TODO(jansel): we should just prevent fusion in cases that hit this
                        next_size = simplify(tree.numel / divisor)
                        assert not eq(next_size, 1)
                        sizes_to_add = []
                        handled_count = len(nodes)
                        had_fallback = True
                    sym = sympy_index_symbol(f"h{len(self.halide_vars)}")
                    if tree.prefix == "r":
                        self.reduction_renames[sym] = sympy_index_symbol(
                            f"hr{len(self.halide_vars)}"
                        )
                    self.halide_vars[sym] = next_size
                    added_sym_size.append((sym, next_size))
                    divisor *= next_size
                    new_sizes = [n.length for n in nodes if eq(n.divisor, divisor)]
                    handled_count += len(new_sizes)
                    prior_len = len(sizes_to_add)
                    sizes_to_add = [
                        sympy.simplify(s / next_size)
                        for s in sizes_to_add
                        if not eq(s, next_size)
                    ]
                    assert len(sizes_to_add) < prior_len or prior_len == 0
                    sizes_to_add.extend(new_sizes)

            # create a mapping to the new set of symbols in self.index_replacements
            for node in nodes:
                try:
                    idx = 0
                    divisor = 1
                    while not eq(node.divisor, divisor):
                        sym, size = added_sym_size[idx]
                        idx += 1
                        divisor *= size
                    length = 1
                    expr = sympy.Integer(0)
                    while not eq(node.length, length):
                        sym, size = added_sym_size[idx]
                        idx += 1
                        expr += length * sym
                        length *= size
                    self.index_replacements[node.symbol()] = expr
                except IndexError:
                    assert had_fallback
                    full_index = sympy.Integer(0)
                    stride = sympy.Integer(1)
                    for sym, size in added_sym_size:
                        full_index += stride * sym
                        stride *= size
                    self.index_replacements[
                        node.symbol()
                    ] = V.graph.sizevars.simplify_with_ranges(
                        ModularIndexing(full_index, node.divisor, node.length),
                        self.halide_vars,  # type: ignore[arg-type]
                    )

        # codegen the variable definitions
        for sym in self.halide_vars:
            self.indexing_code.writeline(f"{sym} = hl.Var({sym.name!r})")
        if self.reduction_renames:
            self.codegen_rdom(
                "rdom",
                {rv: self.halide_vars[v] for v, rv in self.reduction_renames.items()},
            )

    def setup_dom_indexing(self):
        """RDom based indexing uses explicit iteration ranges for Func updates"""
        prefix = "i" if self.inside_reduction else "o"
        if prefix in self.dom_renames:
            return self.dom_renames[prefix]

        renames = {}
        for var in self.halide_vars.keys():
            if not self.inside_reduction and var in self.reduction_renames:
                continue
            m = re.match(r"^h(\d+)$", var.name)
            assert m
            renames[var] = sympy_index_symbol(f"h{prefix}{m.group(1)}")

        self.codegen_rdom(
            f"{prefix}dom", {rv: self.halide_vars[v] for v, rv in renames.items()}
        )

        self.dom_renames[prefix] = renames
        return renames

    def codegen_rdom(self, name, vars):
        rsizes = [
            f"hl.Range(0, {self.kexpr(self.rename_indexing(size))})"
            for size in vars.values()
        ]
        self.indexing_code.writeline(f"{name} = hl.RDom([{', '.join(rsizes)}])")
        for i, rsym in enumerate(vars.keys()):
            self.indexing_code.writeline(f"{rsym} = {name}[{i}]")

    def prepare_indexing(
        self,
        index: sympy.Expr,
    ):
        index = super().prepare_indexing(index)
        index = sympy_subs(index, self.index_replacements)
        return V.graph.sizevars.simplify_with_ranges(index, self.halide_vars)  # type: ignore[arg-type]

    def sym_size(self, sym):
        """The size of an index symbol"""
        if symbol_is_type(sym, SymT.TMP):
            return self.lookup_cse_var(sym.name).indirect_indexing_size
        return self.halide_vars[sym]

    def indexing_to_dimensions(self, var: str, index: sympy.Expr, is_store: bool):
        """Convert address-based indexing into dimensions using self.halide_vars"""
        symbols = []
        for sym in sorted(index.free_symbols, key=lambda x: x.name):  # type: ignore[attr-defined]
            if symbol_is_type(sym, (SymT.HALIDE, SymT.TMP)):
                symbols.append(sym)
            else:
                assert symbol_is_type(
                    sym,
                    (
                        SymT.UNBACKED_INT,
                        SymT.SIZE,
                        SymT.PRECOMPUTED_SIZE,
                    ),
                ), sym

        # group the expression by variables used
        offset = sympy.Integer(0)
        split_expr = {s: sympy.Integer(0) for s in symbols}
        split_failed: List[Tuple[List[sympy.Symbol], sympy.Expr]] = []
        index = sympy.expand(self.rename_indexing(index))
        for part in index.args if isinstance(index, sympy.Add) else [index]:
            part_vars = [v for v in part.free_symbols if v in split_expr]
            if len(part_vars) == 0:
                offset += part
            elif len(part_vars) == 1:
                split_expr[part_vars[0]] += part
            else:
                new_split_failed = []
                for i in range(len(split_failed)):
                    assert split_failed[i] is not None
                    other_vars, other_part = split_failed[i]
                    if set(other_vars) & set(part_vars):
                        part_vars.extend([v for v in other_vars if v not in part_vars])
                        part += other_part
                    else:
                        new_split_failed.append((other_vars, other_part))
                split_failed = [*new_split_failed, (part_vars, part)]

        def expr_to_dimension(expr, syms):
            expr = sympy.factor(expr)
            if len(syms) == 1:
                stride_wild = sympy.Wild("wild", exclude=symbols)
                m = expr.match(stride_wild * syms[0])
                if m:
                    return DimensionInfo(
                        syms[0], self.sym_size(syms[0]), m[stride_wild]
                    )
            assert not is_store, expr
            length = sympy.simplify(
                sympy_subs(expr, {sym: self.sym_size(sym) - 1 for sym in syms}) + 1
            )
            stride = sympy.Integer(1)
            if isinstance(expr, sympy.Mul):
                for term in expr.args:
                    if isinstance(term, sympy.Integer):
                        stride *= term
                        expr = sympy.simplify(expr / term)
                        length = sympy.simplify(sympy.ceiling(length / term))
            return DimensionInfo(expr, length, stride)

        # try to turn each group into a strided access
        dims = []
        for syms, expr in split_failed:
            for v in syms:
                expr += split_expr.pop(v)
            dims.append(expr_to_dimension(expr, syms))
        for sym, expr in split_expr.items():
            dims.append(expr_to_dimension(expr, [sym]))
        dims.sort(key=lambda d: V.graph.sizevars.size_hint(d.stride, fallback=inf))  # type: ignore[arg-type]

        if not dims:  # scalar load/store
            if self.has_indirect_indexing:
                # workaround https://github.com/halide/Halide/issues/8338
                dims.append(DimensionInfo(sympy.Integer(0), 1, 1))
        elif not V.graph.sizevars.statically_known_equals(dims[0].stride, 1):
            # Halide assumes dimension 0 is stride == 1, so add a dummy dimension
            dims.insert(
                0, DimensionInfo(sympy.Integer(0), 1 if is_store else dims[0].stride, 1)
            )

        if dims and not is_store:
            if var in self.buffer_offsets and V.graph.sizevars.statically_known_geq(
                offset, self.buffer_offsets[var]
            ):
                # reuse the existing offset to avoid needing an input alias
                self.apply_offset_to_dimension(dims, offset - self.buffer_offsets[var])
                offset = self.buffer_offsets[var]
            elif V.graph.sizevars.statically_known_gt(
                offset, 0
            ):  # TODO(jansel): negative offsets
                # roll the offset into the dimensions for cleaner indexing
                self.apply_offset_to_dimension(dims, offset)
                offset = 0

        orig_var = var
        for i in itertools.count():
            if self.install_dims(var, dims, offset, is_store):
                return var, dims
            assert not is_store
            var = f"{orig_var}_view{i}"
            if var not in self.buffer_aliases[orig_var]:
                self.buffer_aliases[orig_var].append(var)

    def install_dims(self, var, dims, offset, is_store):
        """Try to set self.buffer_dimensions[var], return True on success"""
        if var not in self.buffer_dimensions:
            self.buffer_dimensions[var] = dims
            self.buffer_offsets[var] = offset
            return True
        if self.buffer_offsets[var] != offset or len(
            self.buffer_dimensions[var]
        ) != len(dims):
            return False
        if is_store:
            return self.buffer_dimensions[var] == dims
        for old, new in zip(self.buffer_dimensions[var], dims):
            if old.stride != new.stride:
                return False
            if old.size != new.size or old.expr != new.expr:
                old.size = V.graph.sizevars.evaluate_max(old.size, new.size)
                old.expr = None
        return True

    def apply_offset_to_dimension(self, dims, offset):
        if offset == 0:
            return
        for i in reversed(range(len(dims))):
            if dims[i].stride == 1 or V.graph.sizevars.statically_known_geq(
                offset, dims[i].stride
            ):
                part = FloorDiv(offset, dims[i].stride)
                offset -= part * dims[i].stride
                dims[i].expr += part
        assert offset == 0

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
            elif symbol_is_type(sym, SymT.HALIDE):
                used_dims.add(sym)
            elif symbol_is_type(
                sym, (SymT.UNBACKED_INT, SymT.SIZE, SymT.PRECOMPUTED_SIZE, SymT.INDEX)
            ):
                pass
            else:
                raise NotImplementedError(f"unhandled symbol {sym}")
        return self.sort_used_dims(used_dims)

    def sort_used_dims(self, used_dims):
        assert all(isinstance(x, sympy.Expr) for x in used_dims)
        ordered = [
            sym
            for sym in itertools.chain(
                self.halide_vars, self.reduction_renames.values()
            )
            if sym in used_dims
        ]
        assert len(ordered) == len(used_dims)
        return ordered

    def make_index_str(self, dims, replacements=None, zero_vars=False):
        index_str = ", ".join(d.index_str(replacements, zero_vars) for d in dims)
        if len(dims) == 0:
            index_str = "()"
        elif len(dims) == 1:
            # workaround for https://github.com/halide/Halide/issues/8299
            index_str = f"{index_str},"
        return index_str

    def load(self, name: str, index: sympy.Expr):
        """Codegen a load from an InputBuffer"""
        var = self.args.input(name)
        index = self.prepare_indexing(index)
        var, dims = self.indexing_to_dimensions(var, index, False)
        line = f"{var}[{self.make_index_str(dims)}]"
        dtype = V.graph.get_dtype(name)
        if dtype in (torch.float16, torch.bfloat16):
            dtype = torch.float32
            line = f"hl.cast(hl.Float(32), {line})"

        if self._load_mask:
            assert (
                isinstance(self._load_mask, HalideCSEVariable)
                and self._load_mask.used_dims is not None
            )
            used_dims = {*self.used_dims_from_index(index), *self._load_mask.used_dims}
            result = self.newfunc(self.sort_used_dims(used_dims))
            if result.used_dims:
                self.body.writeline(f"{result.name}_mask = hl.RDom([hl.Range(0, 1)])")
                self.body.writeline(f"{result.name}_mask.where({self._load_mask})")
                other = self.kexpr(self._load_other or 0)  # type: ignore[arg-type]
                self.body.writeline(
                    f"{result} = hl.cast({halide_type(dtype)}, {other})"
                )
                self.body.writeline(
                    f"{result} = {line} + hl.cast({halide_type(dtype)}, {result.name}_mask)"
                )
            else:
                # scalar case
                self.body.writeline(
                    f"{result} = hl.select({self._load_mask}, {line}, hl.cast({halide_type(dtype)}, 0))"
                )
            return result
        else:
            return self.genfunc(line, self.used_dims_from_index(index))

    def lookup_cse_var(self, name: str):
        return self.cse.varname_map[re.sub(r"\[.*", "", name)]

    def store(
        self, name: str, index: sympy.Expr, value: CSEVariable, mode: StoreMode = None
    ) -> None:
        """Codegen a store to an OutputBuffer"""
        assert isinstance(value, HalideCSEVariable)
        var = self.args.output(name)
        index = self.prepare_indexing(index)
        var, dims = self.indexing_to_dimensions(var, index, True)
        if self.is_indirect_indexing(index) or mode is not None:
            replacements = self.setup_dom_indexing()
            index_str = self.make_index_str(dims, replacements)
            value_str = value.subs_str(replacements)
            undef_dims = (", ".join(["hl.Var()"] * len(dims))) or "()"
            self.body.writeline(
                DeferredLine(name, f"{var}[{undef_dims}] = hl.undef({var}.type())")
            )
        else:
            index_str = self.make_index_str(dims, zero_vars=True)
            value_str = str(value)

        dtype = V.graph.get_dtype(name)
        if mode is None:
            line = f"{var}[{index_str}] = hl.cast({halide_type(dtype)}, {value_str})"
        elif mode == "atomic_add":
            line = f"{var}[{index_str}] += hl.cast({halide_type(dtype)}, {value_str})"
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
        cache_key = (src_dtype, reduction_type, value)
        if cache_key in self.cse.reduction_cache:
            return self.cse.reduction_cache[cache_key]

        if isinstance(value, tuple):
            assert reduction_type == "welford_combine"
            self.cse.reduction_cache[
                cache_key
            ] = result_tuple = self.welford_combine_impl(*value)
            return result_tuple

        assert isinstance(value, HalideCSEVariable) and value.used_dims is not None
        reduction_vars = {*self.reduction_renames}
        result_var = self.newfunc(
            [v for v in value.used_dims if v not in reduction_vars]
        )
        if reduction_vars - {*value.used_dims}:
            value = self.genfunc(
                f"{value}", self.sort_used_dims({*value.used_dims, *reduction_vars})
            )
        value_str = value.subs_str(self.reduction_renames)
        default = ir.Reduction.default_accumulator(reduction_type, src_dtype)
        acc_type = halide_acc_type(dtype)

        if reduction_type in ("argmax", "argmin"):
            index = f"{result_var.name}_{reduction_type}"
            self.body.writeline(f"{index} = hl.{reduction_type}(rdom, {value_str})")
            # turn the N-D argmax index into a 1-D one
            parts = []
            stride = 1
            for i, sym in enumerate(self.reduction_renames):
                parts.append(f"{index}[{i}]")
                if stride != 1:
                    parts[-1] += f"*{stride}"
                stride *= self.halide_vars[sym]
            self.body.writeline(f"{result_var} = {' + '.join(parts)}")
        elif reduction_type == "welford_reduce":
            # TODO(jansel): implement welford_reduce without fallback
            result_var = self.welford_reduce_fallback(dtype, value)
        else:
            combine_fn = get_reduction_combine_fn(reduction_type, acc_type)
            with V.set_ops_handler(AddParenHandler(HalideOverrides(MockHandler()))):
                combine_str = combine_fn(result_var, value_str)  # type: ignore[arg-type]
            default_str = f"hl.cast({acc_type}, {halide_constant(default)})"
            self.body.writeline(f"{result_var} = {default_str}")
            self.body.writeline(f"{result_var} = {combine_str}")

        self.cse.reduction_cache[cache_key] = result_var
        return result_var

    def welford_combine_impl(self, mean, m2, weight):
        assert isinstance(mean, HalideCSEVariable) and mean.used_dims is not None
        assert isinstance(m2, HalideCSEVariable) and m2.used_dims is not None
        assert isinstance(weight, HalideCSEVariable) and weight.used_dims is not None
        used_dims = {*mean.used_dims, *m2.used_dims, *weight.used_dims} or {
            *self.halide_vars
        }
        used_dims -= {*self.reduction_renames}
        result_var = self.newfunc(self.sort_used_dims(used_dims))
        default = [f"hl.cast({x.name}.type(), 0)" for x in (mean, m2, weight)]
        pfx = result_var.name
        self.body.writeline(f"{result_var} = hl.Tuple([{', '.join(default)}])")
        self.body.writeline(f"{pfx}_mean_1 = {result_var}[0]")
        self.body.writeline(f"{pfx}_m2_1 = {result_var}[1]")
        self.body.writeline(f"{pfx}_weight_1 = {result_var}[2]")
        self.body.writeline(f"{pfx}_mean_2 = {mean.subs_str(self.reduction_renames)}")
        self.body.writeline(f"{pfx}_m2_2 = {m2.subs_str(self.reduction_renames)}")
        self.body.writeline(
            f"{pfx}_weight_2 = {weight.subs_str(self.reduction_renames)}"
        )
        self.body.writeline(f"{pfx}_delta = {pfx}_mean_2 - {pfx}_mean_1")
        self.body.writeline(f"{pfx}_new_weight = {pfx}_weight_1 + {pfx}_weight_2")
        self.body.writeline(
            f"{pfx}_w2_over_w = hl.select({pfx}_new_weight == 0.0, 0.0, {pfx}_weight_2 / {pfx}_new_weight)"
        )
        update = [
            f"{pfx}_mean_1 + {pfx}_delta * {pfx}_w2_over_w",
            f"{pfx}_m2_1 + {pfx}_m2_2 + {pfx}_delta * {pfx}_delta * {pfx}_weight_1 * {pfx}_w2_over_w",
            f"{pfx}_new_weight",
        ]
        self.body.writeline(f"{result_var} = hl.Tuple([{', '.join(update)}])")

        unpacked = []
        for i in range(3):
            unpacked.append(self.newfunc(result_var.used_dims))
            self.body.writeline(f"{unpacked[-1]} = {result_var}[{i}]")
        return tuple(unpacked)

    def scan(
        self,
        dtypes: Tuple[torch.dtype, ...],
        combine_fn: Callable[
            [Tuple[CSEVariable, ...], Tuple[CSEVariable, ...]], Tuple[CSEVariable, ...]
        ],
        values_orig: Tuple[CSEVariable, ...],
    ) -> Tuple[CSEVariable, ...]:
        assert self.inside_reduction
        assert len(dtypes) == len(values_orig)
        values: List[HalideCSEVariable] = []
        all_used_dims = set()
        for value in values_orig:
            assert isinstance(value, HalideCSEVariable) and value.used_dims is not None
            if set(value.used_dims) & set(self.reduction_renames):
                values.append(value)
            else:
                values.append(
                    self.genfunc(
                        f"{value}", [*value.used_dims, [*self.reduction_renames][:1]]
                    )
                )
            all_used_dims.update(value.used_dims)
        result_var = self.newfunc(self.sort_used_dims(all_used_dims))
        assert result_var.used_dims and set(result_var.used_dims) & set(
            self.reduction_renames
        )
        initial = [
            f"hl.cast({halide_acc_type(dtype)}, {value})"
            for dtype, value in zip(dtypes, values)
        ]

        length = self.kexpr(self.rename_indexing(self.range_trees[-1].numel))
        scan_dom = f"{result_var.name}_rdom"
        scan = f"{scan_dom}.x"
        self.body.writeline(f"{scan_dom} = hl.RDom([hl.Range(1, {length})])")

        assert (
            len(self.reduction_renames) == 1
        ), "multi-dimensional scan not implemented"
        (scan_var,) = [*self.reduction_renames]  # type: ignore[misc]
        scan_renames_cur = {scan_var: sympy_index_symbol(scan)}
        scan_renames_pri = {scan_var: sympy_index_symbol(scan) - 1}

        if len(values) == 1:

            def maybe_tuple(x):
                return x[0]

            read_left = [result_var.subs_str(scan_renames_pri)]
            read_right = [result_var.subs_str(scan_renames_cur)]
        else:

            def maybe_tuple(x):
                return f"hl.Tuple([{', '.join(x)}])"

            read_left = [
                result_var.subs_str(scan_renames_pri) + f"[{i}]"
                for i in range(len(values))
            ]
            read_right = [
                result_var.subs_str(scan_renames_cur) + f"[{i}]"
                for i in range(len(values))
            ]

        self.body.writeline(f"{result_var} = {maybe_tuple(initial)}")

        # Disable CSE for update fn
        with V.set_ops_handler(AddParenHandler(HalideOverrides(MockHandler()))):
            combine_str = combine_fn(read_left, read_right)  # type: ignore[arg-type]
        self.body.writeline(
            f"{result_var.subs_str(scan_renames_cur)} = {maybe_tuple(combine_str)}"
        )

        if len(values) == 1:
            return (result_var,)

        unpack_vars = [self.newfunc(self.sort_used_dims(all_used_dims)) for _ in values]
        for i, v in enumerate(unpack_vars):
            self.body.writeline(f"{v} = {result_var}[{i}]")
        return tuple(unpack_vars)

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

        result = []
        _, a, b, _ = self.args.python_argdefs()
        for call_str, arg in sorted(zip(a, b), key=arg_order):
            result.append((call_str, arg))
            if isinstance(arg, TensorArg):
                assert arg.offset == 0 and arg.alias_of is None
                for alias in self.buffer_aliases.get(arg.name, ()):
                    result.append(
                        (
                            None,
                            TensorArg(
                                alias,
                                arg.buffer,
                                arg.dtype,
                                arg.offset,
                                alias_of=arg.name,
                            ),
                        )
                    )
        return result

    def halide_kernel_meta(self) -> HalideMeta:
        """Compute metadata required by codecache.py"""
        argtypes = []
        for _, arg in self.halide_argdefs():
            if isinstance(arg, SizeArg):
                shape = None
                stride = None
                offset = None
                dtype = "long"
            else:
                shape = [
                    cexpr(self.rename_indexing(x.size))
                    for x in self.buffer_dimensions[arg.name]
                ]
                stride = [
                    cexpr(self.rename_indexing(x.stride))
                    for x in self.buffer_dimensions[arg.name]
                ]
                assert len(shape) == len(stride)
                offset = cexpr(self.buffer_offsets[arg.name])
                dtype = f"{DTYPE_TO_CPP[arg.dtype]}*"
            argtypes.append(
                HalideInputSpec(
                    dtype,
                    arg.name,
                    shape=shape,
                    stride=stride,
                    offset=offset,
                    alias_of=arg.alias_of,
                )
            )

        current_device = V.graph.scheduler.get_current_device_or_throw()
        if current_device.type == "cpu":
            target = [config.halide.cpu_target]
            schduler = config.halide.scheduler_cpu
            scheduler_flags = {
                "parallelism": parallel_num_threads(),
            }
            cuda_device = None
        else:
            assert current_device.type == "cuda", "only cpu/cuda supported"
            assert current_device.index <= 0, "only default device supported"
            target = [config.halide.gpu_target]
            schduler = config.halide.scheduler_cuda
            capability = torch.cuda.get_device_properties(current_device)
            if "cuda_capability" not in target[0]:
                for major, minor in [(8, 6), (8, 0), (7, 5), (7, 0), (6, 1)]:
                    if capability.major >= major and capability.minor >= minor:
                        target.append(f"cuda_capability_{major}{minor}")
                        break
            target.append("user_context")
            scheduler_flags = {
                "parallelism": capability.multi_processor_count,
                # TODO(jansel): explore other flags, see:
                # grep parser.parse ~/Halide/src/autoschedulers/anderson2021/AutoSchedule.cpp
            }
            cuda_device = max(0, current_device.index)

        # strict_float is requires for correctness
        target.append("strict_float")

        # without this we will initialize cuda once per kernel and hit errors
        target.append("no_runtime")

        if not config.halide.asserts:
            target.append("no_asserts")

        if config.halide.debug:
            target.append("debug")

        if "64" in self.index_dtype:
            # TODO(jansel): it is unclear if this does anything, since input sizes are still int32
            target.append("large_buffers")

        return HalideMeta(
            argtypes,
            target="-".join(target),
            scheduler=schduler,
            scheduler_flags=scheduler_flags,
            cuda_device=cuda_device,
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
            from torch._inductor.runtime import halide_helpers
            from math import inf, nan

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
                ndim = len(self.buffer_dimensions[arg.name])
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
        code.splice(self.indexing_code)

        def update_index(m):
            var = self.cse.varname_map[m.group(1)]
            assert var.used_dims is not None, var
            return str(var)

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
                dims = self.buffer_dimensions[arg.name]
                range_hints = []
                for i, dim in enumerate(dims):
                    hint = self._autoscheduler_workarounds(
                        V.graph.sizevars.size_hint(dim.size, fallback=1), dims
                    )
                    range_hints.append(f"hl.Range(0, {hint})")
                    if "out" not in arg.name:
                        code.writeline(f"{arg.name}.dim({i}).set_min(0)")
                        try:
                            code.writeline(
                                f"{arg.name}.dim({i}).set_stride({int(dim.stride)})"
                            )
                        except TypeError:
                            pass  # not integer
                        try:
                            code.writeline(
                                f"{arg.name}.dim({i}).set_extent({int(dim.size)})"
                            )
                        except TypeError:
                            pass  # not integer
                code.writeline(f"{arg.name}.set_estimates([{', '.join(range_hints)}])")

        code.do_unindent(2)
        code.splice(
            """
            if __name__ == "__main__":
                hl.main()
            """.rstrip(),
        )
        if meta.scheduler:
            code.splice(
                f"""
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
                """,
                strip=True,
            )
        else:
            code.splice(
                f"""
                  else:
                      with hl.GeneratorContext(hl.Target({meta.target!r})):
                          kernel = Kernel().compile_to_callable()
                  """,
                strip=True,
            )
        return code.getvalue()

    @staticmethod
    def _autoscheduler_workarounds(n, dims):
        if (
            len(dims) == 1
            and config.halide.scheduler_cuda == "Anderson2021"
            and V.graph.scheduler.get_current_device_or_throw().type == "cuda"
        ):
            # workaround https://github.com/halide/Halide/issues/8246
            n = max(2, n)
        return n

    def call_kernel(self, name: str, node=None):
        """Codegen a call to this kernel"""
        wrapper = V.graph.wrapper_code
        call_args = [f"{n}" for n, arg in self.halide_argdefs() if arg.alias_of is None]
        current_device = V.graph.scheduler.get_current_device_or_throw()
        if current_device.type == "cuda":
            stream_name = wrapper.write_get_raw_stream(current_device.index, V.graph)
            call_args.append(stream_name)
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
    kernel_type = HalideKernel  # type: ignore[arg-type]

    @classmethod
    def get_backend_features(cls, device: torch.device):
        result = dict.fromkeys(
            [
                BackendFeature.TUPLE_REDUCTION,
                BackendFeature.PREFER_STORE_LOOP_ORDER,
                BackendFeature.REDUCE_TO_SINGLE_ELEMENT,
            ]
        )
        if config.halide.scan_kernels:
            result[BackendFeature.SCAN] = None
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
