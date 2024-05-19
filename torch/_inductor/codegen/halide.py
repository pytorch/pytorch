from __future__ import annotations

import itertools
import logging
import math
import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import sympy

import torch
import torch._logging
from ..._prims_common import is_integer_dtype
from ...utils._sympy.symbol import symbol_is_type, SymT
from ...utils._sympy.value_ranges import ValueRanges
from .. import config, ir
from ..codecache import HalideCodeCache
from ..metrics import is_metric_table_enabled, log_kernel_metadata
from ..ops_handler import ReductionType, StoreMode

from ..runtime.hints import HalideInputSpec, HalideMeta, ReductionHint
from ..utils import (
    get_bounds_index_expr,
    get_kernel_metadata,
    is_welford_reduction,
    parallel_num_threads,
    sympy_dot,
    sympy_index_symbol,
    sympy_product,
    sympy_subs,
)
from ..virtualized import _ops as ops, OpsHandler, V
from .common import (
    CSEVariable,
    DeferredLine,
    IndentedBuffer,
    OpOverrides,
    PythonPrinter,
)
from .cpp import DTYPE_TO_CPP
from .cpp_utils import cexpr
from .simd import (
    IndexingOptions,
    IterationRangesEntry,
    SIMDKernel,
    SIMDScheduling,
    triton_constant,
)

log = logging.getLogger(__name__)


def halide_constant(val):
    if isinstance(val, int) and not (-2147483648 <= val <= 2147483647):
        # workaround https://github.com/halide/Halide/issues/8224
        info = torch.iinfo(torch.int64)
        if val == info.min:
            return "hl.Int(64).min()"
        if val == info.max:
            return "hl.Int(64).max()"
        raise Unsupported("int64 constant")

    return triton_constant(val)


class Unsupported(RuntimeError):
    def __init__(self, thing):
        super().__init__(f"halide backend does not support: {thing}")


class HalidePrinter(PythonPrinter):
    @staticmethod
    def cast_index(expr):
        # TODO(jansel)
        if V.kernel.index_dtype == torch.int32:
            return f"hl.cast(hl.Int(32), {expr})"
        if V.kernel.index_dtype == torch.int64:
            return f"hl.cast(hl.Int(64), {expr})"
        raise AssertionError("not implemented: %s", V.kernel.index_dtype)

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
        return f"hl.abs({self._print(expr.args[0])})"

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

    def _print_RoundDecimal(self, expr):
        raise Unsupported("_print_RoundDecimal")


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
}


def halide_type(dtype):
    if dtype == torch.bfloat16:
        raise Unsupported("torch.bfloat16")
    return _halide_type[dtype]


def halide_acc_type(dtype):
    if is_integer_dtype(dtype) and dtype.is_signed and dtype != torch.int64:
        dtype = torch.int32
    return halide_type(dtype)


class HalideOverrides(OpOverrides):
    @staticmethod
    def to_dtype(x, dtype: torch.dtype, src_dtype: Optional[torch.dtype] = None):
        if dtype == torch.bool:
            return f"({x} != 0)"
        return f"hl.cast({halide_type(dtype)}, {x})"

    @staticmethod
    def to_dtype_bitcast(x, dtype: torch.dtype, src_dtype: torch.dtype):
        return f"hl.reinterpret({halide_type(dtype)}, {x})"

    @classmethod
    def constant(cls, value, dtype):
        return cls.to_dtype(halide_constant(value), dtype)

    @staticmethod
    def abs(x):
        return f"hl.abs({x})"

    @staticmethod
    def libdevice_abs(x):
        return f"hl.abs({x})"

    @staticmethod
    def exp(x):
        return f"hl.fast_exp({x})"

    @staticmethod
    def libdevice_exp(x):
        return f"hl.exp({x})"

    @staticmethod
    def exp2(x):
        return f"hl.exp2({x})"

    @staticmethod
    def expm1(x):
        return f"hl.expm1({x})"

    @staticmethod
    def sqrt(x):
        return f"hl.fast_sqrt({x})"

    @staticmethod
    def libdevice_sqrt(x):
        return f"hl.sqrt({x})"

    @staticmethod
    def relu(x):
        return f"hl.max(0, {x})"

    @staticmethod
    def minimum(a, b):
        return f"hl.min({a}, {b})"

    @staticmethod
    def maximum(a, b):
        return f"hl.max({a}, {b})"

    @staticmethod
    def where(a, b, c):
        return f"hl.select({a}, {b}, {c})"

    @staticmethod
    def cos(x):
        return f"hl.fast_cos({x})"

    @staticmethod
    def libdevice_cos(x):
        return f"hl.cos({x})"

    @staticmethod
    def sin(x):
        return f"hl.fast_sin({x})"

    @staticmethod
    def libdevice_sin(x):
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
    def erfc(x):
        raise Unsupported("erfc")

    @staticmethod
    def erfinv(x):
        raise Unsupported("erfinv")

    @staticmethod
    def hypot(x, y):
        return f"hl.hypot({x}, {y})"

    @staticmethod
    def log10(x):
        return f"hl.fast_log({x}) * {1/math.log(10)!r}"

    @staticmethod
    def log2(x):
        return f"hl.fast_log({x}) * {1/math.log(2)!r}"

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
        return f"hl.fast_inverse_sqrt({x})"

    @staticmethod
    def log1p(x):
        return f"hl.fast_log(({x}) + 1)"

    @staticmethod
    def tan(x):
        return f"hl.tan({x})"

    @staticmethod
    def tanh(x):
        return f"hl.tanh({x})"

    @staticmethod
    def sigmoid(x):
        return f"1./(1. + hl.fast_exp(-({x})))"

    @staticmethod
    def libdevice_sigmoid(x):
        return f"1./(1. + hl.exp(-({x})))"

    @staticmethod
    def signbit(x):
        raise Unsupported("signbit")

    @staticmethod
    def fmod(a, b):
        raise Unsupported("fmod")

    @staticmethod
    def pow(a, b):
        return f"hl.fast_pow({a}, {b})"

    @staticmethod
    def log(x):
        return f"hl.fast_log({x})"

    @staticmethod
    def libdevice_log(x):
        return f"hl.log({x})"

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
    def floordiv(a, b):
        # See the comment in lowering.div_mode. a and b are integer type.
        # Similar to div_floor_kernel_cuda in pytorch core.
        # Notice that // in triton behaves as truncdiv instead of floordiv
        quot = f"{a} // {b}"
        rem = f"{a} % {b}"
        return f"hl.select(({a} < 0) != ({b} < 0), hl.select({rem} != 0, {quot} - 1, {quot}), {quot})"

    @classmethod
    def sign(cls, x):
        left = ops.to_dtype(ops.lt("0", x), torch.int8)
        right = ops.to_dtype(ops.lt(x, "0"), torch.int8)
        sub = ops.sub(left, right)
        return f"hl.cast(({x}).type(), {sub})"

    @staticmethod
    def trunc(x):
        return f"hl.trunc({x})"

    @staticmethod
    def truncdiv(a, b):
        # See the comment in lowering.div_mode. a and b are integer type.
        # Notice that // in triton behaves as truncdiv instead of floordiv
        return f"{a} // {b}"

    @staticmethod
    def ceil(x):
        return f"hl.ceil({x})"

    @classmethod
    def index_expr(cls, expr, dtype):
        indexing = V.kernel.indexing(expr, block_ptr=False)
        assert isinstance(indexing, IndexingOptions)
        var = V.kernel.genfunc(
            indexing.index_str,
            V.kernel.used_dims_from_index(indexing.index),
            bounds=get_bounds_index_expr(expr),
        )
        if dtype not in {torch.int32, torch.int64}:
            return ops.to_dtype(var, dtype)
        return var

    @staticmethod
    def masked(mask, body, other):
        with V.kernel.mask_loads(mask) as new_mask:
            result = body()

        if result.bounds.is_bool:
            other = bool(other)

        # Take dtype from result to prevent accidental promotion
        other = V.kernel.genfunc(
            f"hl.cast({result.name}.type(), {halide_constant(other)})",
            [],
            bounds=ValueRanges.wrap(other),
        )
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
        return f"{self.name}[{', '.join(map(str, dims))}]"

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
        self.indexing_code = self.body
        self.has_reduction = self.inside_reduction

    def create_cse_var(self, name, bounds=None):
        self.body.writeline(f"{name} = hl.Func()")
        return HalideCSEVariable(name, bounds)

    def codegen_iteration_ranges_entry(self, entry: IterationRangesEntry):
        expr = self.rename_indexing(entry.expr)
        self.body.writeline(f"{entry.name} = {self.kexpr(expr)}")

        if self.has_reduction:
            # idom includes iteration ranges of the numel of inputs
            expr_idom = sympy_subs(
                expr,
                {
                    tree.symbol(): sympy_index_symbol(f"{tree.name}_idom")
                    for tree in self.range_trees
                },
            )
            self.body.writeline(f"{entry.name}_idom = {self.kexpr(expr_idom)}")

        if entry.prefix != "r":
            # idom includes iteration ranges of the numel of outputs (which is different for reductions)
            expr_idom = sympy_subs(
                expr,
                {
                    tree.symbol(): sympy_index_symbol(f"{tree.name}_odom")
                    for tree in self.range_trees
                },
            )
            self.body.writeline(f"{entry.name}_odom = {self.kexpr(expr_idom)}")

    def used_dims_from_index(self, index: sympy.Expr):
        """Detect which range trees are used to populate HalideCSEVariable.used_dims"""
        used_dims = set()
        for sym in index.free_symbols:
            assert isinstance(sym, sympy.Symbol)
            if symbol_is_type(sym, SymT.TMP):
                # indirect indexing
                cse_var = self.cse.varname_map[sym.name]
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

        indirect_indexing = self.is_indirect_indexing(index)
        if indirect_indexing:
            raise Unsupported("indirect_indexing")

        indexing = self.indexing(index, block_ptr=False)
        assert isinstance(indexing, IndexingOptions)
        index_str = self.index_to_str(indexing.index)

        if indexing.has_tmpmask():
            # Halide doesn't have a great way to do masked loads
            var = f"hl.BoundaryConditions.constant_exterior({var}, 0)"
            # TODO(jansel): figure out why this didn't work
            # mask, = [m for m in indexing.mask_vars if 'tmp' in str(m)]
            # index_str = f"hl.select({mask}, {index_str}, 0)"

        return self.genfunc(
            f"{var}[{index_str}]", self.used_dims_from_index(indexing.index)
        )

    def index_to_dom(self, index: sympy.Expr, suffix: str):
        """Replace xindex => xindex_dom, x0 => x0_dom, etc for update-style indexing"""
        replacements: Dict[sympy.Expr, Any] = {}
        for sym in index.free_symbols:
            assert isinstance(sym, sympy.Symbol)
            if symbol_is_type(sym, SymT.TMP):
                # indirect indexing
                cse_var = self.cse.varname_map[sym.name]
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

    def determine_store_indexing(
        self, name: str, index: sympy.Expr, value: HalideCSEVariable, var: str
    ):
        """
        Halide requires the initial definition of an output to be done with a plain Var(), while subsequent updates
        can use Expr().  For us index may be an Expr.

        This function tries to make the output index a var, and if that fails switches to the more flexible
        hl.RDom()-update codegen.        .
        """
        assert value.used_dims is not None
        eq = V.graph.sizevars.statically_known_equals

        if index == 0 and eq(self.halide_buffer_numel(name), 1):
            # 1-element case
            index_str = "hl.Var()"  # halide requires storage dst to be a Var
            value_str = value.index_str([0 for _ in value.used_dims])
            return index_str, value_str

        var_ranges = self.var_ranges()
        range_trees = self.active_range_trees()
        numel = self.halide_buffer_numel(name)

        if isinstance(index, sympy.Symbol) and eq(var_ranges[index], numel):
            value_str = str(value)
            index_str = self.index_to_str(index)
            return index_str, value_str

        # collapse 2D tile into 1D output
        if len(range_trees) == 2:
            used_vars = set(index.free_symbols)
            var_ranges = {s: v for s, v in var_ranges.items() if s in used_vars}
            if len(var_ranges) == 2 and eq(sympy_product(var_ranges.values()), numel):
                strides = V.graph.sizevars.stride_vars(index, var_ranges)
                if eq(sympy_dot(var_ranges, strides), index):
                    index_str = f"{var}_index"
                    self.body.writeline(
                        DeferredLine(name, f"{index_str} = hl.Var({index_str!r})")
                    )
                    v0, v1 = var_ranges
                    assert v0.name[0] == range_trees[0].prefix
                    assert v1.name[0] == range_trees[1].prefix
                    assert eq(var_ranges[v0], range_trees[0].numel)
                    assert eq(var_ranges[v1], range_trees[1].numel)

                    if eq(strides[0], 1):
                        div = self.kexpr(strides[1])
                        value_str = (
                            f"{value.name}[{index_str} % {div}, {index_str} // {div}]"
                        )
                    else:
                        assert eq(strides[1], 1)
                        div = self.kexpr(strides[0])
                        value_str = (
                            f"{value.name}[{index_str} // {div}, {index_str} % {div}]"
                        )
                    return index_str, value_str

        # Fall back to using RDom-style store
        self.body.writeline(
            DeferredLine(name, f"{var}[hl.Var()] = hl.undef({var}.type())")
        )
        suffix = "idom" if self.inside_reduction else "odom"
        value_str = value.with_dom(suffix)
        index_str = self.index_to_str(self.index_to_dom(index, suffix))
        return index_str, value_str

    def store(
        self, name: str, index: sympy.Expr, value: CSEVariable, mode: StoreMode = None
    ) -> None:
        """Codegen a store to an OutputBuffer"""
        var = self.args.output(name)
        indexing = self.indexing(index, dense_indexing=True, block_ptr=False)
        assert isinstance(indexing, IndexingOptions)
        assert not indexing.has_tmpmask()
        assert isinstance(value, HalideCSEVariable)

        index_str, value_str = self.determine_store_indexing(
            name, indexing.index, value, var
        )

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

        cache_key = (src_dtype, reduction_type, value)
        if cache_key in self.cse.reduction_cache:
            return self.cse.reduction_cache[cache_key]

        acc_type = halide_acc_type(dtype)
        result_var = self.newfunc([tree.name for tree in self.range_trees[:-1]])

        default = ir.Reduction.default_accumulator(reduction_type, src_dtype)
        assert not isinstance(default, tuple), "TODO"

        assert isinstance(value, HalideCSEVariable) and value.used_dims is not None
        if value.used_dims[-1] != "rindex":
            value = self.wrap_in_dense_index(value)
        value_str = value.reduction_str()

        if reduction_type not in {"argmax", "argmin"}:
            self.body.writeline(
                f"{result_var} = hl.cast({acc_type}, {halide_constant(default)})"
            )

        if reduction_type in {"argmax", "argmin"}:

            def cast_tuple(a, b):
                return f"hl.Tuple([hl.cast({halide_type(src_dtype)}, {a}), hl.cast({halide_type(dtype)}, {b})])"

            tuple_var = self.newfunc(result_var.used_dims)
            self.body.writeline(
                f"{tuple_var} = {cast_tuple(halide_constant(default), 0)}"
            )
            cmp = ">" if reduction_type == "argmax" else "<"
            better = f"{value_str} {cmp} {tuple_var}[0]"
            self.body.writeline(
                f"{tuple_var} = hl.select({better}, {cast_tuple(value_str, 'rdom')}, {tuple_var})"
            )
            self.body.writeline(f"{result_var} = {tuple_var}[1]")
        elif is_welford_reduction(reduction_type):
            raise Unsupported(reduction_type)
        elif reduction_type == "sum":
            self.body.writeline(f"{result_var} += {value_str}")
        elif reduction_type == "any":
            self.body.writeline(f"{result_var} |= {value_str}")
        elif reduction_type == "prod":
            self.body.writeline(f"{result_var} *= {value_str}")
        elif reduction_type == "min":
            self.body.writeline(f"{result_var} = hl.min({result_var}, {value_str})")
        elif reduction_type == "max":
            self.body.writeline(f"{result_var} = hl.max({result_var}, {value_str})")
        elif reduction_type == "xor_sum":
            self.body.writeline(f"{result_var} = {result_var} ^ {value_str}")
        else:
            raise Unsupported(reduction_type)

        # for any
        # if src_dtype == torch.bool:
        #     prior = result_var
        #     result_var: HalideCSEVariable = self.cse.newvar()
        #     result_var.used_dims = prior.used_dims
        #     self.body.writeline(f"{result_var} = hl.cast({halide_type(src_dtype)}, {prior})")

        self.cse.reduction_cache[cache_key] = result_var
        return result_var

    def wrap_in_dense_index(self, var: HalideCSEVariable) -> HalideCSEVariable:
        dense = [tree.name for tree in self.active_range_trees()]
        if var.used_dims is not None and var.used_dims == dense:
            return var
        return self.genfunc(f"{var}", dense)

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

    def halide_kernel_meta(self) -> HalideMeta:
        """Compute metadata required by codecache.py"""
        _, _, signature = self.args.python_argdefs()
        argtypes = []
        for arg in signature:
            numel = cexpr(self.rename_indexing(self.halide_buffer_numel(arg.buffer)))
            dtype = f"{DTYPE_TO_CPP[arg.dtype]}*"
            argtypes.append(
                HalideInputSpec(
                    dtype,
                    arg.name,
                    numel,
                )
            )
        target = "host"
        # cuda_capability_86
        # for cuda: target="host-cuda-cuda_capability_86-user_context"
        if config.halide.no_asserts:
            target += "-no_asserts"

        return HalideMeta(
            argtypes,
            target=target,
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
        self.halide_kernel_meta()  # ensure needed args are added
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

        _, _, signature = self.args.python_argdefs()
        for arg in signature:
            assert arg.buffer, "TODO"
            argcls = "hl.OutputBuffer" if "out" in arg.name else "hl.InputBuffer"
            argtype = halide_type(arg.dtype)
            code.writeline(f"{arg.name} = {argcls}({argtype}, 1)")
        code.splice(
            """
            def generate(g):
        """
        )
        code.do_indent()
        for arg in signature:
            code.writeline(f"{arg.name} = g.{arg.name}")

        dom_size = {}
        for tree in self.active_range_trees(reorder=True):
            code.writeline(f"{tree.name} = hl.Var({tree.name!r})")
            length = self.kexpr(self.rename_indexing(tree.numel))
            dom_size[tree.name] = f"hl.Range(0, {length})"
        assert len(dom_size) <= 3

        if self.inside_reduction:
            sizes = [*dom_size.values()]
            code.writeline(f"idom = hl.RDom([{', '.join(sizes)}])")
            code.writeline(f"odom = hl.RDom([{', '.join(sizes[:-1])}])")
            code.writeline(f"rdom = hl.RDom([{sizes[-1]}])")
            for name, xyz in zip(dom_size.keys(), "xyz"):
                code.writeline(f"{name}_idom = idom.{xyz}")
                if name[0] != "r":
                    code.writeline(f"{name}_odom = odom.{xyz}")
        else:
            code.writeline(f"odom = hl.RDom([{', '.join(dom_size.values())}])")
            for name, xyz in zip(dom_size.keys(), "xyz"):
                code.writeline(f"{name}_odom = odom.{xyz}")

        for old, new in self.args.aliases():
            code.writeline(f"{old} = {new}")

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
        for arg in signature:
            numel = V.graph.sizevars.symbolic_hint(V.graph.get_numel(arg.buffer))
            try:
                low = high = int(numel)
            except TypeError:
                low, high = 0, 8192  # arbitrary range for unbacked symints
            code.writeline(f"{arg.name}.set_estimates([hl.Range({low}, {high})])")

        code.do_unindent(2)
        code.writeline("")
        code.writeline("__name__ == '__main__' and hl.main()")
        return code.getvalue()

    def call_kernel(self, name: str, node=None):
        """Codegen a call to this kernel"""
        wrapper = V.graph.wrapper_code
        _, _, signature = self.args.python_argdefs()
        call_args = []
        for arg in signature:
            call_args.append(arg.buffer)

        current_device = V.graph.scheduler.current_device
        assert current_device.type == "cpu"
        wrapper.generate_kernel_call(
            name,
            call_args,
            cuda=False,
        )


class HalideScheduling(SIMDScheduling):
    int32_type = "hl.Int(32)"
    int64_type = "hl.Int(64)"
    kernel_type = HalideKernel

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
