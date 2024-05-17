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
from .. import ir
from ..metrics import is_metric_table_enabled, log_kernel_metadata
from ..ops_handler import ReductionType, StoreMode

from ..runtime.hints import HalideInputSpec, HalideMeta, ReductionHint
from ..utils import (
    get_kernel_metadata,
    is_welford_reduction,
    sympy_index_symbol,
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
from .cpp import cexpr_index, DTYPE_TO_CPP
from .simd import (
    IndexingOptions,
    IterationRangesEntry,
    SIMDKernel,
    SIMDScheduling,
    triton_constant,
)

log = logging.getLogger(__name__)


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
        return cls.to_dtype(repr(value), dtype)

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
        raise Unsupported("isinf")

    @staticmethod
    def isnan(x):
        raise Unsupported("isnan")

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
        raise Unsupported("index_expr")

    @staticmethod
    def masked(mask, body, other):
        raise Unsupported("masked")


# Use mypy to check protocol implemented correctly
def _typecheck_HalideOverrides(h: HalideOverrides) -> OpsHandler[str]:
    return h


class HalideCSEVariable(CSEVariable):
    undefined_re = re.compile(r"\b(tmp\d+)\[\?\]")

    def __init__(self, name, bounds: ValueRanges[Any]):
        super().__init__(name, bounds)
        self.used_dims: Optional[List[str]] = None

    def update_on_args(self, name, args, kwargs):
        assert self.used_dims is None
        used = set()
        for arg in itertools.chain(args, kwargs.values()):
            if isinstance(arg, HalideCSEVariable):
                assert arg.used_dims is not None
                used.update(arg.used_dims)
        self.used_dims = [t.name for t in V.kernel.range_trees if t.name in used]
        assert len(self.used_dims) == len(used)

    def __str__(self):
        if self.used_dims is None:
            return f"{self.name}[?]"
        return f"{self.name}[{', '.join(self.used_dims)}]"

    def dom_str(self):
        assert self.used_dims is not None
        return f"{self.name}[{', '.join(map('{}_dom'.format, self.used_dims))}]"

    def reduction_str(self):
        assert self.used_dims is not None
        dims = [*self.used_dims]
        assert dims[-1] == "rindex"
        dims[-1] = "rindex_dom"
        return f"{self.name}[{', '.join(dims)}]"


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

    def create_cse_var(self, name, bounds=None):
        self.body.writeline(f"{name} = hl.Func()")
        return HalideCSEVariable(name, bounds)

    def codegen_iteration_ranges_entry(self, entry: IterationRangesEntry):
        expr = self.rename_indexing(entry.expr)
        self.body.writeline(f"{entry.name} = {self.kexpr(expr)}")

        expr_dom = sympy_subs(
            expr,
            {
                tree.symbol(): sympy_index_symbol(f"{tree.name}_dom")
                for tree in self.range_trees
            },
        )
        self.body.writeline(f"{entry.name}_dom = {self.kexpr(expr_dom)}")

    def used_dims_from_index(self, index: sympy.Expr):
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

    def index_to_dom(self, index: sympy.Expr):
        replacements: Dict[sympy.Expr, Any] = {}
        for sym in index.free_symbols:
            assert isinstance(sym, sympy.Symbol)
            if symbol_is_type(sym, SymT.TMP):
                # indirect indexing
                cse_var = self.cse.varname_map[sym.name]
                assert isinstance(cse_var, HalideCSEVariable)
                replacements[sym] = sympy.Symbol(cse_var.dom_str())
            elif symbol_is_type(
                sym, (SymT.UNBACKED_INT, SymT.SIZE, SymT.PRECOMPUTED_SIZE, SymT.INDEX)
            ):
                pass
            else:
                # sym is one of xN, yN or rN
                assert symbol_is_type(
                    sym, (SymT.RINDEX, SymT.XBLOCK, SymT.YBLOCK)
                ), sym.name
                replacements[sym] = sympy.Symbol(f"{sym.name}_dom")
        return sympy_subs(index, replacements)

    def load(self, name: str, index: sympy.Expr):
        var = self.args.input(name)

        indirect_indexing = self.is_indirect_indexing(index)
        if indirect_indexing:
            raise Unsupported("indirect_indexing")

        indexing = self.indexing(index, block_ptr=False)
        assert isinstance(indexing, IndexingOptions)

        has_tmpmask = indexing.has_tmpmask()
        if has_tmpmask:
            raise Unsupported("has_tmpmask")

        var = self.cse.generate(self.body, f"{var}[{indexing.index_str}]")
        assert isinstance(var, HalideCSEVariable)
        var.used_dims = self.used_dims_from_index(indexing.index)
        return var

    def store(
        self, name: str, index: sympy.Expr, value: CSEVariable, mode: StoreMode = None
    ) -> None:
        var = self.args.output(name)
        indexing = self.indexing(index, dense_indexing=True, block_ptr=False)
        assert isinstance(indexing, IndexingOptions)
        assert not indexing.has_tmpmask()

        # Halide requires the initial definition of an output to be done with a plain Var(),
        # while subsequent updates can use Expr().  For us indexing.index may be an Expr.
        # This line is a no-op to get that more flexible "update" handling.
        # TODO(jansel): try removing this when indexing.index == xindex
        line = f"{var}[hl.Var()] = hl.undef({var}.type())"
        self.body.writeline(DeferredLine(name, line))

        assert isinstance(value, HalideCSEVariable)
        value_str = value.dom_str()
        index_str = self.index_to_str(self.index_to_dom(indexing.index))

        if mode is None:
            line = f"{var}[{index_str}] = {value_str}"
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
        assert self.inside_reduction
        assert not self._load_mask

        cache_key = (src_dtype, reduction_type, value)
        if cache_key in self.cse.reduction_cache:
            return self.cse.reduction_cache[cache_key]

        acc_type = halide_acc_type(dtype)

        result_var = self.cse.newvar()
        assert isinstance(result_var, HalideCSEVariable)
        result_var.used_dims = [tree.name for tree in self.range_trees[:-1]]

        default = ir.Reduction.default_accumulator(reduction_type, src_dtype)
        assert not isinstance(default, tuple), "TODO"

        assert isinstance(value, HalideCSEVariable) and value.used_dims is not None
        if value.used_dims[-1] != "rindex":
            value = self.wrap_in_dense_index(value)
        value_str = value.reduction_str()

        self.body.writeline(
            f"{result_var} = hl.cast({acc_type}, {triton_constant(default)})"
        )

        if reduction_type in {"argmax", "argmin"}:
            raise Unsupported(reduction_type)
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
        dense = [tree.name for tree in self.range_trees]
        if not self.inside_reduction:
            removed = dense.pop()
            assert removed == "rindex"
        if var.used_dims == dense:
            return var
        newvar = self.cse.generate(self.body, f"{var}")
        assert isinstance(newvar, HalideCSEVariable)
        newvar.used_dims = dense
        return newvar

    def halide_kernel_meta(self):
        _, _, signature = self.args.python_argdefs()
        argtypes = []
        for arg in signature:
            numel = cexpr_index(self.rename_indexing(V.graph.get_numel(arg.buffer)))
            dtype = f"{DTYPE_TO_CPP[arg.dtype]}*"
            argtypes.append(
                HalideInputSpec(
                    dtype,
                    arg.name,
                    numel,
                )
            )
        return HalideMeta(argtypes)

    def codegen_kernel(self, name=None):
        self.halide_kernel_meta()  # ensure needed args are added
        code = IndentedBuffer()
        code.splice(
            """
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
        for tree in self.range_trees:
            if tree.prefix != "r" or self.inside_reduction:
                code.writeline(f"{tree.name} = hl.Var({tree.name!r})")
                # Use this version for stores
                length = self.kexpr(self.rename_indexing(tree.numel))
                code.writeline(
                    f"{tree.name}_dom = hl.RDom([hl.Range(0, {length})], {tree.name!r}).x"
                )
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
        return code.getvalue()

    def call_kernel(self, name: str, node=None):
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
