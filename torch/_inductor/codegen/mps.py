# This is not a feature-complete compiler backend
# Just an early prototype that shows that one can compile elementwise ops into a Metal shader
from __future__ import annotations

import functools
import itertools
import logging
import math
from typing import Any, Optional, TYPE_CHECKING

import sympy
from sympy.printing.precedence import PRECEDENCE

import torch
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.printers import ExprPrinter as ExprPrinter_
from torch.utils._sympy.value_ranges import ValueRanges

from ..utils import ceildiv, get_bounds_index_expr, get_kernel_metadata
from ..virtualized import ops, OpsWrapper, V
from .common import (
    CSEVariable,
    DeferredLine,
    DTYPE_TO_COMPUTATION_DTYPE,
    IndentedBuffer,
    OpOverrides,
    PythonPrinter,
)
from .simd import IterationRangesEntry, SIMDKernel, SIMDScheduling


if TYPE_CHECKING:
    from typing import Union

    from ..ops_handler import ReductionType, StoreMode
    from ..scheduler import Scheduler, SchedulerNode
    from .common import OpVarT

log = logging.getLogger(__name__)

DTYPE_TO_METAL = {
    torch.bool: "bool",
    torch.int8: "char",
    torch.int16: "short",
    torch.int32: "int",
    torch.int64: "long",
    torch.uint8: "uchar",
    torch.float: "float",
    torch.half: "half",
    torch.bfloat16: "bfloat",
}


def value_to_metal(val: Union[float, int, bool, str, CSEVariable]) -> str:
    if isinstance(val, float):
        if val == torch.inf:
            return "HUGE_VALF"
        elif val == -torch.inf:
            return "-HUGE_VALF"
        elif val != val:  # Only float that not equal to self is nan
            return "NAN"
        return str(val)
    elif isinstance(val, bool):
        return "true" if val else "false"
    return str(val)


class MetalExprPrinter(ExprPrinter_):
    """Converts sympy expression to Metal code snippet"""

    def _print_FloorDiv(self, expr: sympy.Expr) -> str:
        x, div = expr.args
        x = self.doprint(x)
        div = self.doprint(div)
        if expr.is_integer:
            return f"({x}) / ({div})"
        return f"metal::floor({x}) / ({div})"

    def _print_ModularIndexing(self, expr: sympy.Expr) -> str:
        x, div, mod = expr.args
        x = self.doprint(x)
        if div != 1:
            div = self.doprint(div)
            if expr.is_integer:
                x = f"({x}) / ({div})"
            else:
                x = f"metal::floor({x}) / ({div})"
        mod = self.doprint(mod)
        return f"({x}) % ({mod})"

    def _print_Min(self, expr: sympy.Expr) -> str:
        if len(expr.args) != 2:
            raise RuntimeError("metal::min only supported for 2 args")
        a, b = map(self._print, expr.args)
        typecast_a = f"static_cast<decltype({a}+{b})>({a})"
        typecast_b = f"static_cast<decltype({a}+{b})>({b})"
        return f"metal::min({typecast_a}, {typecast_b})"

    def _print_Max(self, expr: sympy.Expr) -> str:
        if len(expr.args) != 2:
            raise RuntimeError("metal::max only supported for 2 args")
        a, b = map(self._print, expr.args)
        typecast_a = f"static_cast<decltype({a}+{b})>({a})"
        typecast_b = f"static_cast<decltype({a}+{b})>({b})"
        return f"metal::max({typecast_a}, {typecast_b})"

    def _print_Abs(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 1
        return f"metal::abs({self._print(expr.args[0])})"

    def _print_RoundToInt(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 1
        return f"static_cast<long>(metal::rint({self._print(expr.args[0])}))"

    def _print_RoundDecimal(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 2
        number, ndigits = expr.args
        if number.is_integer:
            # ndigits < 0 should have been filtered by the sympy function
            assert ndigits < 0
            raise ValueError(
                f"For integer inputs, only non-negative ndigits are currently supported, but got {ndigits}."
            )
        number_str = self.parenthesize(number, PRECEDENCE["Mul"])
        return f"static_cast<float>(metal::rint(1e{ndigits} * {number_str}) * 1e{-ndigits})"

    def _print_IntTrueDiv(self, expr: sympy.Expr) -> str:
        lhs, rhs = expr.args
        # TODO: This is only accurate up to 2**23
        return f"static_cast<float>({self._print(lhs)}) / static_cast<float>({self._print(rhs)})"

    def _print_PowByNatural(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 2
        x, y = map(self.doprint, expr.args)
        return f"metal::pow(static_cast<float>({x}), static_cast<float>({y}))"

    def _print_ToFloat(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 1
        x = self.doprint(expr.args[0])
        return f"static_cast<float>({x})"

    def _print_FloorToInt(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 1
        x = self.doprint(expr.args[0])
        return f"static_cast<int>(metal::floor({x}))"

    def _print_TruncToInt(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 1
        x = self.doprint(expr.args[0])
        return f"static_cast<int>(metal::trunc({x}))"

    def _print_OpaqueUnaryFn_log2(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 1
        x = self.doprint(expr.args[0])
        return f"metal::log2({x})"


class MetalOverrides(OpOverrides):
    """Implements Metal-specific overrids for ops. Base class emits Python-friendly overrides"""

    @staticmethod
    def to_dtype(
        x: CSEVariable,
        dtype: torch.dtype,
        src_dtype: Optional[torch.dtype] = None,
        use_compute_types: bool = True,
    ) -> str:
        if dtype == torch.double:
            log.warning(
                "float64 cast requested, probably from tensorify_python_scalars"
            )
            return f"static_cast<float>({x})"
        return f"static_cast<{DTYPE_TO_METAL[dtype]}>({x})"

    @staticmethod
    def to_dtype_bitcast(
        x: CSEVariable, dtype: torch.dtype, src_dtype: torch.dtype
    ) -> str:
        return f"as_type<{DTYPE_TO_METAL[dtype]}>(static_cast<{DTYPE_TO_METAL[src_dtype]}>({x}))"

    @staticmethod
    def constant(val: Union[bool, float, int], dtype: torch.dtype) -> str:
        return value_to_metal(val)

    @staticmethod
    def index_expr(expr: sympy.Expr, dtype: torch.dtype) -> str:
        idx_str = V.kernel.index_to_str(V.kernel.prepare_indexing(expr))
        var = V.kernel.cse.generate(
            V.kernel.compute, idx_str, bounds=get_bounds_index_expr(expr)
        )
        return ops.to_dtype(var, dtype)

    @staticmethod
    def masked(mask: CSEVariable, body: sympy.Expr, other: CSEVariable) -> str:
        # TODO: Type annotation for other is wrong, it's often float or int
        with V.kernel.mask_loads(mask, other) as new_mask:
            result = body()

        if result.bounds.is_bool:
            other = bool(other)  # type: ignore[assignment]

        return ops.where(new_mask, result, other)

    @staticmethod
    def where(a: OpVarT, b: OpVarT, c: OpVarT) -> str:
        return f"{a} ? {b} : {value_to_metal(c)}"

    @staticmethod
    def remainder(a: OpVarT, b: OpVarT) -> str:
        if (
            isinstance(b, CSEVariable)
            and b.dtype is not None
            and not b.dtype.is_floating_point
        ):
            return f"{a} % {b}"
        # Upcast to float otherwise results of remainder op are wrong for half
        float_a = (
            f"static_cast<float>({a})"
            if isinstance(a, CSEVariable) and a.dtype != torch.float
            else a
        )
        float_b = (
            f"static_cast<float>({b})"
            if isinstance(b, CSEVariable) and b.dtype != torch.float
            else b
        )
        return f"{float_a} - {float_b} * metal::floor({float_a} / {float_b})"

    @staticmethod
    def maximum(a: CSEVariable, b: CSEVariable) -> str:
        typecast_a = f"static_cast<decltype({a}+{b})>({a})"
        typecast_b = f"static_cast<decltype({a}+{b})>({b})"
        return f"c10::metal::max({typecast_a}, {typecast_b})"

    @staticmethod
    def minimum(a: CSEVariable, b: CSEVariable) -> str:
        typecast_a = f"static_cast<decltype({a}+{b})>({a})"
        typecast_b = f"static_cast<decltype({a}+{b})>({b})"
        return f"c10::metal::min({typecast_a}, {typecast_b})"

    @staticmethod
    def logical_or(a: CSEVariable, b: CSEVariable) -> str:
        return f"{a} || {b}"

    @staticmethod
    def logical_and(a: CSEVariable, b: CSEVariable) -> str:
        return f"{a} && {b}"

    @staticmethod
    def isnan(x: CSEVariable) -> str:
        return f"metal::isnan({x})"

    @staticmethod
    def isinf(x: CSEVariable) -> str:
        return f"metal::isinf({x})"

    @staticmethod
    def log(x: CSEVariable) -> str:
        return f"metal::log({x})"

    @staticmethod
    def exp(x: CSEVariable) -> str:
        return f"metal::exp({x})"

    @staticmethod
    def abs(x: CSEVariable) -> str:
        return f"metal::abs({x})"

    @staticmethod
    def signbit(x: CSEVariable) -> str:
        return f"metal::signbit({x})"

    @staticmethod
    def sin(x: CSEVariable) -> str:
        return f"metal::precise::sin({x})"

    @staticmethod
    def sinc(x: CSEVariable) -> str:
        return f"c10::metal::sinc({x})"

    @staticmethod
    def cos(x: CSEVariable) -> str:
        return f"metal::precise::cos({x})"

    @staticmethod
    def tan(x: CSEVariable) -> str:
        return f"metal::tan({x})"

    @staticmethod
    def asin(x: CSEVariable) -> str:
        return f"metal::asin({x})"

    @staticmethod
    def acos(x: CSEVariable) -> str:
        return f"metal::acos({x})"

    @staticmethod
    def atan(x: CSEVariable) -> str:
        return f"metal::atan({x})"

    @staticmethod
    def atan2(x: CSEVariable, y: CSEVariable) -> str:
        return f"::metal::atan2({x}, {y})"

    @staticmethod
    def sqrt(x: CSEVariable) -> str:
        return f"metal::sqrt({x})"

    @staticmethod
    def neg(x: CSEVariable) -> str:
        # TODO: Does it rely on undefined behavior?
        # If so, add special logic for unsigned types
        return f"static_cast<decltype({x})>(-{x})"

    @staticmethod
    def rsqrt(x: CSEVariable) -> str:
        return f"metal::rsqrt({x})"

    @staticmethod
    def tanh(x: CSEVariable) -> str:
        return f"metal::tanh({x})"

    @staticmethod
    def atanh(x: CSEVariable) -> str:
        return f"metal::atanh({x})"

    @staticmethod
    def floordiv(a: CSEVariable, b: CSEVariable) -> str:
        # a and b are integer type
        quot = f"{a} / {b}"
        rem = f"{a} % {b}"
        return f"(({a} < 0) != ({b} < 0) ? ({rem} != 0 ? {quot} - 1 : {quot}) : {quot})"

    @staticmethod
    def floor(x: CSEVariable) -> str:
        return f"metal::floor({x})"

    @staticmethod
    def sign(x: CSEVariable) -> str:
        return f"metal::sign({x})"

    @staticmethod
    def fmod(a: CSEVariable, b: CSEVariable) -> str:
        typecast_a = f"static_cast<decltype({a}+{b})>({a})"
        typecast_b = f"static_cast<decltype({a}+{b})>({b})"
        return f"metal::fmod({typecast_a}, {typecast_b})"

    @staticmethod
    def trunc(x: CSEVariable) -> str:
        return f"metal::trunc({x})"

    @staticmethod
    def truncdiv(a: CSEVariable, b: CSEVariable) -> str:
        quot = f"{a} / {b}"
        if (a.dtype is not None and a.dtype.is_floating_point) or (
            b.dtype is not None and b.dtype.is_floating_point
        ):
            return f"metal::trunc({quot})"
        return quot

    @staticmethod
    def ceil(x: CSEVariable) -> str:
        return f"metal::ceil({x})"

    @staticmethod
    def rand(seed: CSEVariable, offset: CSEVariable) -> str:
        V.kernel.headers.add("random")
        return f"c10::metal::rand({seed}, {offset})"

    @staticmethod
    def randn(seed: CSEVariable, offset: CSEVariable) -> str:
        V.kernel.headers.add("random")
        return f"c10::metal::randn({seed}, {offset})"

    @staticmethod
    def randint64(
        seed: CSEVariable, offset: CSEVariable, low: CSEVariable, high: CSEVariable
    ) -> str:
        V.kernel.headers.add("random")
        return f"c10::metal::randint64({seed}, {offset}, {low}, {high})"

    @staticmethod
    def round(x: CSEVariable) -> str:
        return f"metal::round({x})"

    @staticmethod
    def pow(a: CSEVariable, b: CSEVariable) -> str:
        cast_a = f"static_cast<decltype({a}+{b})>({a})"
        cast_b = f"static_cast<decltype({a}+{b})>({b})"
        return f"metal::pow({cast_a}, {cast_b})"

    def _special_unary(self, a: CSEVariable, name: str) -> str:
        V.kernel.headers.add("special_math")
        return f"c10::metal::{name}({a})"

    def _special_binary(self, a: CSEVariable, b: CSEVariable, name: str) -> str:
        V.kernel.headers.add("special_math")
        return f"c10::metal::{name}({a}, {b})"

    @classmethod
    def _initialize_special_ops(cls) -> None:
        # Unary special ops
        for name in [
            "erf",
            "erfinv",
            "i0",
            "i0e",
            "i1",
            "i1e",
            "digamma",
            "spherical_bessel_j0",
        ]:
            setattr(cls, name, functools.partialmethod(cls._special_unary, name=name))

        cls.lgamma = functools.partialmethod(cls._special_unary, name="log_gamma")  # type: ignore[assignment]

        # Unary special ops with forward in method name
        for name in [
            "bessel_j0",
            "bessel_j1",
            "bessel_y0",
            "bessel_y1",
            "modified_bessel_i0",
            "modified_bessel_i1",
            "modified_bessel_k0",
            "modified_bessel_k1",
            "scaled_modified_bessel_k0",
            "scaled_modified_bessel_k1",
        ]:
            setattr(
                cls,
                name,
                functools.partialmethod(cls._special_unary, name=name + "_forward"),
            )

        # Binary special ops
        for name in [
            "polygamma",
            "zeta",
        ]:
            setattr(cls, name, functools.partialmethod(cls._special_binary, name=name))

        # Binary special ops with forward in method name
        for name in [
            "chebyshev_polynomial_t",
            "chebyshev_polynomial_u",
            "chebyshev_polynomial_v",
            "chebyshev_polynomial_w",
            "hermite_polynomial_h",
            "hermite_polynomial_he",
        ]:
            setattr(
                cls,
                name,
                functools.partialmethod(cls._special_binary, name=name + "_forward"),
            )


MetalOverrides._initialize_pointwise_overrides("mps")
MetalOverrides._initialize_special_ops()


class MetalKernel(SIMDKernel):
    """Implement Metal codegen based on the SIMDKernel abstraction"""

    overrides = MetalOverrides  # type: ignore[assignment]
    suffix = ";"
    newvar_prefix = "auto "
    max_threadgroup_size = 1024
    simd_group_size = 32
    pexpr = PythonPrinter().doprint
    sexpr = MetalExprPrinter().doprint
    kexpr = sexpr
    headers: OrderedSet[str] = OrderedSet(["utils"])

    def __init__(
        self,
        tiling: dict[str, sympy.Expr],
        **kwargs: Any,
    ) -> None:
        super().__init__(tiling, **kwargs)
        self.acc_var_ids = itertools.count()
        self.multistage_reduction = False

    def dtype_to_str(self, dtype: torch.dtype) -> str:
        return DTYPE_TO_METAL[dtype]

    def load(self, name: str, index: sympy.Expr) -> CSEVariable:
        """Codegen a load from an InputBuffer"""
        var = self.args.input(name)
        index = self.prepare_indexing(index)
        dtype = V.graph.get_dtype(name)
        line = f"{var}[{self.index_to_str(index)}]"
        if dtype in [torch.float16, torch.bfloat16]:
            # TODO(NS): Figure out the right balance betwene optype casts
            # op_math_t for half-precision floats should be float32
            # Otherwise it can lead to a corretness issues with eager
            line = f"static_cast<float>({line})"
            dtype = torch.float32
        return self.cse.generate(self.loads, line, dtype=dtype)

    def store(
        self, name: str, index: sympy.Expr, value: CSEVariable, mode: StoreMode = None
    ) -> None:
        var = self.args.output(name)
        index = self.prepare_indexing(index)
        dtype_str = self.dtype_to_str(V.graph.get_dtype(name))
        cast_val = f"static_cast<{dtype_str}>({value})"
        if mode is None:
            line = f"{var}[{self.index_to_str(index)}] = {cast_val};"
        elif mode == "atomic_add":
            self.headers.add("atomic")
            atomic_type = f"c10::metal::AtomicType<{dtype_str}>"
            cast_var = f"reinterpret_cast<device {atomic_type}::type *>({var})"
            line = f"{atomic_type}::atomic_add({cast_var}, {self.index_to_str(index)}, {cast_val});"
        else:
            raise RuntimeError(f"Unimplemented store mode {mode}")
        if self.inside_reduction:
            self.compute.writeline(DeferredLine(name, line))
        else:
            self.stores.writeline(DeferredLine(name, line))

    def store_reduction(self, name: str, index: sympy.Expr, value: CSEVariable) -> None:
        var = self.args.output(name)
        index = self.prepare_indexing(index)
        dtype_str = self.dtype_to_str(V.graph.get_dtype(name))
        reduction_dim = next(t for t in self.range_trees if t.is_reduction)
        # Only one thread in the reduction group needs to store the results
        line = f"{var}[{self.index_to_str(index)}] = static_cast<{dtype_str}>({value});"
        line = f"if ({reduction_dim.name} == 0) {line}"
        self.stores.writeline(DeferredLine(name, line))

    def _new_idxvar(
        self,
        dtype: Union[str | torch.dtype],
        elem_count: Optional[int] = None,
        default_value: Optional[Any] = None,
        is_threadgroup: bool = True,
        bounds: ValueRanges[Any] = ValueRanges.unknown(),
    ) -> CSEVariable:
        if isinstance(dtype, torch.dtype):
            dtype = self.dtype_to_str(dtype)
        var_name = f"tmp_acc_{next(self.acc_var_ids)}"
        var = V.kernel.create_cse_var(var_name, bounds, dtype)
        var_def = "threadgroup " if is_threadgroup else ""
        var_def += f"{dtype} {var_name}"
        if elem_count:
            var_def += f"[{elem_count}]"
        if default_value is not None:
            assert not is_threadgroup, "Thread group var can not have default value"
            var_def += f" = {default_value}"
        self.indexing_code.writeline(var_def + self.suffix)
        return var

    def reduction(
        self,
        dtype: torch.dtype,
        src_dtype: torch.dtype,
        reduction_type: ReductionType,
        value: Union[CSEVariable, tuple[CSEVariable, ...]],
    ) -> Union[CSEVariable, tuple[CSEVariable, ...]]:
        "Caching wrapper around _reduction_nocache"
        cache_key = (src_dtype, reduction_type, value)
        # Return cached reduction
        if cache_key in self.cse.reduction_cache:
            return self.cse.reduction_cache[cache_key]
        result = self._reduction_nocache(dtype, src_dtype, reduction_type, value)
        self.cse.reduction_cache[cache_key] = result  # type: ignore[assignment]
        return result

    def _reduction_nocache(
        self,
        dtype: torch.dtype,
        src_dtype: torch.dtype,
        reduction_type: ReductionType,
        value: Union[CSEVariable, tuple[CSEVariable, ...]],
    ) -> Union[CSEVariable, tuple[CSEVariable, ...]]:
        """Codegen a reduction operation.
        Only sum and prod operations are somewhat reasonable optimized"""
        assert self.inside_reduction
        assert not self._load_mask

        # Establish reduction buffer size and index expression
        reduction_idx = ""
        acc_buf_size = 1
        for rd in self.range_trees:
            if not rd.is_reduction:
                continue
            if reduction_idx:
                reduction_idx += " + "
            reduction_idx += f"{rd.name} * {acc_buf_size}"
            acc_buf_size *= rd.numel
        acc_buf_size = min(acc_buf_size, self.max_threadgroup_size)

        if reduction_type == "any":
            acc = self._new_idxvar(dtype)
            self.indexing_code.writeline(f"{acc} = false;")
            self.indexing_code.writeline(
                "threadgroup_barrier(metal::mem_flags::mem_threadgroup);"
            )
            self.compute.splice(
                f"""
                if ({value}) {{
                    {acc} = true;
                }}
            """
            )
            self.stores.writeline(
                "threadgroup_barrier(metal::mem_flags::mem_threadgroup);"
            )
            return acc

        self.headers.add("reduction_utils")

        if reduction_type in ["prod", "sum"]:
            acc_dtype = DTYPE_TO_COMPUTATION_DTYPE[src_dtype]
            acc_buf = self._new_idxvar(
                acc_dtype, ceildiv(acc_buf_size, self.simd_group_size)
            )
            if not self.multistage_reduction:
                val = value
            else:
                default_val, reduction_op = (
                    (0, "+") if reduction_type == "sum" else (1, "*")
                )
                val = self._new_idxvar(
                    acc_dtype, default_value=default_val, is_threadgroup=False
                )
                self.compute.splice(f"{val} {reduction_op}= {value};")
            return self.cse.generate(
                self.stores,
                f"c10::metal::threadgroup_{reduction_type}({acc_buf}, {val}, {reduction_idx}, {acc_buf_size})",
                dtype=DTYPE_TO_COMPUTATION_DTYPE[dtype],
            )
        if reduction_type in ["max", "min", "argmin", "argmax"]:
            acc_buf = self._new_idxvar(src_dtype, acc_buf_size)
            acc_thread_var = f"{acc_buf}[{reduction_idx}]"
            src_metal_type = DTYPE_TO_METAL[src_dtype]
            if not self.multistage_reduction:
                self.compute.splice(
                    f"{acc_thread_var} = static_cast<{src_metal_type}>({value});"
                )
                return self.cse.generate(
                    self.stores,
                    f"c10::metal::threadgroup_{reduction_type}({acc_buf}, {acc_buf_size})",
                    dtype=dtype,
                )
            lim_fn = "lowest" if reduction_type.endswith("max") else "max"
            self.indexing_code.writeline(
                f"{acc_thread_var} = ::metal::numeric_limits<{src_metal_type}>::{lim_fn}();"
            )
            if reduction_type.startswith("arg"):
                idx_var = next(
                    t for t in self.range_tree_nodes.values() if t.is_reduction
                )
                idx_acc_buf = self._new_idxvar(torch.long, acc_buf_size)
                cmp_op = ">" if reduction_type == "argmax" else "<"
                idx_thread_var = f"{idx_acc_buf}[{reduction_idx}]"
                self.indexing_code.splice(f"{idx_thread_var} = -1;")
                self.compute.splice(f"""
                if ({value} {cmp_op} {acc_thread_var}) {{
                    {acc_thread_var} = {value};
                    {idx_thread_var} = {idx_var.name};
                }}
                """)
                return self.cse.generate(
                    self.stores,
                    f"{idx_acc_buf}[c10::metal::threadgroup_{reduction_type}({acc_buf}, {acc_buf_size})]",
                    dtype=dtype,
                )
            self.compute.writeline(
                f"{acc_thread_var} = ::c10::metal::{reduction_type}({acc_thread_var}, {value});"
            )
            return self.cse.generate(
                self.stores,
                f"c10::metal::threadgroup_{reduction_type}({acc_buf}, {acc_buf_size})",
                dtype=dtype,
            )
        if reduction_type == "welford_reduce":
            if not self.multistage_reduction:
                acc_buf = self._new_idxvar(src_dtype, acc_buf_size)
                self.compute.splice(f"{acc_buf}[{reduction_idx}] = {value};")
                wf_res = self.cse.generate(
                    self.compute,
                    f"c10::metal::threadgroup_{reduction_type}({acc_buf}, {acc_buf_size})",
                )
                return OpsWrapper._unwrap((f"{wf_res}.x", f"{wf_res}.y", f"{wf_res}.z"))
            acc_buf = self._new_idxvar("float3", acc_buf_size)
            acc_thread_var = f"{acc_buf}[{reduction_idx}]"
            self.indexing_code.splice(f"{acc_thread_var} = 0.0;")
            self.compute.writeline(
                f"{acc_thread_var} = ::c10::metal::welford_combine({acc_thread_var}, float3({value}, 0.0, 1.0));"
            )
            wf_res = self.cse.generate(
                self.stores,
                f"c10::metal::threadgroup_welford_combine({acc_buf}, {acc_buf_size})",
            )
            return OpsWrapper._unwrap((f"{wf_res}.x", f"{wf_res}.y", f"{wf_res}.z"))
        if reduction_type == "welford_combine":
            assert isinstance(value, tuple), "Input to welford combine must be tuple"
            acc_buf = self._new_idxvar("float3", acc_buf_size)
            acc_thread_var = f"{acc_buf}[{reduction_idx}]"
            inp_value = f"float3({value[0]}, {value[1]}, {value[2]})"
            self.indexing_code.splice(f"{acc_thread_var} = 0.0;")
            if self.multistage_reduction:
                self.indexing_code.splice(f"{acc_thread_var} = 0.0;")
                self.compute.writeline(
                    f"{acc_thread_var} = ::c10::metal::welford_combine({acc_thread_var}, {inp_value});"
                )
            else:
                self.compute.writeline(f"{acc_thread_var} = {inp_value};")
            wf_res = self.cse.generate(
                self.stores if self.multistage_reduction else self.compute,
                f"c10::metal::threadgroup_{reduction_type}({acc_buf}, {acc_buf_size})",
            )
            return OpsWrapper._unwrap((f"{wf_res}.x", f"{wf_res}.y", f"{wf_res}.z"))
        raise NotImplementedError(reduction_type)

    def codegen_iteration_ranges_entry(self, entry: IterationRangesEntry) -> None:
        index_expr = self.rename_indexing(entry.expr)
        index_str = self.sexpr(index_expr)  # type: ignore[misc]
        if entry.is_reduction:
            self.multistage_reduction = entry.root.numel > self.max_threadgroup_size
        if not entry.is_reduction or not self.multistage_reduction:
            self.indexing_code.writeline(
                f"{self.index_dtype} {entry.name} = {index_str};"
            )
            return
        # When reducing the thensor whose size exceeds max threadgroup size
        # loop over extra indices per reduction thread and perform part of the operation
        # using values in the shared memory
        loop_size = (
            entry.root.numel + self.max_threadgroup_size - 1
        ) // self.max_threadgroup_size
        self.body.writeline(
            f"for(auto {entry.name}_cnt = 0; {entry.name}_cnt < {loop_size}; ++{entry.name}_cnt) {{"
        )
        with self.body.indent():
            self.body.writeline(
                f"{self.index_dtype} {entry.name} = {loop_size} * {index_str} + {entry.name}_cnt;"
            )
            # Check that reduction is performed only within tensor boundary
            if loop_size * self.max_threadgroup_size != entry.root.numel:
                self.body.writeline(f"if ({entry.name} >= {entry.root.numel}) break;")

    def codegen_body(self) -> None:
        """
        Concat output code from index_code, loads, compute, stores,
        suffix into self.body.

        For pointwise kernels, this is called just once at the end.

        For reduction kernels, this generates a loop over the reduction
        axis.
        """
        if self.multistage_reduction:
            with self.body.indent():
                self.body.splice(self.loads)
                self.body.splice(self.compute)
            self.body.writeline("}")
            self.multistage_reduction = False
        else:
            self.body.splice(self.loads)
            self.body.splice(self.compute)
        self.body.splice(self.stores)
        self.loads.clear()
        self.compute.clear()
        self.stores.clear()

    def codegen_kernel(self, name: Optional[str] = None) -> str:
        """Called at the end to generate a final kernel string"""
        self.codegen_body()
        code = IndentedBuffer()
        code.writeline('compile_mps_shader("""')
        idx_vars = self.active_range_trees()
        with code.indent():
            for header in self.headers:
                code.writeline(f"#include <c10/metal/{header}.h>")
            if self.inside_reduction:
                total_reduction_size = math.prod(
                    t.numel for t in self.range_trees if t.is_reduction
                )
                threadgroup_size = min(total_reduction_size, self.max_threadgroup_size)
                code.writeline(
                    f"[[max_total_threads_per_threadgroup({threadgroup_size})]]"
                )
            code.writeline("kernel void generated_kernel(")
            with code.indent():
                for outer, inner in self.args.output_buffers.items():
                    if outer in self.removed_buffers:
                        continue
                    dtype_str = self.dtype_to_str(V.graph.get_dtype(outer))
                    code.writeline(f"device {dtype_str}* {inner},")
                for outer, inner in self.args.input_buffers.items():
                    dtype_str = self.dtype_to_str(V.graph.get_dtype(outer))
                    code.writeline(f"constant {dtype_str}* {inner},")
                for outer, inner in self.args.sizevars.items():
                    code.writeline(f"constant long& {inner},")
                assert len(idx_vars) < 4, "Up to 3 index variables are supported"
                thread_pos_dtype = (
                    f"uint{len(idx_vars)}" if len(idx_vars) > 1 else "uint"
                )
                thread_pos_var_name = (
                    idx_vars[0].name if len(idx_vars) == 1 else "thread_pos"
                )
                thread_pos_suffix = "," if self.inside_reduction else ""
                code.writeline(
                    f"{thread_pos_dtype} {thread_pos_var_name} [[thread_position_in_grid]]{thread_pos_suffix}"
                )
                if self.inside_reduction:
                    code.writeline(
                        f"{thread_pos_dtype} group_pos [[thread_position_in_threadgroup]]"
                    )
            code.writeline(") {")
            with code.indent():
                if len(idx_vars) > 1:
                    for idx, var in enumerate(idx_vars):
                        code.writeline(
                            f"auto {var.name} = thread_pos.{chr(120 + idx)};"
                        )
                code.splice(self.indexing_code)
                code.splice(self.body)
            code.writeline("}")
        code.writeline('""")')

        return code.getvalue()

    def call_kernel(self, name: str, node: Any = None) -> None:
        """Codegen a call to this kernel"""
        wrapper = V.graph.wrapper_code
        # Make sure sizevarss has been computed
        for v in self.args.sizevars.keys():
            wrapper.ensure_size_computed(v)

        args = [*self.args.output_buffers.keys(), *self.args.input_buffers.keys()]
        args = [arg for arg in args if arg not in self.removed_buffers]
        args += [str(v) for v in self.args.sizevars.keys()]
        # For reduction kernels, limit the maximum size over reduction dimentions to
        # a maximum threadgroup size
        if len(self.active_range_trees()) > 0:
            threads = [
                self.pexpr(
                    sympy.Min(v.numel, self.max_threadgroup_size)  # type: ignore[misc]
                    if v.is_reduction
                    else v.numel
                )
                for v in self.active_range_trees()
            ]
            args += [f"threads=[{', '.join(threads)}]"]
        if self.inside_reduction:
            threads = [
                self.pexpr(sympy.Min(v.numel, self.max_threadgroup_size))  # type: ignore[misc]
                if v.is_reduction
                else "1"
                for v in self.active_range_trees()
            ]
            args += [f"group_size=[{', '.join(threads)}]"]

        wrapper.generate_kernel_call(
            name,
            args,
            device=torch.device("cpu"),  # TODO: Fix me, MPS does not expose streams now
            triton=False,
        )

    def check_bounds(
        self, expr: sympy.Expr, size: sympy.Expr, lower: bool, upper: bool
    ) -> None:
        if not (lower or upper):
            return
        # TODO(malfet): support asserts
        # See https://github.com/pytorch/pytorch/issues/144634
        expr_str = self.index_to_str(expr)
        lower_expr = f"{expr_str} < 0" if lower else ""
        # TODO(malfet): Is upper bound inclusive or exclusive?
        upper_expr = f"{expr_str} > {self.index_to_str(size)}" if upper else ""
        if lower and upper:
            line = f"if (({lower_expr}) && ({upper_expr})) return"
        else:
            line = f"if ({lower_expr}{upper_expr}) return"
        self.cse.generate(self.compute, line, assignment=False)


class MetalScheduling(SIMDScheduling):
    kernel_type = MetalKernel  # type: ignore[assignment]

    def __init__(self, scheduler: Optional[Scheduler]) -> None:
        super().__init__(scheduler)
        wrapper = V.graph.wrapper_code
        if wrapper is not None:
            wrapper.header.splice(
                "from torch._inductor.runtime.runtime_utils import compile_mps_shader"
            )

    def define_kernel(
        self, src_code: str, node_schedule: list[SchedulerNode], kernel: MetalKernel
    ) -> str:
        wrapper = V.graph.wrapper_code
        if src_code in wrapper.src_to_kernel:
            kernel_name = wrapper.src_to_kernel[src_code]
        else:
            # TODO: Merge multiple kernels into a single library
            # Either using MultiKernel concept or overriding SIMDScheduling.codegen_node_scheduling
            mps_lib_name = f"mps_lib_{wrapper.next_kernel_suffix()}"
            kernel_name = f"{mps_lib_name}.generated_kernel"
            wrapper.src_to_kernel[src_code] = kernel_name
            origins, detailed_origins = get_kernel_metadata(node_schedule, wrapper)
            metadata_comment = f"{origins}\n{detailed_origins}"
            wrapper.define_kernel(mps_lib_name, src_code, metadata_comment)

        return kernel_name
