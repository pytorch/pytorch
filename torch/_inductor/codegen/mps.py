# This is not a feature-complete compiler backend
# Just an early prototype that shows that one can compile elementwise ops into a Metal shader
from __future__ import annotations

import functools
import itertools
from typing import Any, Optional, TYPE_CHECKING

import sympy
from sympy.printing.precedence import PRECEDENCE

import torch
from torch.utils._sympy.printers import ExprPrinter as ExprPrinter_
from torch.utils._sympy.value_ranges import ValueRanges

from ..utils import get_bounds_index_expr, get_kernel_metadata
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
        return f"metal::min({', '.join(map(self._print, expr.args))})"

    def _print_Max(self, expr: sympy.Expr) -> str:
        if len(expr.args) != 2:
            raise RuntimeError("metal::max only supported for 2 args")
        return f"metal::max({', '.join(map(self._print, expr.args))})"

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


class MetalOverrides(OpOverrides):
    @staticmethod
    def to_dtype(
        x: CSEVariable,
        dtype: torch.dtype,
        src_dtype: Optional[torch.dtype] = None,
        use_compute_types: bool = True,
    ) -> str:
        return f"static_cast<{DTYPE_TO_METAL[dtype]}>({x})"

    @staticmethod
    def to_dtype_bitcast(
        x: CSEVariable, dtype: torch.dtype, src_dtype: torch.dtype
    ) -> str:
        return f"*reinterpret_cast<thread {DTYPE_TO_METAL[dtype]}*>(&{x})"

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
    def i0(x: CSEVariable) -> str:
        return f"c10::metal::i0({x})"

    @staticmethod
    def i1(x: CSEVariable) -> str:
        return f"c10::metal::i1({x})"

    @staticmethod
    def erf(x: CSEVariable) -> str:
        return f"c10::metal::erf({x})"

    @staticmethod
    def erfinv(x: CSEVariable) -> str:
        return f"c10::metal::erfinv({x})"

    @staticmethod
    def lgamma(x: CSEVariable) -> str:
        return f"c10::metal::log_gamma({x})"

    @staticmethod
    def polygamma(x: CSEVariable, y: CSEVariable) -> str:
        return f"c10::metal::polygamma({x}, {y})"

    @staticmethod
    def digamma(x: CSEVariable) -> str:
        return f"c10::metal::digamma({x})"

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
    def sqrt(x: CSEVariable) -> str:
        return f"metal::sqrt({x})"

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
        # Upcast to float otherwise the generated code doesn't typecheck.
        # TODO (dcci): remove this workaround
        float_a = f"static_cast<float>({a})" if a.dtype != torch.float else a
        float_b = f"static_cast<float>({b})" if b.dtype != torch.float else b
        return f"metal::trunc({float_a}/{float_b})"

    @staticmethod
    def ceil(x: CSEVariable) -> str:
        return f"metal::ceil({x})"

    @staticmethod
    def rand(seed: CSEVariable, offset: CSEVariable) -> str:
        return f"c10::metal::rand({seed}, {offset})"

    @staticmethod
    def randn(seed: CSEVariable, offset: CSEVariable) -> str:
        return f"c10::metal::randn({seed}, {offset})"

    @staticmethod
    def randint64(
        seed: CSEVariable, offset: CSEVariable, low: CSEVariable, high: CSEVariable
    ) -> str:
        return f"c10::metal::randint64({seed}, {offset}, {low}, {high})"

    @staticmethod
    def round(x: CSEVariable) -> str:
        return f"metal::round({x})"

    @staticmethod
    def pow(a: CSEVariable, b: CSEVariable) -> str:
        cast_a = f"static_cast<decltype({a}+{b})>({a})"
        cast_b = f"static_cast<decltype({a}+{b})>({b})"
        return f"metal::pow({cast_a}, {cast_b})"

    @staticmethod
    def zeta(a: CSEVariable, b: CSEVariable) -> str:
        return f"c10::metal::zeta({a}, {b})"

    @staticmethod
    def spherical_bessel_j0(x: CSEVariable) -> str:
        return f"c10::metal::spherical_bessel_j0({x})"

    @staticmethod
    def xlog1py(x: CSEVariable) -> str:
        return f"c10::metal::xlog1py({x})"

    @staticmethod
    def entr(x: CSEVariable) -> str:
        return f"c10::metal::entr({x})"


MetalOverrides._initialize_pointwise_overrides("mps")


class MetalKernel(SIMDKernel):
    overrides = MetalOverrides  # type: ignore[assignment]
    suffix = ";"
    newvar_prefix = "auto "
    max_threadgroup_size = 1024
    pexpr = PythonPrinter().doprint
    sexpr = MetalExprPrinter().doprint
    kexpr = sexpr

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
        line = f"{var}[{self.index_to_str(index)}]"
        return self.cse.generate(self.loads, line, dtype=V.graph.get_dtype(name))

    def store(
        self, name: str, index: sympy.Expr, value: CSEVariable, mode: StoreMode = None
    ) -> None:
        var = self.args.output(name)
        index = self.prepare_indexing(index)
        dtype_str = self.dtype_to_str(V.graph.get_dtype(name))
        line = f"{var}[{self.index_to_str(index)}] = static_cast<{dtype_str}>({value});"
        if self.inside_reduction:
            self.compute.writeline(DeferredLine(name, line))
        else:
            self.stores.writeline(DeferredLine(name, line))

    def _new_accvar(
        self,
        dtype: torch.dtype,
        elem_count: Optional[int] = None,
        bounds: ValueRanges[Any] = ValueRanges.unknown(),
    ) -> CSEVariable:
        var_name = f"tmp_acc_{next(self.acc_var_ids)}"
        var = V.kernel.create_cse_var(var_name, bounds, dtype)
        if elem_count:
            self.indexing_code.writeline(
                f"threadgroup {self.dtype_to_str(dtype)} {var_name}[{elem_count}];"
            )
        else:
            self.indexing_code.writeline(
                f"threadgroup {self.dtype_to_str(dtype)} {var_name};"
            )
        return var

    def reduction(
        self,
        dtype: torch.dtype,
        src_dtype: torch.dtype,
        reduction_type: ReductionType,
        value: Union[CSEVariable, tuple[CSEVariable, ...]],
    ) -> Union[CSEVariable, tuple[CSEVariable, ...]]:
        """Codegen a reduction operation"""
        reduction_dim = next(t for t in self.range_trees if t.is_reduction)
        acc_buf_size = min(reduction_dim.numel, self.max_threadgroup_size)
        if reduction_type == "any":
            acc = self._new_accvar(dtype)
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
        if reduction_type in ["prod", "sum"]:
            acc_buf = self._new_accvar(src_dtype, acc_buf_size)
            if self.multistage_reduction:
                default_val, reduction_op = (
                    (0, "+") if reduction_type == "sum" else (1, "*")
                )
                self.indexing_code.writeline(
                    f"{acc_buf}[{reduction_dim.name}] = {default_val};"
                )
                self.compute.splice(
                    f"{acc_buf}[{reduction_dim.name}] {reduction_op}= {value};"
                )
            else:
                self.compute.splice(f"{acc_buf}[{reduction_dim.name}] = {value};")
            return self.cse.generate(
                self.stores,
                f"c10::metal::threadgroup_{reduction_type}({acc_buf}, {acc_buf_size})",
                dtype=DTYPE_TO_COMPUTATION_DTYPE[dtype],
            )
        if reduction_type in ["max", "min", "argmin", "argmax"]:
            acc_buf = self._new_accvar(src_dtype, acc_buf_size)
            acc_thread_var = f"{acc_buf}[{reduction_dim.name}]"
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
                idx_acc_buf = self._new_accvar(torch.long, acc_buf_size)
                cmp_op = ">" if reduction_type == "argmax" else "<"
                idx_thread_var = f"{idx_acc_buf}[{reduction_dim.name}]"
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
            assert not self.multistage_reduction, (
                f"Multistage reduction not yet supported for {reduction_type}"
            )
            acc_buf = self._new_accvar(src_dtype, acc_buf_size)
            self.compute.splice(f"{acc_buf}[{reduction_dim.name}] = {value};")
            wf_res = self.cse.generate(
                self.compute,
                f"c10::metal::threadgroup_{reduction_type}({acc_buf}, {acc_buf_size})",
            )
            return OpsWrapper._unwrap(
                (f"{wf_res}.x", f"{wf_res}.y", self.features.reduction_numel)
            )
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
            code.splice(
                """
            #include <c10/metal/random.h>
            #include <c10/metal/special_math.h>
            #include <c10/metal/utils.h>
            """,
                strip=True,
            )
            if self.inside_reduction:
                code.writeline("#include <c10/metal/reduction_utils.h>")
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
        upper_expr = f"{expr_str} >= {self.index_to_str(size)}" if upper else ""
        if lower and upper:
            line = f"if (({lower_expr}) && ({upper_expr})) return"
        else:
            line = f"if ({lower_expr}{upper_expr}) return"
        self.cse.generate(self.compute, line, assignment=False)


@functools.cache
def _warn_prototype() -> None:
    import warnings

    warnings.warn(
        "torch.compile for Metal is an early protoype and might not work as expected."
        " For details see https://github.com/pytorch/pytorch/issues/150121",
        stacklevel=2,
    )


class MetalScheduling(SIMDScheduling):
    kernel_type = MetalKernel  # type: ignore[assignment]

    def __init__(self, scheduler: Optional[Scheduler]) -> None:
        super().__init__(scheduler)
        _warn_prototype()
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
