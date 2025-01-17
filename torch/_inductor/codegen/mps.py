# This is not a feature-complete compiler backend
# Just an early prototype that shows that one can compile elementwise ops into a Metal shader
from typing import Any, Optional

import sympy

import torch
from torch.utils._sympy.printers import ExprPrinter as ExprPrinter_

from ..ops_handler import StoreMode
from ..scheduler import Scheduler, SchedulerNode
from ..utils import get_bounds_index_expr, get_kernel_metadata
from ..virtualized import ops, V
from .common import CSEVariable, DeferredLine, IndentedBuffer, OpOverrides
from .simd import IterationRangesEntry, SIMDKernel, SIMDScheduling


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


def value_to_metal(val: CSEVariable) -> str:
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
    def constant(val: CSEVariable, dtype: torch.dtype) -> str:
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
    def where(a: CSEVariable, b: CSEVariable, c: CSEVariable) -> str:
        return f"{a} ? {b} : {value_to_metal(c)}"

    @staticmethod
    def remainder(a: CSEVariable, b: CSEVariable) -> str:
        if b.dtype is not None and not b.dtype.is_floating_point:
            return f"{a} % {b}"
        # Upcast to float otherwise results of remainder op are wrong for half
        float_a = f"static_cast<float>({a})" if a.dtype != torch.float else a
        float_b = f"static_cast<float>({b})" if b.dtype != torch.float else b
        return f"{float_a} - {float_b} * metal::floor({float_a} / {float_b})"

    @staticmethod
    def maximum(a: CSEVariable, b: CSEVariable) -> str:
        typecast_a = f"static_cast<decltype({a}+{b})>({a})"
        typecast_b = f"static_cast<decltype({a}+{b})>({b})"
        max_res = f"metal::max({typecast_a}, {typecast_b})"
        return f"isnan({a} + {b}) ? {a} + {b} : {max_res}"

    @staticmethod
    def minimum(a: CSEVariable, b: CSEVariable) -> str:
        typecast_a = f"static_cast<decltype({a}+{b})>({a})"
        typecast_b = f"static_cast<decltype({a}+{b})>({b})"
        min_res = f"metal::min({typecast_a}, {typecast_b})"
        return f"isnan({a} + {b})  ? {a} + {b} : {min_res}"

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
    def round(x: CSEVariable) -> str:
        return f"metal::round({x})"


class MetalKernel(SIMDKernel):
    overrides = MetalOverrides  # type: ignore[assignment]
    suffix = ";"
    newvar_prefix = "auto "
    sexpr = MetalExprPrinter().doprint
    kexpr = sexpr

    def __init__(
        self,
        tiling: dict[str, sympy.Expr],
        **kwargs: Any,
    ) -> None:
        super().__init__(tiling, **kwargs)
        self.compute = self.body
        self.loads = self.body
        self.stores = self.body

    def dtype_to_str(self, dtype: torch.dtype) -> str:
        return DTYPE_TO_METAL[dtype]

    def load(self, name: str, index: sympy.Expr) -> CSEVariable:
        """Codegen a load from an InputBuffer"""
        var = self.args.input(name)
        index = self.prepare_indexing(index)
        line = f"{var}[{self.sexpr(index)}]"
        return self.cse.generate(self.body, line, dtype=V.graph.get_dtype(name))

    def store(
        self, name: str, index: sympy.Expr, value: CSEVariable, mode: StoreMode = None
    ) -> None:
        var = self.args.output(name)
        index = self.prepare_indexing(index)
        dtype_str = self.dtype_to_str(V.graph.get_dtype(name))
        line = f"{var}[{index}] = static_cast<{dtype_str}>({value});"
        self.body.writeline(DeferredLine(name, line))

    def codegen_iteration_ranges_entry(self, entry: IterationRangesEntry) -> None:
        index_expr = self.rename_indexing(entry.expr)
        index_str = self.sexpr(index_expr)  # type: ignore[misc]
        self.body.writeline(f"{self.index_dtype} {entry.name} = {index_str};")

    def codegen_kernel(self, name: Optional[str] = None) -> str:
        """Called at the end to generate a final kernel string"""
        code = IndentedBuffer()
        code.writeline('compile_mps_shader("""')
        idx_var_names = [v.name for v in self.active_range_trees()]
        with code.indent():
            code.splice(
                """
            template<typename T> inline bool isnan(T) { return false; }
            template<> inline bool isnan(float x) { return metal::isnan(x); }
            template<> inline bool isnan(half x) { return metal::isnan(x); }
            #if __METAL_VERSION__ >= 310
            template<> inline bool isnan(bfloat x) { return metal::isnan(x); }
            #endif
            """,
                strip=True,
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
                if len(idx_var_names) == 1:
                    code.writeline(
                        f"uint {idx_var_names[0]} [[thread_position_in_grid]]"
                    )
                else:
                    assert (
                        len(idx_var_names) < 4
                    ), "Up to 3 index variables are supported"
                    code.writeline(
                        f"uint{len(idx_var_names)} thread_pos [[thread_position_in_grid]]"
                    )

            code.writeline(") {")
            with code.indent():
                if len(idx_var_names) > 1:
                    for idx, name in enumerate(idx_var_names):
                        code.writeline(f"auto {name} = thread_pos.{chr(120 + idx)};")
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
        if len(self.active_range_trees()) > 0:
            args += [
                f"threads=[{', '.join(str(v.numel) for v in self.active_range_trees())}]"
            ]

        wrapper.generate_kernel_call(
            name,
            args,
            gpu=False,  # TODO: Fix me, MPS does not expose streams now
            triton=False,
        )

    def check_bounds(
        self, expr: sympy.Expr, size: sympy.Expr, lower: bool, upper: bool
    ) -> None:
        if not (lower or upper):
            return
        # TODO(malfet): support asserts
        # See https://github.com/pytorch/pytorch/issues/144634
        lower_expr = f"{self.sexpr(expr)} < 0" if lower else ""
        upper_expr = f"{self.sexpr(expr)} >= {self.sexpr(size)}" if upper else ""
        if lower and upper:
            line = f"if (({lower_expr}) && ({upper_expr})) return"
        else:
            line = f"if ({lower_expr}{upper_expr}) return"
        self.cse.generate(self.body, line, assignment=False)


class MetalScheduling(SIMDScheduling):
    kernel_type = MetalKernel  # type: ignore[assignment]

    def __init__(self, scheduler: Scheduler) -> None:
        super().__init__(scheduler)
        wrapper = V.graph.wrapper_code
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
