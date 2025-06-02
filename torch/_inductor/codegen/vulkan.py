# This is not a feature-complete compiler backend
# Just an early prototype that shows that one can compile elementwise ops into a Vulkan shader

import logging
from typing import Any, Optional, TYPE_CHECKING, Union

import sympy
from sympy.printing.precedence import PRECEDENCE

import torch
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.printers import ExprPrinter as ExprPrinter_
from torch.utils._sympy.value_ranges import ValueRanges

from ..ops_handler import StoreMode
from ..scheduler import Scheduler, SchedulerNode
from ..utils import DeferredLineBase, get_bounds_index_expr, get_kernel_metadata
from ..virtualized import ops, OpsValue, V
from .common import (
    CSE,
    CSEVariable,
    DeferredLine,
    IndentedBuffer,
    OpOverrides,
    PythonPrinter,
)
from .simd import IterationRangesEntry, SIMDKernel, SIMDScheduling


if TYPE_CHECKING:
    from .common import OpVarT
else:
    OpVarT = str

log = logging.getLogger(__name__)

DTYPE_TO_VULKAN_GLSL_AND_EXTENSION = {
    torch.bool: (
        "uint8_t",
        "GL_EXT_shader_explicit_arithmetic_types_int8 : require",
    ),  # GLSL bool is 32 bits, don't use it.
    torch.int8: ("int8_t", "GL_EXT_shader_explicit_arithmetic_types_int8 : require"),
    torch.int16: ("int16_t", "GL_EXT_shader_explicit_arithmetic_types_int16 : require"),
    torch.int32: ("int32_t", "GL_EXT_shader_explicit_arithmetic_types_int32 : require"),
    torch.int64: (
        "int64_t",
        "GL_EXT_shader_explicit_arithmethic_types_int64 : require",
    ),
    torch.uint8: ("uint8_t", "GL_EXT_shader_explicit_arithmetic_types_int8 : require"),
    torch.uint16: ("uint16_t", "GL_EXT_shader_explicit_arithmetic_types_int16 : require"),
    torch.uint32: (
        "uint32_t",
        "GL_EXT_shader_explicit_arithmetic_types_int32 : require",
    ),
    torch.uint64: (
        "uint64_t",
        "GL_EXT_shader_explicit_arithmethic_types_int64 : require",
    ),
    torch.float: ("float", None),
    torch.half: ("float16_t", "GL_EXT_shader_explicit_arithmetic_types_float16 : require"),
    torch.bfloat16: (
        "bfloat16_t",
        "GL_EXT_bfloat16 : require",
    ),  # TODO: may not be supported at runtime
}


def dtype_to_vulkan_glsl(x):
    dtype, ext = DTYPE_TO_VULKAN_GLSL_AND_EXTENSION[x]
    if ext is not None:
        V.kernel.extensions.add(ext)
    return dtype


def value_to_vulkan_glsl(val: Union[float, int, bool, str, CSEVariable]) -> str:
    if isinstance(val, float):
        if val == torch.inf:
            return "(1.0 / 0.0)"
        elif val == -torch.inf:
            return "(-1.0 / 0.0)"
        elif val != val:  # Only float that not equal to self is nan
            raise RuntimeError("GLSL doesn't support NaN")
        return str(val)
    elif isinstance(val, bool):
        return "true" if val else "false"
    return str(val)


class VulkanGLSLExprPrinter(ExprPrinter_):
    """Converts sympy expression to GLSL code snippet."""

    def _print_FloorDiv(self, expr: sympy.Expr) -> str:
        x, div = expr.args
        x = self.doprint(x)
        div = self.doprint(div)
        if expr.is_integer:
            return f"({x}) / ({div})"
        return f"floor(float({x})) / ({div})"

    def _print_ModularIndexing(self, expr: sympy.Expr) -> str:
        x, div, mod = expr.args
        x = self.doprint(x)
        if div != 1:
            div = self.doprint(div)
            if expr.is_integer:
                x = f"({x}) / ({div})"
            else:
                x = f"floor(float({x})) / ({div})"
        mod = self.doprint(mod)
        return f"({x}) % ({mod})"

    def _print_Min_or_Max(self, expr: sympy.Expr, name: str) -> str:
        if len(expr.args) != 2:
            raise RuntimeError(f"{name} only supported for 2 args")
        a, b = map(self._print, expr.args)
        return f"{name}({a}, {b})"

    def _print_Min(self, expr: sympy.Expr) -> str:
        return self._print_Min_or_Max(expr, "min")

    def _print_Max(self, expr: sympy.Expr) -> str:
        return self._print_Min_or_Max(expr, "max")

    def _print_Abs(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 1
        return f"abs({self._print(expr.args[0])})"

    def _print_RoundToInt(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 1
        # roundEven matches metal::rint and libdevice.llrint behavior.
        # TODO: support double?
        V.kernel.extensions.add("GL_EXT_shader_explicit_arithmethic_types_int32")
        return f"int32_t(roundEven({self._print(expr.args[0])})))"

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
        return f"(roundEven(1e{ndigits} * {number_str}) * 1e{-ndigits})"

    def _print_IntTrueDiv(self, expr: sympy.Expr) -> str:
        lhs, rhs = expr.args
        # TODO: This is only accurate up to 2**23
        return f"(float({self._print(lhs)}) / float({self._print(rhs)}))"

    def _print_FloatPow(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 2
        x, y = map(self.doprint, expr.args)
        return f"pow({x}, {y})"

    def _print_PowByNatural(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 2
        x, y = map(self.doprint, expr.args)
        return f"pow(float({x}), float({y}))"

    def _print_ToFloat(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 1
        x = self.doprint(expr.args[0])
        return f"float({x})"

    def _print_FloorToInt(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 1
        x = self.doprint(expr.args[0])
        return f"int32_t(floor({x}))"

    def _print_TruncToInt(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 1
        x = self.doprint(expr.args[0])
        return f"int32_t(trunc({x}))"


class VulkanGLSLOverrides(OpOverrides):
    """Implements GLSL-specific overrides for ops."""

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
            return f"float({x})"
        # REVIEW: this may not work for float16/bfloat16
        return f"{dtype_to_vulkan_glsl(dtype)}({x})"

    @staticmethod
    def to_dtype_bitcast(
        x: CSEVariable, dtype: torch.dtype, src_dtype: torch.dtype
    ) -> str:
        ALLOWED_DTYPES = (torch.int32, torch.uint32, torch.float)
        assert dtype in ALLOWED_DTYPES, f"can't bitcast to {dtype} yet"
        assert src_dtype in ALLOWED_DTYPES, f"can't bitcast from {src_dtype} yet"
        if dtype == torch.float:
            if src_dtype == torch.float:
                return f"({x})"
            if src_dtype == torch.uint32:
                return f"uintBitsToFloat({x})"
            if src_dtype == torch.int32:
                return f"intBitsToFloat({x})"
            raise Exception(f"Unsupported src_dtype {src_dtype}")
        if dtype == torch.int32:
            if src_dtype == torch.float:
                return f"floatBitsToInt({x})"
            if src_dtype == torch.uint32:
                return f"int({x})"
            if src_dtype == torch.int32:
                return f"({x})"
            raise Exception(f"Unsupported src_dtype {src_dtype}")
        if dtype == torch.uint32:
            if src_dtype == torch.float:
                return f"floatBitsToUint({x})"
            if src_dtype == torch.uint32:
                return f"({x})"
            if src_dtype == torch.int32:
                return f"uint({x})"
        raise Exception(f"Unhandled dtype {dtype}")

    @staticmethod
    def constant(val: Union[bool, float, int], dtype: torch.dtype) -> str:
        return value_to_vulkan_glsl(val)

    @staticmethod
    def index_expr(expr: sympy.Expr, dtype: torch.dtype) -> str:
        idx_str = V.kernel.index_to_str(V.kernel.prepare_indexing(expr))
        var = V.kernel.cse.generate(
            V.kernel.compute, idx_str, bounds=get_bounds_index_expr(expr)
        )
        return ops.to_dtype(var, dtype)

    @staticmethod
    def masked(mask: CSEVariable, body: sympy.Expr, other) -> str:
        with V.kernel.mask_loads(mask, other) as new_mask:
            result = body()

        if result.bounds.is_bool:
            other = bool(other)  # type: ignore[assignment]

        return ops.where(new_mask, result, other)

    @staticmethod
    def where(a: OpVarT, b: OpVarT, c: OpVarT) -> str:
        return f"{a} ? {b} : {value_to_vulkan_glsl(c)}"

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
            f"float({a})"
            if isinstance(a, CSEVariable) and a.dtype != torch.float
            else a
        )
        float_b = (
            f"float({b})"
            if isinstance(b, CSEVariable) and b.dtype != torch.float
            else b
        )
        return f"{float_a} - {float_b} * floor({float_a} / {float_b})"

    @staticmethod
    def maximum(a: CSEVariable, b: CSEVariable) -> str:
        return f"max({a}, {b})"

    @staticmethod
    def minimum(a: CSEVariable, b: CSEVariable) -> str:
        return f"min({a}, {b})"

    @staticmethod
    def logical_or(a: CSEVariable, b: CSEVariable) -> str:
        return f"bool({a}) || bool({b})"

    @staticmethod
    def logical_and(a: CSEVariable, b: CSEVariable) -> str:
        return f"bool({a}) && bool({b})"

    @staticmethod
    def isnan(x: CSEVariable) -> str:
        return f"isnan({x})"

    @staticmethod
    def isinf(x: CSEVariable) -> str:
        return f"isinf({x})"

    @staticmethod
    def log(x: CSEVariable) -> str:
        return f"log({x})"

    @staticmethod
    def exp(x: CSEVariable) -> str:
        return f"exp({x})"

    @staticmethod
    def abs(x: CSEVariable) -> str:
        return f"abs({x})"

    @staticmethod
    def signbit(x: CSEVariable) -> str:
        if x.dtype.is_floating_point:
            return f"((floatBitsToUint(float({x})) & 0x80000000) != 0)"
        return f"({x} < 0)"

    @staticmethod
    def sin(x: CSEVariable) -> str:
        return f"sin({x})"

    # TODO: implement sinc
    # @staticmethod
    # def sinc(x: CSEVariable) -> str:
    #     return f"c10::metal::sinc({x})"

    @staticmethod
    def cos(x: CSEVariable) -> str:
        return f"cos({x})"

    @staticmethod
    def tan(x: CSEVariable) -> str:
        return f"tan({x})"

    @staticmethod
    def asin(x: CSEVariable) -> str:
        return f"asin({x})"

    @staticmethod
    def acos(x: CSEVariable) -> str:
        return f"acos({x})"

    @staticmethod
    def atan(x: CSEVariable) -> str:
        return f"atan({x})"

    @staticmethod
    def atan2(x: CSEVariable, y: CSEVariable) -> str:
        return f"atan({x}, {y})"

    @staticmethod
    def sqrt(x: CSEVariable) -> str:
        return f"sqrt({x})"

    @staticmethod
    def neg(x: CSEVariable) -> str:
        return f"{dtype_to_vulkan_glsl(x.dtype)}(-{x})"

    @staticmethod
    def rsqrt(x: CSEVariable) -> str:
        return f"inversesqrt({x})"

    @staticmethod
    def tanh(x: CSEVariable) -> str:
        return f"tanh({x})"

    @staticmethod
    def atanh(x: CSEVariable) -> str:
        return f"atanh({x})"

    @staticmethod
    def floordiv(a: CSEVariable, b: CSEVariable) -> str:
        # a and b are integer type
        quot = f"{a} / {b}"
        rem = f"{a} % {b}"
        return f"(({a} < 0) != ({b} < 0) ? ({rem} != 0 ? {quot} - 1 : {quot}) : {quot})"

    @staticmethod
    def floor(x: CSEVariable) -> str:
        return f"floor({x})"

    @staticmethod
    def sign(x: CSEVariable) -> str:
        return f"sign({x})"

    @staticmethod
    def fmod(a: CSEVariable, b: CSEVariable) -> str:
        return f"fmod({a}, {b})"

    @staticmethod
    def trunc(x: CSEVariable) -> str:
        return f"trunc({x})"

    @staticmethod
    def truncdiv(a: CSEVariable, b: CSEVariable) -> str:
        quot = f"{a} / {b}"
        if (a.dtype is not None and a.dtype.is_floating_point) or (
            b.dtype is not None and b.dtype.is_floating_point
        ):
            return f"trunc({quot})"
        return quot

    @staticmethod
    def rand(seed: CSEVariable, offset: CSEVariable) -> str:
        raise NotImplementedError("Vulkan doesn't implement randomness yet")

    @staticmethod
    def randn(seed: CSEVariable, offset: CSEVariable) -> str:
        raise NotImplementedError("Vulkan doesn't implement randomness yet")

    @staticmethod
    def randint64(
        seed: CSEVariable, offset: CSEVariable, low: CSEVariable, high: CSEVariable
    ) -> str:
        raise NotImplementedError("Vulkan doesn't implement randomness yet")

    @staticmethod
    def round(x: CSEVariable) -> str:
        return f"roundEven({x})"

    @staticmethod
    def pow(a: CSEVariable, b: CSEVariable) -> str:
        return f"pow(float({a}), float({b}))"

    # TODO: special unary/binary functions from metal


class VulkanCSEVariable(CSEVariable):
    def __init__(
        self,
        name: str,
        bounds: ValueRanges[Any],
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(name, bounds, dtype)
        assert dtype is not None, "VulkanCSEVariable must have dtype"


class VulkanCSE(CSE):
    def get_prefix(self, var: CSEVariable) -> str:
        return f"{dtype_to_vulkan_glsl(var.dtype)} "

    # GLSL requires more explicit casting than you would expect. For example,
    # you can't assign a bool to a uint8 (possibly because bool is 32 bits).
    def wrap_expr_for_assignment_to_var(
        self,
        expr: Union[str, CSEVariable, OpsValue, IndentedBuffer, DeferredLineBase],
        var: CSEVariable,
    ) -> str:
        return f"{dtype_to_vulkan_glsl(var.dtype)}({expr})"


class VulkanGLSLKernel(SIMDKernel):
    """Implement Vulkan codegen based on the SIMDKernel abstraction."""

    overrides = VulkanGLSLOverrides
    suffix = ";"
    # We cannot use a simple prefix for variables. GLSL does not have type
    # deduction and requires explicit variable typing, like C.
    newvar_prefix = ""
    # On desktop this is likely 1024 (confirmed on M4 Mac). Per
    # https://docs.vulkan.org/spec/latest/chapters/limits.html information for
    # maxComputeWorkGroupInvocations, 128 is the minimum, and it goes up to 256
    # on Vulkan 1.4. Not actually needed until we implement reductions.
    max_threadgroup_size = 128
    pexpr = PythonPrinter().doprint
    sexpr = VulkanGLSLExprPrinter().doprint
    kexpr = sexpr
    extensions: OrderedSet[str] = OrderedSet()

    def __init__(
        self,
        tiling: dict[str, sympy.Expr],
        **kwargs: Any,
    ) -> None:
        super().__init__(tiling, **kwargs)
        self.cse = VulkanCSE(self.newvar_prefix, self.suffix)

    def dtype_to_str(self, dtype: torch.dtype) -> str:
        return dtype_to_vulkan_glsl(dtype)

    def create_cse_var(self, *args: Any, **kwargs: Any) -> CSEVariable:
        return VulkanCSEVariable(*args, **kwargs)

    def load(self, name: str, index: sympy.Expr) -> CSEVariable:
        """Codegen a load from an InputBuffer."""
        var = self.args.input(name)
        index = self.prepare_indexing(index)
        dtype = V.graph.get_dtype(name)
        line = f"{var}[{self.index_to_str(index)}]"
        if dtype == torch.float16 or dtype == torch.bfloat16:
            line = f"float({line})"
            dtype = torch.float32
        return self.cse.generate(self.loads, line, dtype=dtype)

    def store(
        self, name: str, index: sympy.Expr, value: CSEVariable, mode: StoreMode = None
    ) -> None:
        var = self.args.output(name)
        index = self.prepare_indexing(index)
        dtype_str = self.dtype_to_str(V.graph.get_dtype(name))
        cast_val = f"{dtype_str}({value})"
        if mode is None:
            line = f"{var}[{self.index_to_str(index)}] = {cast_val};"
        else:
            raise RuntimeError(f"Unimplemented store mode {mode}")
        self.stores.writeline(DeferredLine(name, line))

    # TODO: Implement reductions.

    def codegen_iteration_ranges_entry(self, entry: IterationRangesEntry) -> None:
        index_expr = self.rename_indexing(entry.expr)
        index_str = self.sexpr(index_expr)  # type: ignore[misc]
        self.indexing_code.writeline(f"{self.index_dtype} {entry.name} = {index_str};")

    def codegen_body(self) -> None:
        """
        Concatenate output code from loads, compute, stores, suffix into self.body.
        """
        for chunk in (self.loads, self.compute, self.stores):
            self.body.splice(chunk)
            chunk.clear()

    def codegen_kernel(self, name: Optional[str] = None) -> str:
        """Called at the end to generate a final kernel string."""
        self.codegen_body()
        code = IndentedBuffer()
        if V.graph.cpp_wrapper:
            code.writeline('(R"GLSL(')
        else:
            code.writeline('compile_vulkan_shader("""')
        idx_vars = self.active_range_trees()
        with code.indent():
            code.writeline("#version 450")
            for extension in self.extensions:
                code.writeline(f"#extension {extension}")
            code.writeline("layout(std430) buffer;")
            code.writeline(
                "layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;"
            )
            next_binding = 0
            for outer, inner in self.args.output_buffers.items():
                if outer in self.removed_buffers:
                    continue
                dtype_str = self.dtype_to_str(V.graph.get_dtype(outer))
                code.writeline(
                    f"layout(set = 0, binding = {next_binding}) buffer writeonly {inner}Buffer {{{dtype_str} {inner}[];}};"
                )
                next_binding += 1
            for outer, inner in self.args.input_buffers.items():
                dtype = V.graph.get_dtype(outer)
                if dtype == torch.float64:
                    raise RuntimeError("float64 is not supported by Vulkan")
                dtype_str = self.dtype_to_str(dtype)
                code.writeline(
                    f"layout(set = 0, binding = {next_binding}) buffer {inner}Buffer {{{dtype_str} {inner}[];}};"
                )
                next_binding += 1
            if self.args.sizevars:
                code.writeline(
                    "layout(set = 0, binding = {next_binding}) uniform restrict Block {{"
                )
                with code.indent():
                    for outer, inner in self.args.sizevars.items():
                        code.writeline(f"int {inner};")
                code.writeline("};")

            code.writeline("void main() {")
            with code.indent():
                assert len(idx_vars) < 4, "Up to 3 index variables are supported"
                positions = "xyz"
                for idx_var, idx_name in zip(idx_vars, positions):
                    code.writeline(
                        f"{self.index_dtype} {idx_var.name} = {self.index_dtype}(gl_GlobalInvocationID.{idx_name});"
                    )
                # TODO: don't we need to make sure we don't write out of bounds? I don't see this in mps.py.
                # code.writeline("if (gl_GlobalInvocationID.x > 1024) { return; }")
                code.splice(self.indexing_code)
                code.splice(self.body)
            code.writeline("}")
        if V.graph.cpp_wrapper:
            code.writeline(')GLSL");')
        else:
            code.writeline('""")')
        return code.getvalue()

    def call_kernel(self, name: str, node: Any = None) -> None:
        """Generate a call to this kernel."""
        wrapper = V.graph.wrapper_code
        for v in self.args.sizevars:
            wrapper.ensure_size_computed(v)

        # For simplicitly, Inductor-generated kernels accept Tensors on the
        # "vulkan" device. However, they actually require SSBO-backed Vulkan
        # tensors for both input and output, and SSBO-backed tensors are not
        # well-supported by the backend. We accommodate this by copying inputs
        # to SSBO-backed Tensors and allocating SSBO-backed outputs, which we
        # then copy to the allocated outputs. This is egregiously inefficient
        # but unblocks prototyping.
        for key in self.args.input_buffers:
            wrapper.writeline(f'{key}_ssbo_input = {key}.cpu().to("privateuseone:0")')
        for key in self.args.output_buffers:
            wrapper.writeline(f'{key}_ssbo_output = {key}.cpu().to("privateuseone:0")')
        args = [
            *[f"{key}_ssbo_output" for key in self.args.output_buffers.keys()],
            *[f"{key}_ssbo_input" for key in self.args.input_buffers.keys()],
        ]
        args = [arg for arg in args if arg not in self.removed_buffers]
        args += [str(v) for v in self.args.sizevars.keys()]

        num_active_range_trees = len(self.active_range_trees())
        if num_active_range_trees > 0:
            threads = [self.pexpr(v.numel) for v in self.active_range_trees()]

            if V.graph.cpp_wrapper:
                args.append(f"{', '.join(threads)}")
            else:
                args.append(f"threads=[{', '.join(threads)}]")

            # TODO: assess performance implications; these are just guesses.
            group_size = [("64", "1", "1"), ("16", "4", "1"), ("4", "4", "4")][
                num_active_range_trees - 1
            ]
            if V.graph.cpp_wrapper:
                args.append(f"{{{', '.join(group_size)}}}")
            else:
                args.append(f"group_size=[{', '.join(group_size)}]")
        else:
            if V.graph.cpp_wrapper:
                raise RuntimeError("We should always have threads?")

        wrapper.generate_kernel_call(
            name, args, device=torch.device("cpu"), triton=False
        )
        for key in self.args.output_buffers:
            wrapper.writeline(f"{key}.copy_({key}_ssbo_output.cpu())")


class VulkanScheduling(SIMDScheduling):
    kernel_type = VulkanGLSLKernel  # type: ignore[assignment]

    def __init__(self, scheduler: Optional[Scheduler]) -> None:
        super().__init__(scheduler)
        wrapper = V.graph.wrapper_code
        if wrapper is not None and not V.graph.cpp_wrapper:
            wrapper.header.splice(
                "from torch._inductor.runtime.runtime_utils import compile_vulkan_shader"
            )

    def define_kernel(
        self,
        src_code: str,
        node_schedule: list[SchedulerNode],
        kernel: VulkanGLSLKernel,
    ) -> str:
        wrapper = V.graph.wrapper_code
        if src_code in wrapper.src_to_kernel:
            kernel_name = wrapper.src_to_kernel[src_code]
        else:
            vulkan_lib_name = f"vulkan_lib_{wrapper.next_kernel_suffix()}"
            if V.graph.cpp_wrapper:
                raise NotImplementedError()
            else:
                kernel_name = f"{vulkan_lib_name}"
            wrapper.src_to_kernel[src_code] = kernel_name
            origins, detailed_origins = get_kernel_metadata(node_schedule, wrapper)
            metadata_comment = f"{origins}\n{detailed_origins}"
            wrapper.define_kernel(vulkan_lib_name, src_code, metadata_comment)

        return kernel_name
