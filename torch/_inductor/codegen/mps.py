# This is not a feature-complete compiler backend
# Just an early prototype that shows that one can compile elementwise ops into a Metal shader
from typing import Any, Optional

import sympy

import torch

from ..ops_handler import StoreMode
from ..scheduler import SchedulerNode
from ..utils import get_kernel_metadata
from ..virtualized import V
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
    def where(a: CSEVariable, b: CSEVariable, c: CSEVariable) -> str:
        return f"{a} ? {b} : {c}"

    @staticmethod
    def logical_or(a: CSEVariable, b: CSEVariable) -> str:
        return f"{a} | {b}"

    @staticmethod
    def logical_and(a: CSEVariable, b: CSEVariable) -> str:
        return f"{a} & {b}"

    @staticmethod
    def abs(x: CSEVariable) -> str:
        return f"metal::abs({x})"

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


class MetalKernel(SIMDKernel):
    overrides = MetalOverrides  # type: ignore[assignment]
    suffix = ";"
    newvar_prefix = "auto "

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
        line = f"{var}[{index}]"
        return self.cse.generate(self.body, line)

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
        code.writeline('torch.mps._compile_shader("""')
        with code.indent():
            code.writeline("kernel void kernel_0(")
            with code.indent():
                for outer, inner in self.args.output_buffers.items():
                    if outer in self.removed_buffers:
                        continue
                    dtype_str = self.dtype_to_str(V.graph.get_dtype(outer))
                    code.writeline(f"device {dtype_str}* {inner},")
                for outer, inner in self.args.input_buffers.items():
                    dtype_str = self.dtype_to_str(V.graph.get_dtype(outer))
                    code.writeline(f"constant {dtype_str}* {inner},")
                code.writeline("uint xindex [[thread_position_in_grid]]")
            code.writeline(") {")
            with code.indent():
                code.splice(self.body)
            code.writeline("}")
        code.writeline('""")')

        return code.getvalue()

    def call_kernel(self, name: str, node: Any = None) -> None:
        """Codegen a call to this kernel"""
        wrapper = V.graph.wrapper_code
        args = [*self.args.output_buffers.keys(), *self.args.input_buffers.keys()]
        args = [arg for arg in args if arg not in self.removed_buffers]
        wrapper.generate_kernel_call(
            name,
            args,
            gpu=False,  # TODO: Fix me, MPS does not expose streams now
            triton=False,
        )


class MetalScheduling(SIMDScheduling):
    kernel_type = MetalKernel  # type: ignore[assignment]

    def define_kernel(
        self, src_code: str, node_schedule: list[SchedulerNode], kernel: MetalKernel
    ) -> str:
        wrapper = V.graph.wrapper_code
        if src_code in wrapper.src_to_kernel:
            kernel_name = wrapper.src_to_kernel[src_code]
        else:
            kernel_name = f"mps_lib.kernel_{wrapper.next_kernel_suffix()}"
            wrapper.src_to_kernel[src_code] = kernel_name
            origins, detailed_origins = get_kernel_metadata(node_schedule, wrapper)
            metadata_comment = f"{origins}\n{detailed_origins}"
            wrapper.define_kernel("mps_lib", src_code, metadata_comment)

        return kernel_name
