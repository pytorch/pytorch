# mypy: allow-untyped-defs
# This is not a feature-complete compiler backend end
# Just an early prototype that shows that one can compile
# Easy models to Metal
import sympy

import torch

from ..ops_handler import StoreMode
from ..utils import get_kernel_metadata
from ..virtualized import V
from .common import CSEVariable, DeferredLine, IndentedBuffer, OpOverrides
from .simd import SIMDKernel, SIMDScheduling


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
        x,
        dtype: torch.dtype,
        src_dtype: torch.dtype | None = None,
        use_compute_types=True,
    ):
        return f"static_cast<{DTYPE_TO_METAL[dtype]}>({x})"

    @staticmethod
    def logical_or(a, b):
        return f"{a} | {b}"

    @staticmethod
    def atan(x):
        return f"metal::atan({x})"

    @staticmethod
    def sin(x):
        return f"metal::sin({x})"

    @staticmethod
    def cos(x):
        return f"metal::cos({x})"

    @staticmethod
    def acos(x):
        return f"metal::acos({x})"


class MetalKernel(SIMDKernel):
    overrides = MetalOverrides  # type: ignore[assignment]
    suffix = ";"
    newvar_prefix = "auto "

    def __init__(
        self,
        tiling: dict[str, sympy.Expr],
        **kwargs,
    ) -> None:
        super().__init__(tiling, **kwargs)
        self.compute = self.body
        self.loads = self.body
        self.stores = self.body

    def dtype_to_str(self, dtype: torch.dtype) -> str:
        return DTYPE_TO_METAL[dtype]

    def load(self, name: str, index: sympy.Expr):
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

    def codegen_kernel(self, name=None):
        """Called at the end to generate a final kernel string"""
        code = IndentedBuffer()
        code.writeline('torch.mps._compile_shader("""')
        with code.indent():
            code.writeline("kernel void kernel_0(")
            with code.indent():
                for outer, inner in self.args.output_buffers.items():
                    dtype_str = self.dtype_to_str(V.graph.get_dtype(outer))
                    code.writeline(f"device {dtype_str}* {inner},")
                for outer, inner in self.args.input_buffers.items():
                    dtype_str = self.dtype_to_str(V.graph.get_dtype(outer))
                    code.writeline(f"constant {dtype_str}* {inner},")
                code.writeline("uint x0 [[thread_position_in_grid]]")
            code.writeline(") {")
            with code.indent():
                code.splice(self.body)
            code.writeline("}")
        code.writeline('""")')

        return code.getvalue()

    def call_kernel(self, name: str, node=None):
        """Codegen a call to this kernel"""
        wrapper = V.graph.wrapper_code
        args = list(self.args.output_buffers.keys()) + list(
            self.args.input_buffers.keys()
        )
        wrapper.generate_kernel_call(
            name,
            args,
            gpu=False,  # TODO: Fix me
            triton=False,
        )


class MetalScheduling(SIMDScheduling):
    kernel_type = MetalKernel  # type: ignore[assignment]

    def define_kernel(self, src_code, node_schedule, kernel):
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
