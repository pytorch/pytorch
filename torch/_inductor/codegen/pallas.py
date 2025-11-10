# mypy: allow-untyped-defs
from __future__ import annotations

import hashlib
from typing import Any, Optional, TYPE_CHECKING

import sympy  # noqa: TC002

import torch  # noqa: TC001
from torch.utils._ordered_set import OrderedSet

from .. import config
from ..utils import get_fused_kernel_name, get_kernel_metadata
from ..virtualized import V
from .common import BackendFeature, CSEVariable, IndentedBuffer, OpOverrides
from .simd import SIMDKernel, SIMDScheduling


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ..ir import IRNode
    from ..scheduler import BaseSchedulerNode


# Main function suffix used in generated Pallas code
MAIN_SUFFIX = "main"

# Logger for Pallas kernel code
kernel_code_log = torch._logging.getArtifactLogger(__name__, "kernel_code")


class PallasKernelWrapper:
    """Wrapper to provide .run() interface for Pallas kernels"""

    def __init__(
        self, kernel_fn: Callable[..., Any], kernel_path: Optional[str] = None
    ):
        self.kernel_fn = kernel_fn
        self.kernel_path = kernel_path
        kernel_code_log.info("Pallas kernel path: %s", kernel_path)

    def run(self, *args, stream=None, **kwargs):
        """
        Execute the Pallas kernel.

        Args:
            *args: Arguments to pass to the kernel function
            stream: CUDA stream to pass to the kernel function
            **kwargs: Additional keyword arguments for the kernel

        Returns:
            Result of the kernel execution
        """
        return self.kernel_fn(*args, stream=stream, **kwargs)


class Unsupported(RuntimeError):
    """Exception raised when an operation is not supported by the Pallas backend."""


class PallasKernelOverrides(OpOverrides):
    """
    Map element-wise ops to JAX/Pallas operations.

    For now, we use the default Python operators which are compatible
    with JAX numpy broadcasting semantics.
    """

    @staticmethod
    def sin(x: str) -> str:
        return f"jnp.sin({x})"

    @staticmethod
    def cos(x: str) -> str:
        return f"jnp.cos({x})"

    @staticmethod
    def tan(x: str) -> str:
        return f"jnp.tan({x})"

    @staticmethod
    def sinh(x: str) -> str:
        return f"jnp.sinh({x})"

    @staticmethod
    def cosh(x: str) -> str:
        return f"jnp.cosh({x})"

    @staticmethod
    def tanh(x: str) -> str:
        return f"jnp.tanh({x})"

    @staticmethod
    def asin(x: str) -> str:
        return f"jnp.arcsin({x})"

    @staticmethod
    def acos(x: str) -> str:
        return f"jnp.arccos({x})"

    @staticmethod
    def atan(x: str) -> str:
        return f"jnp.arctan({x})"

    @staticmethod
    def exp(x: str) -> str:
        return f"jnp.exp({x})"

    @staticmethod
    def exp2(x: str) -> str:
        return f"jnp.exp2({x})"

    @staticmethod
    def expm1(x: str) -> str:
        return f"jnp.expm1({x})"

    @staticmethod
    def log(x: str) -> str:
        return f"jnp.log({x})"

    @staticmethod
    def log10(x: str) -> str:
        return f"jnp.log10({x})"

    @staticmethod
    def log2(x: str) -> str:
        return f"jnp.log2({x})"

    @staticmethod
    def log1p(x: str) -> str:
        return f"jnp.log1p({x})"

    @staticmethod
    def sqrt(x: str) -> str:
        return f"jnp.sqrt({x})"

    @staticmethod
    def rsqrt(x: str) -> str:
        return f"(1.0 / jnp.sqrt({x}))"

    @staticmethod
    def abs(x: str) -> str:
        return f"jnp.abs({x})"

    @staticmethod
    def neg(x: str) -> str:
        return f"(-{x})"

    @staticmethod
    def floor(x: str) -> str:
        return f"jnp.floor({x})"

    @staticmethod
    def ceil(x: str) -> str:
        return f"jnp.ceil({x})"

    @staticmethod
    def trunc(x: str) -> str:
        return f"jnp.trunc({x})"

    @staticmethod
    def round(x: str) -> str:
        return f"jnp.round({x})"

    @staticmethod
    def sigmoid(x: str) -> str:
        return f"(1.0 / (1.0 + jnp.exp(-{x})))"

    @staticmethod
    def relu(x: str) -> str:
        return f"jnp.maximum({x}, 0)"

    @staticmethod
    def pow(a: str, b: str) -> str:
        return f"jnp.power({a}, {b})"

    @staticmethod
    def maximum(a: str, b: str) -> str:
        return f"jnp.maximum({a}, {b})"

    @staticmethod
    def minimum(a: str, b: str) -> str:
        return f"jnp.minimum({a}, {b})"

    @staticmethod
    def where(cond: str, a: str, b: str) -> str:
        return f"jnp.where({cond}, {a}, {b})"


class PallasKernel(SIMDKernel):
    """
    Minimal Pallas kernel for simple elementwise operations.

    Strategy:
    - Treat loads as full-array refs: "in_ptrX[...]"
    - Compute expression with Python operators (compatible with jax.numpy broadcasting)
    - Store as full-array ref assignment: "out_ptrY[...] = <expr>"
    - Generate Python code that defines a Pallas kernel and a host entrypoint.
    - Use async_compile.pallas path to compile and load Python code.
    """

    overrides = PallasKernelOverrides  # type: ignore[assignment]

    def _get_contiguous_index_str(self, index: sympy.Expr) -> str:
        """
        Validate that the index represents contiguous access and return the indexing string.

        For Pallas, we only support simple contiguous access patterns where the index
        is a single symbol (e.g., xindex) representing a flattened iteration.
        This ensures the load/store order is contiguous.

        Args:
            index: The indexing expression to validate

        Returns:
            The indexing string to use (currently always "...")

        Raises:
            Unsupported: If the index is not a simple contiguous pattern
        """
        # Prepare and simplify the index
        prepared_index = self.prepare_indexing(index)

        # For contiguous access, we expect a single symbol (like xindex)
        # or a simple integer (for scalar operations)
        if isinstance(prepared_index, sympy.Symbol):
            # This is the expected case: a single symbol representing contiguous iteration
            return "..."
        elif prepared_index.is_Integer:
            # Scalar case
            return "..."
        else:
            # If there's any complex expression (ModularIndexing, FloorDiv, etc.),
            # it's not a simple contiguous pattern
            raise Unsupported(
                f"Pallas backend only supports contiguous access patterns. "
                f"Got complex index: {prepared_index}"
            )

    def load(self, name: str, index: sympy.Expr) -> CSEVariable:  # type: ignore[override]
        buf = self.args.input(name)
        dtype = V.graph.get_dtype(name)
        # Validate contiguous access and get index string
        index_str = self._get_contiguous_index_str(index)
        # Pallas refs must be unpacked with [...] to load the array
        return self.cse.generate(
            self.compute,
            f"{buf}[{index_str}]",
            dtype=dtype,
        )

    def store(
        self, name: str, index: sympy.Expr, value: CSEVariable, mode: Any = None
    ) -> None:  # type: ignore[override]
        if mode is not None:
            raise Unsupported("pallas store mode not supported")
        out = self.args.output(name)
        self.store_buffer_names.add(name)
        # Validate contiguous access and get index string
        index_str = self._get_contiguous_index_str(index)
        # Pallas refs must use [...] assignment to store back to the ref
        self.stores.writeline(f"{out}[{index_str}] = {value}")

    def codegen_kernel(self, name: Optional[str] = None) -> str:  # type: ignore[override]
        """
        Generate the complete Pallas kernel code as a Python string.

        This includes:
        - Import statements for JAX/Pallas
        - The kernel function that operates on refs
        - The main wrapper function that handles PyTorch<->JAX conversions via DLPack

        Args:
            name: Optional kernel name (will use placeholder if not provided)

        Returns:
            str: Complete Python source code for the Pallas kernel
        """
        # Ensure one (1) output for now
        live_outs = list(self.args.live_output_buffers())
        if len(live_outs) != 1:
            raise Unsupported(
                "Pallas backend currently supports single-output elementwise kernels only"
            )

        code = IndentedBuffer()
        code.splice(
            """
            import functools
            import torch
            import jax
            import jax.numpy as jnp
            from jax.experimental import pallas as pl
            """,
            strip=True,
        )

        # Define the Pallas kernel: accepts refs, uses broadcasted expressions
        arg_defs, _, _, _ = self.args.python_argdefs()
        # Order: inputs (in_ptr*), then outputs (out_ptr*), then sizes/workspaces
        kernel_params = [a.name for a in arg_defs]

        kernel_name = name or "<KERNEL_NAME>"
        interpret_literal = (
            "True" if V.graph.get_current_device_or_throw().type == "cpu" else "False"
        )
        code.writeline(f"def {kernel_name}_kernel({', '.join(kernel_params)}):")
        with code.indent():
            # Emit compute (CSE) and store lines; they reference *_ptr[...] directly
            for line in self.compute._lines:
                code.writeline(str(line))
            for line in self.stores._lines:
                code.writeline(str(line))

        jit_wrapper_name = f"{kernel_name}_jit_wrapper"
        code.writeline("@functools.partial(jax.jit, static_argnums=(0, 1))")
        code.writeline(f"def {jit_wrapper_name}(out_shape, out_dtype, *kernel_refs):")
        with code.indent():
            code.writeline("out_spec = jax.ShapeDtypeStruct(out_shape, out_dtype)")
            code.writeline("return pl.pallas_call(")
            code.writeline(f"    {kernel_name}_kernel,")
            code.writeline("    out_shape=out_spec,")
            code.writeline(f"    interpret={interpret_literal},")
            code.writeline("    grid=(1,),")
            code.writeline(")(*kernel_refs)")

        # Host entry: convert torch tensors <-> jax, call pallas_call and copy back
        main_name = f"{kernel_name}_main"
        code.writeline(f"def {main_name}({', '.join(kernel_params)}, stream=None):")
        with code.indent():
            # Identify inputs (in_ptr*) and output (out_ptr*)
            input_params = [
                p for p in kernel_params if p.startswith(("in_ptr", "in_out_ptr"))
            ]
            output_params = [p for p in kernel_params if p.startswith("out_ptr")]

            if len(output_params) != 1:
                raise RuntimeError(
                    f"Expected exactly 1 output, got {len(output_params)}"
                )

            output_param = output_params[0]

            # Convert inputs to JAX arrays
            code.writeline("# Convert Torch -> JAX for inputs")
            for inp in input_params:
                code.writeline(f"{inp}_jax = jax.dlpack.from_dlpack({inp})")

            # Get output metadata from PyTorch tensor
            code.writeline("# Prepare output metadata from PyTorch tensor")
            code.writeline("# Map PyTorch dtype to JAX dtype")
            code.writeline("_torch_dtype_to_jax = {")
            code.writeline(
                "    torch.float32: jnp.float32, torch.float64: jnp.float64, torch.float16: jnp.float16,"
            )
            code.writeline(
                "    torch.int32: jnp.int32, torch.int64: jnp.int64, torch.int16: jnp.int16, torch.int8: jnp.int8,"
            )
            code.writeline("    torch.uint8: jnp.uint8, torch.bool: jnp.bool_,")
            code.writeline("}")
            code.writeline(f"out_shape = tuple({output_param}.shape)")
            code.writeline(f"out_dtype = _torch_dtype_to_jax[{output_param}.dtype]")

            call_args = ["out_shape", "out_dtype"] + [
                f"{inp}_jax" for inp in input_params
            ]
            call_arg_str = ", ".join(call_args)
            code.writeline(f"res = {jit_wrapper_name}({call_arg_str})")

            # Copy result back
            code.writeline("# Copy result back into the provided torch output tensor")
            code.writeline("res_t = torch.from_dlpack(res)")
            code.writeline(f"{output_param}.copy_(res_t)")

        return code.getvalue()

    def call_kernel(self, name: str, node: Optional[IRNode] = None) -> None:  # type: ignore[override]
        """Generate the Python code that calls this Pallas kernel."""
        wrapper = V.graph.wrapper_code
        _, call_args, _, arg_types = self.args.python_argdefs()

        # Generate kernel call: kernel_name.run(arg1, arg2, ...)
        # Note: async_compile.pallas loads {name}_main function and wraps it in PallasKernelWrapper
        # which exposes a run() method
        kernel_call = f"{name}.run({', '.join(map(str, call_args))})"
        wrapper.writeline(kernel_call)


class PallasScheduling(SIMDScheduling):
    kernel_type = PallasKernel  # type: ignore[assignment]

    @classmethod
    def get_backend_features(cls, device: torch.device) -> OrderedSet[BackendFeature]:
        # Start minimal: no special features advertised
        return OrderedSet()

    def define_kernel(
        self,
        src_code: str,
        node_schedule: Sequence[BaseSchedulerNode],
        kernel: PallasKernel,
    ) -> str:  # type: ignore[override]
        wrapper = V.graph.wrapper_code
        if src_code in wrapper.src_to_kernel:
            return wrapper.src_to_kernel[src_code]

        fused_name = (
            get_fused_kernel_name(node_schedule, config.triton.descriptive_names)
            if config.triton.descriptive_names
            else ""
        )
        kernel_hash = hashlib.sha256(src_code.encode("utf-8")).hexdigest()[:8]
        if fused_name == "fused":
            kernel_name = f"pallas_{kernel_hash}"
        else:
            kernel_name = f"pallas_{fused_name}_{kernel_hash}"
        wrapper.src_to_kernel[src_code] = kernel_name

        # Replace placeholder if any
        src_code = src_code.replace("<KERNEL_NAME>", kernel_name)

        compile_wrapper = IndentedBuffer()
        compile_wrapper.writeline(f"async_compile.pallas({kernel_name!r}, r'''")
        compile_wrapper.splice(src_code, strip=True)
        compile_wrapper.writeline("''')")

        origins, detailed_origins = get_kernel_metadata(node_schedule, wrapper)
        metadata_comment = f"{origins}\n{detailed_origins}"
        wrapper.define_kernel(kernel_name, compile_wrapper.getvalue(), metadata_comment)

        return kernel_name
