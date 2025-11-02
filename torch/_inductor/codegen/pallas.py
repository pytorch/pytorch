import hashlib
from typing import Any, Optional

import sympy

import torch
from torch.utils._ordered_set import OrderedSet

from .. import config
from ..codecache import code_hash
from ..utils import get_fused_kernel_name, get_kernel_metadata
from ..virtualized import V
from .common import (
    BackendFeature,
    CSEVariable,
    DeferredLine,
    IndentedBuffer,
    OpOverrides,
)
from .simd import IterationRangesEntry, SIMDKernel, SIMDScheduling


class PallasKernelOverrides(OpOverrides):
    """
    Map element-wise ops to JAX/Pallas operations.

    For now, we use the default Python operators which are compatible
    with JAX numpy broadcasting semantics.
    """

    @staticmethod
    def sin(x):
        return f"jnp.sin({x})"

    @staticmethod
    def cos(x):
        return f"jnp.cos({x})"

    @staticmethod
    def tan(x):
        return f"jnp.tan({x})"

    @staticmethod
    def sinh(x):
        return f"jnp.sinh({x})"

    @staticmethod
    def cosh(x):
        return f"jnp.cosh({x})"

    @staticmethod
    def tanh(x):
        return f"jnp.tanh({x})"

    @staticmethod
    def asin(x):
        return f"jnp.arcsin({x})"

    @staticmethod
    def acos(x):
        return f"jnp.arccos({x})"

    @staticmethod
    def atan(x):
        return f"jnp.arctan({x})"

    @staticmethod
    def exp(x):
        return f"jnp.exp({x})"

    @staticmethod
    def exp2(x):
        return f"jnp.exp2({x})"

    @staticmethod
    def expm1(x):
        return f"jnp.expm1({x})"

    @staticmethod
    def log(x):
        return f"jnp.log({x})"

    @staticmethod
    def log10(x):
        return f"jnp.log10({x})"

    @staticmethod
    def log2(x):
        return f"jnp.log2({x})"

    @staticmethod
    def log1p(x):
        return f"jnp.log1p({x})"

    @staticmethod
    def sqrt(x):
        return f"jnp.sqrt({x})"

    @staticmethod
    def rsqrt(x):
        return f"(1.0 / jnp.sqrt({x}))"

    @staticmethod
    def abs(x):
        return f"jnp.abs({x})"

    @staticmethod
    def neg(x):
        return f"(-{x})"

    @staticmethod
    def floor(x):
        return f"jnp.floor({x})"

    @staticmethod
    def ceil(x):
        return f"jnp.ceil({x})"

    @staticmethod
    def trunc(x):
        return f"jnp.trunc({x})"

    @staticmethod
    def round(x):
        return f"jnp.round({x})"

    @staticmethod
    def sigmoid(x):
        return f"(1.0 / (1.0 + jnp.exp(-{x})))"

    @staticmethod
    def relu(x):
        return f"jnp.maximum({x}, 0)"

    @staticmethod
    def pow(a, b):
        return f"jnp.power({a}, {b})"

    @staticmethod
    def maximum(a, b):
        return f"jnp.maximum({a}, {b})"

    @staticmethod
    def minimum(a, b):
        return f"jnp.minimum({a}, {b})"

    @staticmethod
    def where(cond, a, b):
        return f"jnp.where({cond}, {a}, {b})"


class PallasKernel(SIMDKernel):
    """
    Minimal Pallas kernel for simple elementwise operations.

    Strategy:
    - Treat loads as full-array refs: "in_ptrX[...]"
    - Compute expression with Python operators (compatible with jax.numpy broadcasting)
    - Store as full-array ref assignment: "out_ptrY[...] = <expr>"
    - Generate Python code that defines a Pallas kernel and a host entrypoint.
    - Use async_compile.cutedsl path to compile and load Python code (generic wrapper).
    """

    overrides = PallasKernelOverrides  # type: ignore[assignment]

    def load(self, name: str, index: sympy.Expr) -> CSEVariable:  # type: ignore[override]
        buf = self.args.input(name)
        dtype = V.graph.get_dtype(name)
        # Pallas refs must be unpacked with [...] to load the array
        return self.cse.generate(
            self.compute,
            f"{buf}[...]",
            dtype=dtype,
        )

    def store(
        self, name: str, index: sympy.Expr, value: CSEVariable, mode: Any = None
    ) -> None:  # type: ignore[override]
        if mode is not None:
            raise Unsupported("pallas store mode not supported")
        out = self.args.output(name)
        self.store_buffer_names.add(name)
        # Pallas refs must use [...] assignment to store back to the ref
        self.stores.writeline(f"{out}[...] = {value}")

    def codegen_kernel(self, name: Optional[str] = None) -> str:  # type: ignore[override]
        # Ensure one (1) output for now
        live_outs = list(self.args.live_output_buffers())
        if len(live_outs) != 1:
            raise Unsupported(
                "Pallas backend currently supports single-output elementwise kernels only"
            )

        code = IndentedBuffer()
        code.splice(
            """
            import torch
            import jax
            import jax.numpy as jnp
            from jax.experimental import pallas as pl
            from torch.utils import dlpack as torch_dlpack
            """,
            strip=True,
        )

        # Define the Pallas kernel: accepts refs, uses broadcasted expressions
        arg_defs, _, _, _ = self.args.python_argdefs()
        # Order: inputs (in_ptr*), then outputs (out_ptr*), then sizes/workspaces
        kernel_params = [a.name for a in arg_defs]

        kernel_name = name or "<KERNEL_NAME>"
        code.writeline(f"def {kernel_name}_kernel({', '.join(kernel_params)}):")
        with code.indent():
            # Emit compute (CSE) and store lines; they reference *_ptr[...] directly
            for line in self.compute._lines:
                code.writeline(str(line))
            for line in self.stores._lines:
                code.writeline(str(line))

        # Host entry: convert torch tensors <-> jax, call pallas_call and copy back
        main_name = f"{kernel_name}_main"
        code.writeline(f"def {main_name}({', '.join(kernel_params)}, stream=None):")
        with code.indent():
            # Identify inputs (in_ptr*) and output (out_ptr*)
            input_params = [
                p
                for p in kernel_params
                if p.startswith("in_ptr") or p.startswith("in_out_ptr")
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
                code.writeline(
                    f"{inp}_jax = jax.dlpack.from_dlpack(torch_dlpack.to_dlpack({inp}))"
                )

            # Get output spec from PyTorch tensor
            code.writeline("# Prepare output spec from PyTorch tensor")
            code.writeline("# Map PyTorch dtype to JAX dtype string")
            code.writeline(f"_torch_dtype_to_jax = {{")
            code.writeline(
                f"    torch.float32: jnp.float32, torch.float64: jnp.float64, torch.float16: jnp.float16,"
            )
            code.writeline(
                f"    torch.int32: jnp.int32, torch.int64: jnp.int64, torch.int16: jnp.int16, torch.int8: jnp.int8,"
            )
            code.writeline(f"    torch.uint8: jnp.uint8, torch.bool: jnp.bool_,")
            code.writeline(f"}}")
            code.writeline(
                f"out_spec = jax.ShapeDtypeStruct({output_param}.shape, _torch_dtype_to_jax[{output_param}.dtype])"
            )

            # Call pallas
            code.writeline("compiled = pl.pallas_call(")
            code.writeline(f"    lambda *refs: {kernel_name}_kernel(*refs),")
            code.writeline("    out_shape=out_spec,")
            code.writeline("    grid=(1,),")
            code.writeline(")")

            jax_input_args = ", ".join([f"{inp}_jax" for inp in input_params])
            code.writeline(f"res = compiled({jax_input_args})")

            # Copy result back
            code.writeline("# Copy result back into the provided torch output tensor")
            code.writeline(
                f"res_t = torch_dlpack.from_dlpack(jax.dlpack.to_dlpack(res))"
            )
            code.writeline(f"{output_param}.copy_(res_t)")

        return code.getvalue()

    def call_kernel(self, name: str, node=None) -> None:  # type: ignore[override]
        """Generate the Python code that calls this Pallas kernel."""
        wrapper = V.graph.wrapper_code
        _, call_args, _, arg_types = self.args.python_argdefs()

        # Generate kernel call: kernel_name.run(arg1, arg2, ...)
        # Note: cutedsl loads {name}_main function and wraps it in CuteDSLKernelWrapper
        # which exposes a run() method
        kernel_call = f"{name}.run({', '.join(map(str, call_args))})"
        wrapper.writeline(kernel_call)


class PallasScheduling(SIMDScheduling):
    kernel_type = PallasKernel  # type: ignore[assignment]

    @classmethod
    def get_backend_features(
        cls, device: torch.device
    ) -> "Optional[set[BackendFeature]]":
        # Start minimal: no special features advertised
        return OrderedSet()

    def define_kernel(self, src_code: str, node_schedule, kernel: PallasKernel) -> str:  # type: ignore[override]
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
        compile_wrapper.writeline(f"async_compile.cutedsl({kernel_name!r}, r'''")
        compile_wrapper.splice(src_code, strip=True)
        compile_wrapper.writeline("''')")

        origins, detailed_origins = get_kernel_metadata(node_schedule, wrapper)
        metadata_comment = f"{origins}\n{detailed_origins}"
        wrapper.define_kernel(kernel_name, compile_wrapper.getvalue(), metadata_comment)

        return kernel_name
