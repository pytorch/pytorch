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

    @staticmethod
    def _buffer_is_contiguous(buffer_name: str) -> bool:
        buf = V.graph.get_buffer(buffer_name)
        layout = buf.get_layout()
        return layout.is_contiguous()

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
        kernel_params = [a.name for a in arg_defs]
        pure_out_params = [p for p in kernel_params if p.startswith("out_ptr")]
        output_params = [
            p for p in kernel_params if p.startswith(("out_ptr", "in_out_ptr"))
        ]
        if not output_params:
            raise RuntimeError("Pallas backend requires at least one output buffer")

        output_buffer_lookup = {
            inner: outer
            for outer, inner in self.args.output_buffers.items()
            if isinstance(inner, str)
        }

        kernel_name = name or "<KERNEL_NAME>"
        interpret_is_cpu = V.graph.get_current_device_or_throw().type == "cpu"
        interpret_literal = "True" if interpret_is_cpu else "False"

        aliasable_flags: dict[str, bool] = {}
        for param in pure_out_params:
            buffer_name = output_buffer_lookup.get(param)
            is_contiguous = buffer_name is not None and self._buffer_is_contiguous(
                buffer_name
            )
            aliasable_flags[param] = (not interpret_is_cpu) and is_contiguous
        alias_params = [
            f"{param}_alias" for param in pure_out_params if aliasable_flags[param]
        ]
        pointer_tail = [
            p for p in kernel_params if p.startswith(("in_out_ptr", "in_ptr"))
        ]
        kernel_input_params = alias_params + pointer_tail
        full_kernel_params = alias_params + kernel_params
        non_alias_out_set = OrderedSet(
            [name for name, flag in aliasable_flags.items() if not flag]
        )
        copy_output_indices = [
            idx for idx, name in enumerate(output_params) if name in non_alias_out_set
        ]
        self.aliasable_out_ptrs = aliasable_flags
        code.writeline(f"def {kernel_name}_kernel({', '.join(full_kernel_params)}):")
        with code.indent():
            for line in self.compute._lines:
                code.writeline(str(line))
            for line in self.stores._lines:
                code.writeline(str(line))

        jit_wrapper_name = f"{kernel_name}_jit_wrapper"
        donate_indices = []
        for idx, name in enumerate(kernel_input_params):
            if (name in alias_params) or name.startswith("in_out_ptr"):
                donate_indices.append(idx + 2)
        if donate_indices:
            donate_literal = "(" + ", ".join(str(x) for x in donate_indices) + ",)"
        else:
            donate_literal = "()"
        code.writeline(
            "@functools.partial("
            "jax.jit, static_argnums=(0, 1), donate_argnums="
            f"{donate_literal})"
        )
        code.writeline(
            f"def {jit_wrapper_name}(out_shapes, out_dtypes, {', '.join(kernel_input_params)}):"
        )
        with code.indent():
            code.writeline("out_specs = tuple(")
            code.writeline("    jax.ShapeDtypeStruct(shape, dtype)")
            code.writeline("    for shape, dtype in zip(out_shapes, out_dtypes)")
            code.writeline(")")
            alias_pairs: list[tuple[int, int]] = []
            for out_idx, name in enumerate(output_params):
                if name.startswith("out_ptr"):
                    if aliasable_flags.get(name, False):
                        alias_name = f"{name}_alias"
                        input_idx = kernel_input_params.index(alias_name)
                        alias_pairs.append((input_idx, out_idx))
                else:
                    input_idx = kernel_input_params.index(name)
                    alias_pairs.append((input_idx, out_idx))
            alias_map_literal = ", ".join(f"{i}: {o}" for (i, o) in alias_pairs)
            code.writeline("return pl.pallas_call(")
            code.writeline(f"    {kernel_name}_kernel,")
            code.writeline("    out_shape=out_specs,")
            code.writeline(f"    interpret={interpret_literal},")
            code.writeline("    grid=(1,),")
            code.writeline(
                f"    input_output_aliases={{ {alias_map_literal} }},"
                if alias_pairs
                else "    input_output_aliases={},"
            )
            code.writeline(")(")
            code.writeline(f"    {', '.join(kernel_input_params)},")
            code.writeline(")")

        main_name = f"{kernel_name}_main"
        code.writeline(
            f"def {main_name}({', '.join(full_kernel_params)}, stream=None):"
        )
        with code.indent():
            if alias_params:
                code.writeline("# Convert Torch -> JAX for donated outputs")
                for alias_name in alias_params:
                    code.writeline(
                        f"{alias_name}_jax = jax.dlpack.from_dlpack({alias_name})"
                    )
            code.writeline("# Convert Torch -> JAX for in-place tensors")
            for ptr in pointer_tail:
                if ptr.startswith("in_out_ptr"):
                    code.writeline(f"{ptr}_jax = jax.dlpack.from_dlpack({ptr})")
            code.writeline("# Convert Torch -> JAX for inputs")
            for ptr in pointer_tail:
                if ptr.startswith("in_ptr"):
                    code.writeline(f"{ptr}_jax = jax.dlpack.from_dlpack({ptr})")

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
            code.writeline(
                "out_shapes = ("
                + ", ".join([f"tuple({name}.shape)" for name in output_params])
                + ",)"
            )
            code.writeline(
                "out_dtypes = ("
                + ", ".join(
                    [f"_torch_dtype_to_jax[{name}.dtype]" for name in output_params]
                )
                + ",)"
            )
            arg_name_map: dict[str, str] = {}
            for alias_name in alias_params:
                arg_name_map[alias_name] = f"{alias_name}_jax"
            for ptr in pointer_tail:
                arg_name_map[ptr] = f"{ptr}_jax"

            if kernel_input_params:
                alias_args_str = ", ".join(
                    arg_name_map[name] for name in kernel_input_params
                )
                code.writeline(
                    f"res = {jit_wrapper_name}(out_shapes, out_dtypes, {alias_args_str})"
                )
            else:
                code.writeline(f"res = {jit_wrapper_name}(out_shapes, out_dtypes)")
            if copy_output_indices:
                code.writeline(
                    "result_values = res if isinstance(res, tuple) else (res,)"
                )
                for idx in copy_output_indices:
                    name = output_params[idx]
                    code.writeline(
                        f"{name}.copy_(torch.from_dlpack(result_values[{idx}]))"
                    )

        return code.getvalue()

    def call_kernel(self, name: str, node: Optional[IRNode] = None) -> None:  # type: ignore[override]
        """Generate the Python code that calls this Pallas kernel."""
        wrapper = V.graph.wrapper_code
        arg_defs, call_args, _, _ = self.args.python_argdefs()
        kernel_param_names = [a.name for a in arg_defs]
        pure_out_params = [p for p in kernel_param_names if p.startswith("out_ptr")]
        call_arg_strs = list(map(str, call_args))
        aliasable = getattr(self, "aliasable_out_ptrs", {})
        alias_call_args = [
            call_arg_strs[kernel_param_names.index(p)]
            for p in pure_out_params
            if aliasable.get(p, False)
        ]

        # Generate kernel call: kernel_name.run(arg1, arg2, ...)
        # Note: async_compile.pallas loads {name}_main function and wraps it in PallasKernelWrapper
        # which exposes a run() method
        kernel_call = f"{name}.run({', '.join(alias_call_args + call_arg_strs)})"
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
