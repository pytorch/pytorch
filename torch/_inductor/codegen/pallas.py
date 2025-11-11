# mypy: allow-untyped-defs
from __future__ import annotations

import hashlib
from typing import Any, Optional, TYPE_CHECKING

import sympy  # noqa: TC002

import torch  # noqa: TC001
from torch.utils._ordered_set import OrderedSet

from .. import config
from ..runtime.runtime_utils import torch_dtype_to_jax
from ..utils import get_fused_kernel_name, get_kernel_metadata
from ..virtualized import V
from .block_analysis import BlockPatternMatcher
from .common import BackendFeature, CSEVariable, IndentedBuffer, OpOverrides
from .simd import pexpr, SIMDKernel, SIMDScheduling


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

    @staticmethod
    def to_dtype(
        x: str,
        dtype: torch.dtype,
        src_dtype: Optional[torch.dtype] = None,
        use_compute_types: bool = True,
    ) -> str:
        jax_dtype = torch_dtype_to_jax(dtype)
        # Wrap in jnp.asarray to handle scalars from integer indexing
        return f"jnp.asarray({x}).astype({jax_dtype})"

    @staticmethod
    def index_expr(expr: sympy.Expr, dtype: torch.dtype) -> str:
        """Convert a sympy expression to a JAX array indexing expression."""
        from ..utils import get_bounds_index_expr

        idx_str = V.kernel.kexpr(V.kernel.prepare_indexing(expr))
        var = V.kernel.cse.generate(
            V.kernel.compute, idx_str, bounds=get_bounds_index_expr(expr)
        )
        return PallasKernelOverrides.to_dtype(var, dtype)

    @staticmethod
    def constant(val, dtype: torch.dtype) -> str:
        """Convert a constant value to JAX representation."""
        jax_dtype = torch_dtype_to_jax(dtype)
        if dtype == torch.bool:
            return "True" if val else "False"
        return f"jnp.array({val}, dtype={jax_dtype})"


class PallasKernel(SIMDKernel):
    """
    Pallas kernel for elementwise operations with support for strided/scatter access.

    Strategy:
    - Convert index expressions to JAX-compatible array slicing
    - Load/store using indexed access: "in_ptrX[slice]" or full-array "in_ptrX[...]"
    - Compute expression with Python operators (compatible with jax.numpy broadcasting)
    - Generate Python code that defines a Pallas kernel and a host entrypoint.
    - Use async_compile.pallas path to compile and load Python code.
    """

    overrides = PallasKernelOverrides  # type: ignore[assignment]
    kexpr: Callable[[sympy.Expr], str] = pexpr  # Use Python expression printer

    def check_bounds(
        self, expr: sympy.Expr, size: sympy.Expr, lower: bool, upper: bool
    ) -> None:
        """Check array bounds for indirect indexing."""
        # For now, skip explicit bounds checking as JAX/Pallas handles this internally
        # TODO: Implement explicit bounds checking with assertions if needed

    def _get_index_str(self, index: sympy.Expr) -> str:
        """
        Convert an index expression to a string suitable for Pallas indexing.

        Pallas operates on full arrays, so we need to convert index expressions
        to JAX array slicing. For example:
        - x0 -> "..." (contiguous access, full array)
        - 2*x0 -> "::2" (strided access with stride 2)
        - 2*x0 + 1 -> "1::2" (strided access with offset 1, stride 2)

        Args:
            index: The indexing expression to convert

        Returns:
            The indexing string to use in generated code
        """
        # Prepare and simplify the index
        prepared_index = self.prepare_indexing(index)

        # For simple single-symbol access (contiguous case), we can use [...]
        # which is more efficient as it operates on the entire array at once
        if isinstance(prepared_index, sympy.Symbol):
            return "..."
        elif prepared_index.is_Integer:
            # Scalar index
            return str(prepared_index)
        else:
            # Complex expression (strided/scatter access)
            # Try to extract stride and offset for common patterns
            return self._convert_to_jax_slice(prepared_index)

    def _convert_to_jax_slice(self, index: sympy.Expr) -> str:
        """
        Convert a sympy index expression to JAX slice notation.

        Handles common patterns like:
        - stride*var -> ::stride
        - stride*var + offset -> offset::stride

        For more complex patterns, falls back to explicit indexing.
        Uses BlockPatternMatcher for robust pattern matching.
        """
        # Get the iteration variables for this kernel
        if not self.range_trees:
            return "..."

        # Simplify the index
        index = V.graph.sizevars.simplify(index)
        free_symbols = index.free_symbols

        # Get iteration variables from range_tree_nodes
        iter_vars = OrderedSet(self.range_tree_nodes.keys())

        # Find which iteration variable(s) are used
        used_vars = free_symbols & iter_vars

        if len(used_vars) == 0:
            # No iteration variables, this is a constant index
            return str(index)
        elif len(used_vars) == 1:
            # Single iteration variable - try to extract stride and offset using BlockPatternMatcher
            var = next(iter(used_vars))

            # Get the subexpression involving this variable
            var_expr = BlockPatternMatcher.get_subexpr_involving_symbol(index, var)

            # Try to match affine pattern: stride * var
            stride = BlockPatternMatcher.match_affine_block_expr(var_expr, var)

            if stride is not None:
                # Extract the constant offset (terms not involving var)
                offset = index - var_expr
                offset = V.graph.sizevars.simplify(offset)

                # Generate JAX slice notation
                if stride == 1 and offset == 0:
                    # Contiguous access
                    return "..."
                elif offset == 0:
                    # Pure stride: ::stride
                    stride_str = self.kexpr(stride)
                    return f"::{stride_str}"
                else:
                    # Offset + stride: offset::stride
                    offset_str = self.kexpr(offset)
                    stride_str = self.kexpr(stride)
                    return f"{offset_str}::{stride_str}"
            else:
                # Couldn't match affine pattern, fall back to original logic
                offset = index - var_expr
                offset = V.graph.sizevars.simplify(offset)
                if offset == 0 and var_expr == var:
                    # Just the variable itself, unit stride
                    return "..."
        elif len(used_vars) > 1:
            # Multi-dimensional indexing
            # For contiguous multi-dim access, all terms should have unit stride
            all_unit_stride = True
            for var in used_vars:
                var_expr = BlockPatternMatcher.get_subexpr_involving_symbol(index, var)
                stride = BlockPatternMatcher.match_affine_block_expr(var_expr, var)
                if stride != 1:
                    all_unit_stride = False
                    break

            if all_unit_stride:
                # Contiguous multi-dimensional access
                return "..."
            else:
                # Strided multi-dimensional access - requires advanced indexing
                # For now, use ellipsis which may work for many cases
                # TODO: Implement proper multi-dimensional strided indexing
                return "..."

        # For complex cases, raise an error
        return self._generate_index_array(index)

    def _generate_index_array(self, index: sympy.Expr) -> str:
        """
        Generate JAX code to compute an index array for complex indexing patterns.

        For very complex patterns that can't be expressed as simple slices,
        we need to compute the indices explicitly. This is not yet fully implemented.
        """
        # For now, raise an error for complex patterns
        # TODO: Implement advanced indexing support
        raise Unsupported(
            f"Pallas backend does not yet support complex indexing pattern: {index}"
        )

    def _has_iteration_vars(self, index: sympy.Expr) -> bool:
        """Check if index expression contains iteration variables (x0, x1, etc.)."""
        free_symbols = index.free_symbols
        iter_vars = OrderedSet(self.range_tree_nodes.keys())
        return bool(free_symbols & iter_vars)

    def _has_indirect_vars(self, index: sympy.Expr) -> bool:
        """Check if index expression contains indirect variables (tmp0, tmp1, etc.)."""
        free_symbols = index.free_symbols
        for sym in free_symbols:
            if str(sym).startswith("tmp"):
                return True
        return False

    def _get_index_expr(self, index: sympy.Expr) -> tuple[str, bool]:
        """
        Get the index expression string and whether it needs flattening.

        Returns:
            Tuple of (index_str, needs_flatten) where needs_flatten indicates
            if the buffer should be flattened before indexing (for mixed indexing).
        """
        has_indirect = self._has_indirect_vars(index)
        has_iter_vars = self._has_iteration_vars(index)

        if has_indirect and has_iter_vars:
            return self._handle_mixed_indexing(index), True
        elif has_indirect:
            return self.kexpr(index), False
        else:
            return self._get_index_str(index), False

    def load(self, name: str, index: sympy.Expr) -> CSEVariable:  # type: ignore[override]
        buf = self.args.input(name)
        dtype = V.graph.get_dtype(name)

        index_str, needs_flatten = self._get_index_expr(index)
        if needs_flatten:
            load_expr = f"{buf}[...].flatten()[{index_str}]"
        else:
            load_expr = f"{buf}[{index_str}]"

        return self.cse.generate(
            self.compute,
            load_expr,
            dtype=dtype,
        )

    def _handle_mixed_indexing(self, index: sympy.Expr) -> str:
        """
        Handle indexing with both indirect variables and iteration variables.

        For example, x[indices, :] generates index = i0 + stride * tmp0
        where tmp0 is loaded from indices and i0 is the iteration variable.

        We need to convert this to JAX advanced indexing with proper broadcasting.
        """
        # Get iteration variables
        iter_vars = OrderedSet(self.range_tree_nodes.keys())
        free_symbols = index.free_symbols
        used_iter_vars = sorted(free_symbols & iter_vars, key=str)

        if len(used_iter_vars) == 0:
            return self.kexpr(index)

        index_str = self.kexpr(index)
        indirect_vars = [str(sym) for sym in free_symbols if str(sym).startswith("tmp")]

        for i, var in enumerate(used_iter_vars):
            var_name = str(var)
            if var in self.range_tree_nodes:
                range_entry = self.range_tree_nodes[var]
                range_size = range_entry.length

                arange_expr = f"jnp.arange({self.kexpr(range_size)})"
                if indirect_vars:
                    arange_expr = f"{arange_expr}[None, :]"

                index_str = index_str.replace(var_name, arange_expr)

        # Reshape indirect variables for proper broadcasting
        for indirect_var in indirect_vars:
            index_str = index_str.replace(indirect_var, f"{indirect_var}[:, None]")

        return index_str

    def store(
        self, name: str, index: sympy.Expr, value: CSEVariable, mode: Any = None
    ) -> None:  # type: ignore[override]
        if mode is not None:
            raise Unsupported("pallas store mode not supported")
        out = self.args.output(name)
        self.store_buffer_names.add(name)

        index_str, _ = self._get_index_expr(index)
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

        # Get output dtype at compile time
        output_name = live_outs[0]
        output_dtype = V.graph.get_dtype(output_name)
        output_dtype_jax = torch_dtype_to_jax(output_dtype)

        code = IndentedBuffer()
        code.splice(
            """
            import functools
            import torch
            import jax
            import jax.numpy as jnp
            from jax.experimental import pallas as pl
            from torch._inductor.runtime.runtime_utils import next_power_of_2
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
            # Emit compute (CSE) and store lines; they reference *_ptr[index] directly
            # The iteration variables are implicitly handled by JAX's vectorization
            # When using [...], it processes the whole array
            # When using explicit indices, they should be JAX-traced values
            for line in self.compute._lines:
                code.writeline(str(line))
            for line in self.stores._lines:
                code.writeline(str(line))

        jit_wrapper_name = f"{kernel_name}_jit_wrapper"
        code.writeline("@functools.partial(jax.jit, static_argnums=(0, 1, 2))")
        code.writeline(
            f"def {jit_wrapper_name}(out_shape, out_dtype, device_type, *kernel_refs):"
        )
        with code.indent():
            code.writeline(
                "# On GPU (Triton backend), Pallas requires each dimension to be power of 2"
            )
            code.writeline("if device_type == 'cuda':")
            with code.indent():
                code.writeline("# Pad each dimension of each input to power of 2")
                code.writeline("kernel_refs_padded = []")
                code.writeline("for ref in kernel_refs:")
                with code.indent():
                    code.writeline(
                        "padded_shape = tuple(next_power_of_2(d) for d in ref.shape)"
                    )
                    code.writeline(
                        "pad_widths = [(0, padded_shape[i] - ref.shape[i]) for i in range(len(ref.shape))]"
                    )
                    code.writeline(
                        "padded_ref = jnp.pad(ref, pad_widths, mode='constant', constant_values=0)"
                    )
                    code.writeline("kernel_refs_padded.append(padded_ref)")
                code.writeline("kernel_refs_padded = tuple(kernel_refs_padded)")
                code.writeline("")
                code.writeline("# Pad output shape to power of 2")
                code.writeline(
                    "padded_out_shape = tuple(next_power_of_2(d) for d in out_shape)"
                )
            code.writeline("else:")
            with code.indent():
                code.writeline("# On CPU, no padding needed")
                code.writeline("kernel_refs_padded = kernel_refs")
                code.writeline("padded_out_shape = out_shape")
            code.writeline("")
            code.writeline("# Output spec with padded shape")
            code.writeline(
                "out_spec = jax.ShapeDtypeStruct(padded_out_shape, out_dtype)"
            )
            code.writeline("")
            code.writeline("result_padded = pl.pallas_call(")
            code.writeline(f"    {kernel_name}_kernel,")
            code.writeline("    out_shape=out_spec,")
            code.writeline(f"    interpret={interpret_literal},")
            code.writeline("    grid=(1,),")
            code.writeline(")(*kernel_refs_padded)")
            code.writeline("")
            code.writeline("# Slice back to original shape")
            code.writeline("if device_type == 'cuda':")
            with code.indent():
                code.writeline("# Build slice for each dimension")
                code.writeline(
                    "slices = tuple(slice(0, out_shape[i]) for i in range(len(out_shape)))"
                )
                code.writeline("return result_padded[slices]")
            code.writeline("else:")
            with code.indent():
                code.writeline("return result_padded")

        # Host entry: convert torch tensors <-> jax, call pallas_call and copy back
        main_name = f"{kernel_name}_main"
        code.writeline(f"def {main_name}({', '.join(kernel_params)}, stream=None):")
        with code.indent():
            # Enable JAX x64 mode to support float64/int64 types
            code.writeline("# Enable JAX x64 mode for float64/int64 support")
            code.writeline("jax.config.update('jax_enable_x64', True)")
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

            # Convert inputs to JAX arrays (keep original shapes for broadcasting)
            for inp in input_params:
                code.writeline(
                    f"{inp}_jax = jax.dlpack.from_dlpack({inp}.contiguous())"
                )

            # Get output metadata from PyTorch tensor
            code.writeline("# Prepare output metadata from PyTorch tensor")
            code.writeline(f"out_shape = tuple({output_param}.shape)")
            code.writeline(f"out_dtype = {output_dtype_jax}")
            code.writeline(f"device_type = {output_param}.device.type")

            call_args = ["out_shape", "out_dtype", "device_type"] + [
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
