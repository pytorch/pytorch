# mypy: allow-untyped-defs
"""
NVIDIA Universal GEMM kernel code generation.

This module generates Python code that calls cutlass_api to execute GEMM operations.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, TYPE_CHECKING, Union

from torch._inductor.codegen.common import (
    IndentedBuffer,
    Kernel,
    WorkspaceArg,
    WorkspaceZeroMode,
)
from torch._inductor.codegen.cutedsl.cutedsl_op_overrides import CuteDSLOpOverrides
from torch._inductor.ir import (
    BaseView,
    Buffer,
    ExternKernel,
    MutableBox,
    ReinterpretView,
)
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet


if TYPE_CHECKING:
    from torch._inductor.codegen.nv_universal_gemm.nv_universal_gemm import GemmVariant


log = logging.getLogger(__name__)


class NVUniversalGemmKernelWrapper:
    """Wrapper to provide .run() interface for NVIDIA Universal GEMM kernels."""

    def __init__(self, kernel_fn, kernel_path: Optional[str] = None):
        self.kernel_fn = kernel_fn
        self.kernel_path = kernel_path

    def run(self, *args, stream=None, **kwargs):
        """Execute the NVIDIA Universal GEMM kernel."""
        return self.kernel_fn(*args, stream=stream, **kwargs)


class NVUniversalGemmKernel(Kernel):
    """
    Kernel implementation for NVIDIA Universal GEMM.

    Generates Python code that calls cutlass_api to execute GEMM operations.
    Unlike CuteDSL which uses Jinja templates, this generates simpler direct
    Python code.
    """

    def __init__(
        self,
        kernel_name: str,
        input_nodes: list[Buffer],
        output_node: Buffer,
        kernel_metadata: dict[str, Any],
        accumulator_type: Any,
        variant: GemmVariant,
        workspace_size: int = 0,
    ) -> None:
        super().__init__()
        self.kernel_name = kernel_name
        self.input_nodes = input_nodes
        self.output_node = output_node
        self.kernel_metadata = kernel_metadata
        self.accumulator_type = accumulator_type
        self.workspace_size = workspace_size
        self.variant = variant

        self._template_input_args: list[tuple[str, Buffer]] = []
        self._seen_input_args: OrderedSet[str] = OrderedSet()

        for i, input_node in enumerate(input_nodes):
            param_name = f"in_ptr{i}"
            self._template_input_args.append((param_name, input_node))
            self._seen_input_args.add(param_name)

    def render(self) -> str:
        """
        Render the NVIDIA Universal GEMM kernel code as a Python source string.

        Generates Python code that:
        1. Looks up the cutlass_api kernel by name from the manifest (cached in
           _nv_universal_gemm_kernel_cache to avoid repeated manifest searches)
        2. Creates GemmArguments with the input/output tensors and accumulator type
        3. Compiles the kernel for the specific tensor shapes/dtypes (cached in
           _nv_universal_gemm_artifact_cache keyed by (shape, dtype) tuple)
        4. Runs the kernel with the compiled artifact and CUDA stream

        The caching strategy ensures:
        - Kernel lookup happens once per unique kernel name
        - Compilation happens once per unique (shape, dtype) combination
        - Runtime execution is just the kernel.run() call with cached artifact

        Returns:
            Python source code string to be written to a .py file and loaded
            via async_compile.nv_universal_gemm()
        """
        from torch._inductor.codegen.nv_universal_gemm.nv_universal_gemm import (
            GemmVariant,
        )

        kernel_name_str = self.kernel_metadata["kernel_name"]
        is_grouped = self.variant == GemmVariant.GROUPED_GEMM

        acc_dtype_str = CuteDSLOpOverrides.TORCH_TO_CUTE_DTYPE.get(
            self.accumulator_type, "cutlass.Float32"
        )

        input_params = [f"in_ptr{i}" for i, _ in enumerate(self.input_nodes)]
        input_params.append("out_ptr0")
        if self.workspace_size > 0:
            input_params.append("workspace")
        input_params.append("stream=None")
        params_str = ", ".join(input_params)

        workspace_arg = "workspace" if self.workspace_size > 0 else "None"

        var_prefix = self.variant.op_name.upper()
        cache_var = f"_{var_prefix}_compiled_cache"
        kernel_name_var = f"_{var_prefix}_KERNEL_NAME"

        # Variant-specific code generation with canonical hook points:
        # - preprocess_inputs: input transformations before cache check
        # - cache_key_code: expression for cache key
        # - create_args_code: code to create Arguments object
        # - populate_args: code to update all runtime tensor references
        # - clear_args: code to clear all runtime tensor references
        if is_grouped:
            preprocess_inputs = """# Transpose B from K-major to N-major for CUTLASS compatibility
                in_ptr1_transposed = in_ptr1.permute(0, 2, 1).contiguous().permute(0, 2, 1)"""
            cache_key_code = "(in_ptr0.shape, in_ptr0.dtype, in_ptr1.shape, in_ptr1.dtype, in_ptr2.shape)"
            create_args_code = f"""args = cutlass_api.arguments.GroupedGemmArguments(
                        in_ptr0,
                        in_ptr1_transposed,
                        out_ptr0,
                        accumulator_type={acc_dtype_str},
                        offsets=in_ptr2,
                    )"""
            populate_args = """args.A.tensor.runtime_tensor = in_ptr0
                args.B.tensor.runtime_tensor = in_ptr1_transposed
                args.out.tensor.runtime_tensor = out_ptr0
                args.offsets.tensor.runtime_tensor = in_ptr2"""
            clear_args = """args.A.tensor.runtime_tensor = None
                args.B.tensor.runtime_tensor = None
                args.out.tensor.runtime_tensor = None
                args.offsets.tensor.runtime_tensor = None"""
        else:
            preprocess_inputs = ""
            cache_key_code = (
                "(in_ptr0.shape, in_ptr0.dtype, in_ptr1.shape, in_ptr1.dtype)"
            )
            create_args_code = f"""args = cutlass_api.arguments.GemmArguments(
                        in_ptr0,
                        in_ptr1,
                        out_ptr0,
                        accumulator_type={acc_dtype_str},
                    )"""
            populate_args = """args.A.tensor.runtime_tensor = in_ptr0
                args.B.tensor.runtime_tensor = in_ptr1
                args.out.tensor.runtime_tensor = out_ptr0"""
            clear_args = """args.A.tensor.runtime_tensor = None
                args.B.tensor.runtime_tensor = None
                args.out.tensor.runtime_tensor = None"""

        code = IndentedBuffer()
        code.splice(
            f"""
            import cutlass
            import cutlass_api
            from torch._inductor.codegen.nv_universal_gemm.kernel_cache import get_kernel_by_name

            {kernel_name_var} = "{kernel_name_str}"

            # Caching strategy for NVGEMM kernels:
            # - Global kernel cache (in kernel_cache.py): kernel_name -> kernel object
            #   Built lazily on first access, shared across all NVGEMM kernels
            # - compiled_cache: stores (Arguments, artifact) tuple per (shape, dtype)
            #   - Arguments: tensor wrapper object
            #   - artifact: compiled GPU binary
            # On subsequent calls, we reuse cached args and just update tensor pointers (A, B, out).
            # After kernel.run(), we clear tensor references to avoid holding them in the cache,
            # which would interfere with CUDA graph trees memory tracking.
            {cache_var} = {{}}

            def {self.kernel_name}_main({params_str}):
                global {cache_var}

                kernel = get_kernel_by_name({kernel_name_var})
                if kernel is None:
                    raise RuntimeError(f"Could not find kernel: {{{kernel_name_var}}}")

                {preprocess_inputs}

                cache_key = {cache_key_code}

                if cache_key not in {cache_var}:
                    {create_args_code}
                    artifact = kernel.compile(args)
                    {cache_var}[cache_key] = (args, artifact)
                else:
                    args, artifact = {cache_var}[cache_key]

                {populate_args}

                kernel.run(args, artifact, stream=stream, workspace={workspace_arg}, assume_supported_args=True)

                {clear_args}
            """
        )

        return code.getvalue()

    def _get_reinterpret_view(self, node) -> Optional[ReinterpretView]:
        """Extract or convert to ReinterpretView from a node, handling all views."""
        while isinstance(node, MutableBox):
            node = node.data
        if isinstance(node, BaseView):
            return ExternKernel.convert_to_reinterpret_view(node)
        return None

    def call_kernel(self, name: str, node=None):
        """
        Generate the kernel call in the wrapper code.

        Similar to CuteDSLTemplateKernel.call_kernel but simplified for NVIDIA Universal GEMM.
        """
        wrapper = V.graph.wrapper_code

        call_args: list[str] = []
        arg_types: list[Any] = []
        raw_args: list[Union[Buffer, ReinterpretView, None]] = []

        for _, input_node in self._template_input_args:
            reinterpret_view = self._get_reinterpret_view(input_node)
            if reinterpret_view is not None:
                call_args.append(reinterpret_view.codegen_reference())
                # Pass the ReinterpretView as raw_arg so autotune_at_compile_time
                # can use it to generate example tensors
                raw_args.append(reinterpret_view)
            else:
                call_args.append(input_node.get_name())
                raw_args.append(input_node)
            arg_types.append(V.graph.get_dtype(input_node.get_name()))

        output_name = self.output_node.get_name()
        call_args.append(output_name)
        arg_types.append(V.graph.get_dtype(output_name))
        raw_args.append(None)  # Output buffer is findable by name

        # Allocate workspace if needed
        ws: Optional[WorkspaceArg] = None
        if self.workspace_size > 0:
            ws = WorkspaceArg(
                count=self.workspace_size,
                device=V.graph.get_current_device_or_throw(),
                zero_mode=WorkspaceZeroMode.UNINITIALIZED,
                outer_name=WorkspaceArg.unique_name(),
            )
            wrapper.generate_workspace_allocation(ws)
            call_args.append(ws.outer_name)
            arg_types.append(ws.dtype)
            raw_args.append(None)

        # Generate the kernel call using triton=True for Python-based kernels
        # Pass raw_keys as None list to match raw_args length
        # TODO(nikhilap)  We don't use autotune_args like the Triton path
        wrapper.generate_kernel_call(
            name,
            call_args,
            triton=True,
            arg_types=arg_types,
            raw_args=raw_args,
            raw_keys=[None] * len(raw_args),
        )

        # Deallocate workspace after kernel call
        if ws is not None:
            wrapper.generate_workspace_deallocation(ws)
