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
from torch._inductor.codegen.nv_universal_gemm.nv_universal_gemm_utils import (
    to_cutlass_scale_mode,
)
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
        scale_type_a: Optional[Any] = None,
        scale_type_b: Optional[Any] = None,
        swizzle_type_a: Optional[Any] = None,
        swizzle_type_b: Optional[Any] = None,
        epilogue_fn_code: Optional[str] = None,
        epilogue_reads: Optional[list[str]] = None,
        epilogue_writes: Optional[list[str]] = None,
        epilogue_var_renames: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.kernel_name = kernel_name
        self.input_nodes = input_nodes
        self.output_node = output_node
        self.kernel_metadata = kernel_metadata
        self.accumulator_type = accumulator_type
        self.workspace_size = workspace_size
        self.variant = variant
        self.scale_type_a = scale_type_a
        self.scale_type_b = scale_type_b
        self.swizzle_type_a = swizzle_type_a
        self.swizzle_type_b = swizzle_type_b
        self.epilogue_fn_code = epilogue_fn_code
        self.epilogue_reads = epilogue_reads or []
        self.epilogue_writes = epilogue_writes or []
        self.epilogue_var_renames = epilogue_var_renames or {}

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
        3. Optionally creates EpilogueArguments if epilogue fusion is enabled
        4. Compiles the kernel for the specific tensor shapes/dtypes (cached in
           _nv_universal_gemm_artifact_cache keyed by (shape, dtype) tuple)
        5. Runs the kernel with the compiled artifact and CUDA stream

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
        is_scaled = self.variant == GemmVariant.SCALED_GEMM

        acc_dtype_str = CuteDSLOpOverrides.TORCH_TO_CUTE_DTYPE.get(
            self.accumulator_type, "cutlass.Float32"
        )

        # Build function parameters
        input_params = [f"in_ptr{i}" for i, _ in enumerate(self.input_nodes)]
        input_params.append("out_ptr0")

        # Add epilogue read parameters
        for read_name in self.epilogue_reads:
            input_params.append(read_name)

        # Add workspace parameter if needed
        if self.workspace_size > 0:
            input_params.append("workspace")
        input_params.append("stream=None")
        params_str = ", ".join(input_params)

        workspace_arg = "workspace" if self.workspace_size > 0 else "None"

        var_prefix = self.variant.op_name.upper()
        cache_var = f"_{var_prefix}_compiled_cache"
        kernel_name_var = f"_{var_prefix}_KERNEL_NAME"

        extra_imports = ""
        if is_scaled:
            extra_imports = """from cutlass_api.arguments import ScaledTensor
            from cutlass_api.library import ScaleMode, ScaleSwizzleMode"""

        # Variant-specific code generation:
        # - preprocess_inputs: input transformations before args creation
        # - cache_key_code: expression for cache key
        # - create_args_code: code to create Arguments object
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
        elif is_scaled:
            scale_mode_a, swizzle_mode_a = to_cutlass_scale_mode(
                self.scale_type_a, self.swizzle_type_a
            )
            scale_mode_b, swizzle_mode_b = to_cutlass_scale_mode(
                self.scale_type_b, self.swizzle_type_b
            )
            scale_mode_a_str = scale_mode_a.name if scale_mode_a else ""
            scale_mode_b_str = scale_mode_b.name if scale_mode_b else ""
            swizzle_mode_a_str = swizzle_mode_a.name if swizzle_mode_a else ""
            swizzle_mode_b_str = swizzle_mode_b.name if swizzle_mode_b else ""
            preprocess_inputs = ""
            cache_key_code = "(in_ptr0.shape, in_ptr0.dtype, in_ptr1.shape, in_ptr1.dtype, in_ptr2.shape, in_ptr3.shape)"
            create_args_code = f"""scaled_a = ScaledTensor(
                    in_ptr0, in_ptr2, ScaleMode.{scale_mode_a_str}, ScaleSwizzleMode.{swizzle_mode_a_str}
                )
                scaled_b = ScaledTensor(
                    in_ptr1, in_ptr3, ScaleMode.{scale_mode_b_str}, ScaleSwizzleMode.{swizzle_mode_b_str}
                )
                args = cutlass_api.arguments.GemmArguments(
                    scaled_a,
                    scaled_b,
                    out_ptr0,
                    accumulator_type={acc_dtype_str},
                )"""
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

        # Build epilogue code if present
        if self.epilogue_fn_code:
            epilogue_kwargs = self._render_epilogue_kwargs()
            epilogue_import = "from cutlass_api.arguments import EpilogueArguments"
            # Embed epilogue function directly as code (not as a string)
            epilogue_fn_def = self.epilogue_fn_code
            epilogue_args_construction = f"""
    epi_args = EpilogueArguments(
        epilogue_fn=_epilogue_fn,
        {epilogue_kwargs}
    )"""
            gemm_epilogue_arg = "epilogue=epi_args,"
        else:
            epilogue_import = ""
            epilogue_fn_def = ""
            epilogue_args_construction = ""
            gemm_epilogue_arg = ""

        # Generate args creation code based on epilogue presence
        if self.epilogue_fn_code:
            # With epilogue: add epilogue args to GemmArguments
            create_args_with_epilogue = f"""args = cutlass_api.arguments.GemmArguments(
        in_ptr0,
        in_ptr1,
        out_ptr0,
        accumulator_type={acc_dtype_str},
        {gemm_epilogue_arg}
    )"""
            # Include "epilogue" in cache key to differentiate from non-epilogue compilations
            cache_key_code_with_epilogue = cache_key_code[:-1] + ', "epilogue")'
        else:
            create_args_with_epilogue = create_args_code
            cache_key_code_with_epilogue = cache_key_code

        # Kernel lookup
        # For epilogue cases, we need to use get_kernels(args) to get a properly configured
        # EFC kernel. The cached kernel from get_kernel_by_name() doesn't work with epilogue
        # because it lacks the epilogue configuration.
        if self.epilogue_fn_code:
            # EFC kernels are directly in choices, so the kernel is already an EFC kernel
            kernel_cache_import = "from torch._inductor.codegen.nv_universal_gemm.kernel_cache import get_efc_kernel_with_epilogue"
            # Use get_efc_kernel_with_epilogue for fast O(1) lookup + kernel construction
            # This avoids the slow get_kernels() call by constructing the kernel directly
            # with epilogue metadata from the cached base kernel
            kernel_lookup_code = f"""# Get EFC kernel with epilogue (fast path using cached metadata)
    kernel = get_efc_kernel_with_epilogue("{kernel_name_str}", epi_args)
    if kernel is None:
        raise RuntimeError(f"Could not find EFC kernel: {kernel_name_str}")"""
        else:
            kernel_cache_import = "from torch._inductor.codegen.nv_universal_gemm.kernel_cache import get_kernel_by_name"
            kernel_lookup_code = f"""kernel = get_kernel_by_name({kernel_name_var})
    if kernel is None:
        raise RuntimeError(f"Could not find kernel: {{{kernel_name_var}}}")"""

        # Generate main body
        if self.epilogue_fn_code:
            # Epilogue case: create epilogue args first, then GEMM args with epilogue
            # Kernel caching is handled by get_efc_kernel_with_epilogue
            efc_kernel_cache_init = ""
            global_decl = f"global {cache_var}"
            main_body = f"""
    import time as _time
    {epilogue_args_construction}

    {preprocess_inputs}

    {create_args_with_epilogue}

    {kernel_lookup_code}

    cache_key = {cache_key_code_with_epilogue}
    artifact = {cache_var}.get(cache_key)
    if artifact is None:
        _tc0 = _time.perf_counter()
        artifact = kernel.compile(args)
        _tc1 = _time.perf_counter()
        print(f"      [Kernel] compile: {{(_tc1-_tc0)*1000:.2f}} ms")
        {cache_var}[cache_key] = artifact
    else:
        print(f"      [Kernel] compile: cached")

    _tr0 = _time.perf_counter()
    kernel.run(args, artifact, stream=stream, workspace={workspace_arg}, assume_supported_args=True)
    _tr1 = _time.perf_counter()
    print(f"      [Kernel] run: {{(_tr1-_tr0)*1000:.2f}} ms")"""
        else:
            # Non-epilogue case: standard order
            efc_kernel_cache_init = ""
            global_decl = f"global {cache_var}"
            main_body = f"""
    {kernel_lookup_code}

    {preprocess_inputs}

    {create_args_with_epilogue}

    cache_key = {cache_key_code_with_epilogue}
    artifact = {cache_var}.get(cache_key)
    if artifact is None:
        artifact = kernel.compile(args)
        {cache_var}[cache_key] = artifact

    kernel.run(args, artifact, stream=stream, workspace={workspace_arg}, assume_supported_args=True)"""

        code = IndentedBuffer()
        code.splice(
            f"""
import cutlass
import cutlass_api
{kernel_cache_import}
{epilogue_import}
{extra_imports}

{epilogue_fn_def}

{kernel_name_var} = "{kernel_name_str}"
# Maps (shape, dtype, shape, dtype, ...) -> compiled kernel artifact
{cache_var} = {{}}
{efc_kernel_cache_init}

def {self.kernel_name}_main({params_str}):
    {global_decl}
{main_body}
            """
        )

        return code.getvalue()

    def _render_epilogue_kwargs(self) -> str:
        """
        Render kwargs for EpilogueArguments constructor.

        Maps Python variable names from the epilogue function to actual tensor arguments.
        The 'D' output is mapped to out_ptr0, 'accum' is implicit (provided by GEMM),
        and other reads are passed directly.
        """
        kwargs_parts = []

        # Map Python var names to actual tensor args
        for var_name, buffer_name in self.epilogue_var_renames.items():
            if var_name == "D":
                # D is the output, map to out_ptr0
                kwargs_parts.append("D=out_ptr0")
            elif var_name == "accum":
                # Skip accum, it's implicit (the GEMM result)
                continue
            elif buffer_name in self.epilogue_reads:
                # This is an epilogue input tensor - use buffer_name as the parameter
                kwargs_parts.append(f"{var_name}={buffer_name}")
            else:
                # Could be a scalar or intermediate value
                kwargs_parts.append(f"{var_name}={buffer_name}")

        return ", ".join(kwargs_parts)

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
        Includes epilogue input tensors when epilogue fusion is enabled.
        """
        wrapper = V.graph.wrapper_code

        call_args: list[str] = []
        arg_types: list[Any] = []
        raw_args: list[Union[Buffer, ReinterpretView, None]] = []
        raw_keys: list[Optional[str]] = []

        # Add GEMM input args (A, B)
        for param_name, input_node in self._template_input_args:
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
            raw_keys.append(param_name)

        # Add output arg
        # When epilogue is fused, use the epilogue output; otherwise use GEMM output
        if self.epilogue_writes:
            # The epilogue's D output is stored in epilogue_writes
            # Use the last epilogue write as the actual output
            output_name = self.epilogue_writes[-1]
        else:
            output_name = self.output_node.get_name()
        call_args.append(output_name)
        arg_types.append(V.graph.get_dtype(output_name))
        raw_args.append(None)  # Output buffer is findable by name
        raw_keys.append("out_ptr0")

        # Add epilogue input args
        for read_name in self.epilogue_reads:
            call_args.append(read_name)
            arg_types.append(V.graph.get_dtype(read_name))
            buf = V.graph.get_buffer(read_name)
            raw_args.append(buf)
            raw_keys.append(read_name)

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
            raw_keys.append(None)

        # Generate the kernel call using triton=True for Python-based kernels
        wrapper.generate_kernel_call(
            name,
            call_args,
            triton=True,
            arg_types=arg_types,
            raw_args=raw_args,
            raw_keys=raw_keys,
        )

        # Deallocate workspace after kernel call
        if ws is not None:
            wrapper.generate_workspace_deallocation(ws)
