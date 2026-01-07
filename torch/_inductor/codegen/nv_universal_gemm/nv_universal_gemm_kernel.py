# mypy: allow-untyped-defs
"""
NVIDIA Universal GEMM kernel code generation.

This module generates Python code that calls cutlass_api to execute GEMM operations.
"""

import logging
from typing import Any, Optional, Union

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
        workspace_size: int = 0,
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
        kernel_name_str = self.kernel_metadata["kernel_name"]

        acc_dtype_str = CuteDSLOpOverrides.TORCH_TO_CUTE_DTYPE.get(
            self.accumulator_type, "cutlass.Float32"
        )

        # Build function parameters
        input_params = [f"in_ptr{i}" for i, _ in enumerate(self.input_nodes)]
        input_params.extend(["out_ptr0"])

        # Add epilogue read parameters
        for read_name in self.epilogue_reads:
            input_params.append(read_name)

        # Add workspace parameter if needed
        if self.workspace_size > 0:
            input_params.append("workspace")
        input_params.append("stream=None")
        params_str = ", ".join(input_params)

        code = IndentedBuffer()

        # Build workspace argument for kernel.run() call
        workspace_arg = "workspace" if self.workspace_size > 0 else "None"

        # Build epilogue code if present
        if self.epilogue_fn_code:
            epilogue_kwargs = self._render_epilogue_kwargs()
            epilogue_import = "from cutlass_api.arguments import EpilogueArguments"
            epilogue_fn_def = f'_epilogue_fn = """{self.epilogue_fn_code}"""'
            epilogue_args_construction = f"""
    epi_args = EpilogueArguments(
        _epilogue_fn,
        {epilogue_kwargs}
    )"""
            gemm_epilogue_arg = "epilogue=epi_args,"
        else:
            epilogue_import = ""
            epilogue_fn_def = ""
            epilogue_args_construction = ""
            gemm_epilogue_arg = ""

        # Generate different kernel selection code based on whether there's an epilogue
        # When there's an epilogue, we must use get_kernels(args) to find compatible kernels
        # because the original kernel (selected without epilogue) may not support the epilogue
        if self.epilogue_fn_code:
            kernel_selection_code = f"""
    args = cutlass_api.arguments.GemmArguments(
        in_ptr0,
        in_ptr1,
        out_ptr0,
        accumulator_type={acc_dtype_str},
        {gemm_epilogue_arg}
    )

    cache_key = (in_ptr0.shape, in_ptr0.dtype, in_ptr1.shape, in_ptr1.dtype, "epilogue")
    if cache_key not in _nv_universal_gemm_kernel_cache:
        kernels = cutlass_api.get_kernels(args, cc=100)
        if not kernels:
            raise RuntimeError("Could not find NVIDIA Universal GEMM kernel with epilogue support")
        _nv_universal_gemm_kernel_cache[cache_key] = kernels[0]
    kernel = _nv_universal_gemm_kernel_cache[cache_key]

    if cache_key not in _nv_universal_gemm_artifact_cache:
        _nv_universal_gemm_artifact_cache[cache_key] = kernel.compile(args)

    artifact = _nv_universal_gemm_artifact_cache[cache_key]
    kernel.run(args, artifact, stream=stream, workspace={workspace_arg}, assume_supported_args=True)"""
        else:
            kernel_selection_code = f"""
    if _NV_UNIVERSAL_GEMM_KERNEL_NAME not in _nv_universal_gemm_kernel_cache:
        kernels = cutlass_api.get_kernels(
            metadata_filter=lambda m: m.kernel_name == _NV_UNIVERSAL_GEMM_KERNEL_NAME
        )
        if not kernels:
            raise RuntimeError(f"Could not find NVIDIA Universal GEMM kernel: {{_NV_UNIVERSAL_GEMM_KERNEL_NAME}}")
        _nv_universal_gemm_kernel_cache[_NV_UNIVERSAL_GEMM_KERNEL_NAME] = kernels[0]

    kernel = _nv_universal_gemm_kernel_cache[_NV_UNIVERSAL_GEMM_KERNEL_NAME]

    args = cutlass_api.arguments.GemmArguments(
        in_ptr0,
        in_ptr1,
        out_ptr0,
        accumulator_type={acc_dtype_str},
    )

    cache_key = (in_ptr0.shape, in_ptr0.dtype, in_ptr1.shape, in_ptr1.dtype)
    if cache_key not in _nv_universal_gemm_artifact_cache:
        _nv_universal_gemm_artifact_cache[cache_key] = kernel.compile(args)

    artifact = _nv_universal_gemm_artifact_cache[cache_key]
    kernel.run(args, artifact, stream=stream, workspace={workspace_arg}, assume_supported_args=True)"""

        code.splice(
            f"""
import cutlass
import cutlass_api
{epilogue_import}

{epilogue_fn_def}

_NV_UNIVERSAL_GEMM_KERNEL_NAME = "{kernel_name_str}"
_nv_universal_gemm_kernel_cache = {{}}
_nv_universal_gemm_artifact_cache = {{}}

def {self.kernel_name}_main({params_str}):
    global _nv_universal_gemm_kernel_cache, _nv_universal_gemm_artifact_cache
    {epilogue_args_construction}
{kernel_selection_code}
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
