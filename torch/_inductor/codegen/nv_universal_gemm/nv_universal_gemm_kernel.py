# mypy: allow-untyped-defs
"""
NVIDIA Universal GEMM kernel code generation.

This module generates Python code that calls cutlass_api to execute GEMM operations.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, TYPE_CHECKING

from torch._inductor.codegen.common import (
    IndentedBuffer,
    Kernel,
    WorkspaceArg,
    WorkspaceZeroMode,
)
from torch._inductor.codegen.cutedsl.cutedsl_op_overrides import CuteDSLOpOverrides
from torch._inductor.codegen.cutlass.python_evt import _ACCUMULATOR_ARG_NAME
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

    def __init__(self, kernel_fn, kernel_path: str | None = None):
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
        scale_type_a: Any | None = None,
        scale_type_b: Any | None = None,
        swizzle_type_a: Any | None = None,
        swizzle_type_b: Any | None = None,
        epilogue_fn_code: str | None = None,
        epilogue_reads: list[str] | None = None,
        epilogue_writes: list[str] | None = None,
        epilogue_var_renames: dict[str, Any] | None = None,
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
        """Render the Python source for the NVGEMM kernel wrapper."""
        from torch._inductor.codegen.nv_universal_gemm.nv_universal_gemm import (
            GemmVariant,
        )

        kernel_name_str = self.kernel_metadata["kernel_name"]
        is_grouped = self.variant == GemmVariant.GROUPED_GEMM
        is_scaled = self.variant == GemmVariant.SCALED_GEMM
        has_epilogue = bool(self.epilogue_fn_code)

        if has_epilogue and (is_grouped or is_scaled):
            raise NotImplementedError(
                "Epilogue fusion is not yet supported for grouped or scaled GEMM variants"
            )

        acc_dtype_str = CuteDSLOpOverrides.TORCH_TO_CUTE_DTYPE.get(
            self.accumulator_type, "cutlass.Float32"
        )

        input_params = [f"in_ptr{i}" for i, _ in enumerate(self.input_nodes)]
        input_params.append("out_ptr0")
        input_params.extend(self.epilogue_reads)
        if self.workspace_size > 0:
            input_params.append("workspace")
        input_params.append("stream=None")
        params_str = ", ".join(input_params)

        workspace_arg = "workspace" if self.workspace_size > 0 else "None"
        var_prefix = self.variant.op_name.upper()
        cache_var = f"_{var_prefix}_compiled_cache"
        kernel_name_var = f"_{var_prefix}_KERNEL_NAME"

        extra_imports = ""
        if is_grouped:
            cache_key = "(in_ptr0.shape, in_ptr0.dtype, in_ptr1.shape, in_ptr1.dtype, in_ptr2.shape)"
            create_args = f"""    args = cutlass_api.arguments.GroupedGemmArguments(
        in_ptr0, in_ptr1, out_ptr0,
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
            extra_imports = (
                "from cutlass_api.arguments import ScaledTensor\n"
                "from cutlass_api.library import ScaleMode, ScaleSwizzleMode"
            )
            cache_key = "(in_ptr0.shape, in_ptr0.dtype, in_ptr1.shape, in_ptr1.dtype, in_ptr2.shape, in_ptr3.shape)"
            sma = scale_mode_a.name if scale_mode_a else ""
            smb = scale_mode_b.name if scale_mode_b else ""
            swa = swizzle_mode_a.name if swizzle_mode_a else ""
            swb = swizzle_mode_b.name if swizzle_mode_b else ""
            create_args = f"""    scaled_a = ScaledTensor(in_ptr0, in_ptr2, ScaleMode.{sma}, ScaleSwizzleMode.{swa})
    scaled_b = ScaledTensor(in_ptr1, in_ptr3, ScaleMode.{smb}, ScaleSwizzleMode.{swb})
    args = cutlass_api.arguments.GemmArguments(
        scaled_a, scaled_b, out_ptr0,
        accumulator_type={acc_dtype_str},
    )"""
        else:
            cache_key = "(in_ptr0.shape, in_ptr0.dtype, in_ptr1.shape, in_ptr1.dtype)"
            create_args = f"""    args = cutlass_api.arguments.GemmArguments(
        in_ptr0, in_ptr1, out_ptr0,
        accumulator_type={acc_dtype_str},
    )"""

        if has_epilogue:
            assert self.epilogue_fn_code is not None  # narrowed by has_epilogue
            epilogue_kwargs = self._render_epilogue_kwargs()
            source_hash = hashlib.sha256(self.epilogue_fn_code.encode()).hexdigest()

            kernel_cache_import = "from torch._inductor.codegen.nv_universal_gemm.kernel_cache import get_efc_kernel_with_epilogue"
            epilogue_import = "from cutlass_api.arguments import EpilogueArguments"
            epilogue_fn_def = (
                self.epilogue_fn_code + f'\n_EPILOGUE_FN_SOURCE = "{source_hash}"'
            )

            epilogue_setup = f"""
    epi_args = EpilogueArguments(epilogue_fn=_epilogue_fn, {epilogue_kwargs})
"""
            kernel_lookup = f"""    kernel = get_efc_kernel_with_epilogue({kernel_name_str!r}, epi_args, epilogue_source=_EPILOGUE_FN_SOURCE)
    if kernel is None:
        raise RuntimeError(f"Could not find EFC kernel: {{{kernel_name_var}}}")"""

            create_args = f"""    args = cutlass_api.arguments.GemmArguments(
        in_ptr0, in_ptr1, out_ptr0,
        accumulator_type={acc_dtype_str},
        epilogue=epi_args,
    )"""
            # Aux tensors from the epilogue change the kernel's compiled
            # artifact (cutlass_api dispatches on their dtype/shape) but the
            # base cache_key only fingerprints A/B. Without folding aux
            # tensor metadata in, a wrapper invoked with the same A/B but a
            # differently-shaped aux input (e.g. dynamic-shape bias) would
            # silently reuse a stale artifact.
            aux_sig = ", ".join(
                f"{name}.shape, {name}.dtype" for name in self.epilogue_reads
            )
            if aux_sig:
                cache_key = f'({cache_key}, "epilogue", {aux_sig})'
            else:
                cache_key = f'({cache_key}, "epilogue")'
        else:
            kernel_cache_import = "from torch._inductor.codegen.nv_universal_gemm.kernel_cache import get_kernel_by_name"
            epilogue_import = ""
            epilogue_fn_def = ""
            epilogue_setup = ""
            kernel_lookup = f"""    kernel = get_kernel_by_name({kernel_name_var})
    if kernel is None:
        raise RuntimeError(f"Could not find kernel: {{{kernel_name_var}}}")"""

        code = IndentedBuffer()
        code.splice(
            f"""
import cutlass
import cutlass_api
{kernel_cache_import}
{epilogue_import}
{extra_imports}

{epilogue_fn_def}

{kernel_name_var} = {kernel_name_str!r}
{cache_var} = {{}}

def {self.kernel_name}_main({params_str}):
    global {cache_var}
{epilogue_setup}
{create_args}

{kernel_lookup}

    cache_key = {cache_key}
    artifact = {cache_var}.get(cache_key)
    if artifact is None:
        artifact = kernel.compile(args)
        {cache_var}[cache_key] = artifact

    kernel.run(args, artifact, stream=stream, workspace={workspace_arg}, assume_supported_args=True)
            """
        )

        return code.getvalue()

    def _render_epilogue_kwargs(self) -> str:
        """Render kwargs for EpilogueArguments constructor.

        Skips intermediate stores (write_buffer entries from CutlassEVTCodegen.store()
        in a multi-node epilogue chain) — those names are not kernel parameters and
        would produce NameError at runtime.
        """
        kwargs_parts = []
        write_buffer_names = OrderedSet(self.epilogue_writes)

        for var_name, buffer_name in self.epilogue_var_renames.items():
            if var_name == "D":
                kwargs_parts.append("D=out_ptr0")
            elif var_name == _ACCUMULATOR_ARG_NAME:
                continue
            elif buffer_name in write_buffer_names:
                continue
            else:
                kwargs_parts.append(f"{var_name}={buffer_name}")

        return ", ".join(kwargs_parts)

    def _get_reinterpret_view(self, node) -> ReinterpretView | None:
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
        raw_args: list[Buffer | ReinterpretView | None] = []
        raw_keys: list[str | None] = []

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

        # The kernel writes to the epilogue's final output, not the GEMM buffer
        # (which is removed via removed_buffers aliasing).
        if self.epilogue_writes:
            output_name = self.epilogue_writes[-1]
        else:
            output_name = self.output_node.get_name()
        call_args.append(output_name)
        arg_types.append(V.graph.get_dtype(output_name))
        raw_args.append(None)  # Output buffer is findable by name
        raw_keys.append("out_ptr0")

        for read_name in self.epilogue_reads:
            call_args.append(read_name)
            arg_types.append(V.graph.get_dtype(read_name))
            buf = V.graph.get_buffer(read_name)
            if buf is None:
                buf = V.graph.graph_inputs.get(read_name)
            # pyrefly: ignore [bad-argument-type]
            raw_args.append(buf)
            raw_keys.append(read_name)

        # Allocate workspace if needed
        ws: WorkspaceArg | None = None
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
