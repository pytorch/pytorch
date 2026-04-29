# mypy: allow-untyped-defs
"""
NVIDIA Universal GEMM kernel code generation.

This module generates Python code that calls cutlass_api to execute GEMM operations.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
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


@dataclass(frozen=True)
class _VariantRenderSpec:
    import_lines: tuple[str, ...] = ()
    helper_kwargs: tuple[str, ...] = ()


@dataclass(frozen=True)
class _EpilogueRenderSpec:
    import_lines: tuple[str, ...] = ()
    module_block: str | None = None
    module_lines: tuple[str, ...] = ()
    setup_arg_lines: tuple[str, ...] = ()
    gemm_arg_kwargs: tuple[str, ...] = ()
    kernel_lookup_kwargs: tuple[str, ...] = ()
    enabled: bool = False


def _get_scaled_gemm_modes(
    scale_type_a: Any | None,
    swizzle_type_a: Any | None,
    scale_type_b: Any | None,
    swizzle_type_b: Any | None,
) -> tuple[Any, Any, Any, Any]:
    scale_mode_a, swizzle_mode_a = to_cutlass_scale_mode(scale_type_a, swizzle_type_a)
    scale_mode_b, swizzle_mode_b = to_cutlass_scale_mode(scale_type_b, swizzle_type_b)
    if any(
        mode is None
        for mode in (scale_mode_a, swizzle_mode_a, scale_mode_b, swizzle_mode_b)
    ):
        raise NotImplementedError("Unsupported scale/swizzle mode for scaled GEMM")
    return scale_mode_a, swizzle_mode_a, scale_mode_b, swizzle_mode_b


def _create_gemm_arguments(
    variant_name: str,
    input_tensors,
    out,
    accumulator_type: Any,
    *,
    scale_mode_a: Any | None = None,
    swizzle_mode_a: Any | None = None,
    scale_mode_b: Any | None = None,
    swizzle_mode_b: Any | None = None,
    epilogue: Any | None = None,
):
    import cutlass_api

    if epilogue is not None and variant_name != "GEMM":
        raise NotImplementedError(
            "Epilogue fusion is not yet supported for grouped or scaled GEMM variants"
        )

    if variant_name == "GROUPED_GEMM":
        a, b, offsets = input_tensors
        return cutlass_api.arguments.GroupedGemmArguments(
            a,
            b,
            out,
            accumulator_type=accumulator_type,
            offsets=offsets,
        )

    if variant_name == "SCALED_GEMM":
        from cutlass_api.arguments import ScaledTensor

        if any(
            mode is None
            for mode in (scale_mode_a, swizzle_mode_a, scale_mode_b, swizzle_mode_b)
        ):
            raise NotImplementedError(
                "Scaled GEMM requires supported scale and swizzle modes"
            )

        a, b, scale_a, scale_b = input_tensors
        scaled_a = ScaledTensor(a, scale_a, scale_mode_a, swizzle_mode_a)
        scaled_b = ScaledTensor(b, scale_b, scale_mode_b, swizzle_mode_b)
        return cutlass_api.arguments.GemmArguments(
            scaled_a,
            scaled_b,
            out,
            accumulator_type=accumulator_type,
        )

    if variant_name == "GEMM":
        a, b = input_tensors
        kwargs = {"accumulator_type": accumulator_type}
        if epilogue is not None:
            kwargs["epilogue"] = epilogue
        return cutlass_api.arguments.GemmArguments(a, b, out, **kwargs)

    raise NotImplementedError(f"Unsupported NVGEMM variant: {variant_name}")


def _lookup_gemm_kernel(
    kernel_name: str,
    *,
    epilogue_args: Any | None = None,
    epilogue_source: str = "",
):
    if epilogue_args is None:
        from torch._inductor.codegen.nv_universal_gemm.kernel_cache import (
            get_kernel_by_name,
        )

        kernel = get_kernel_by_name(kernel_name)
        if kernel is None:
            raise RuntimeError(f"Could not find kernel: {kernel_name}")
        return kernel

    from torch._inductor.codegen.nv_universal_gemm.kernel_cache import (
        get_efc_kernel_with_epilogue,
    )

    kernel = get_efc_kernel_with_epilogue(
        kernel_name,
        epilogue_args,
        epilogue_source=epilogue_source,
    )
    if kernel is None:
        raise RuntimeError(f"Could not find EFC kernel: {kernel_name}")
    return kernel


def _create_gemm_cache_key(
    variant_name: str,
    input_tensors,
    *,
    has_epilogue: bool = False,
    aux_tensors: tuple = (),
):
    if variant_name == "GROUPED_GEMM":
        a, b, offsets = input_tensors
        cache_key = (a.shape, a.dtype, b.shape, b.dtype, offsets.shape)
    elif variant_name == "SCALED_GEMM":
        a, b, scale_a, scale_b = input_tensors
        cache_key = (a.shape, a.dtype, b.shape, b.dtype, scale_a.shape, scale_b.shape)
    elif variant_name == "GEMM":
        a, b = input_tensors
        cache_key = (a.shape, a.dtype, b.shape, b.dtype)
    else:
        raise NotImplementedError(f"Unsupported NVGEMM variant: {variant_name}")

    if has_epilogue:
        # Aux tensors from the epilogue change the kernel's compiled artifact
        # (cutlass_api dispatches on their dtype/shape) but the base cache_key
        # only fingerprints A/B. Without folding aux tensor metadata in, a
        # wrapper invoked with the same A/B but a differently-shaped aux input
        # (e.g. dynamic-shape bias) would silently reuse a stale artifact.
        aux_sig = tuple((t.shape, t.dtype) for t in aux_tensors)
        return (*cache_key, "epilogue", aux_sig)
    return cache_key


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

    @staticmethod
    def _write_assign_call(
        code: IndentedBuffer,
        target: str,
        fn_name: str,
        args: tuple[str, ...] | list[str],
    ) -> None:
        code.writeline(f"{target} = {fn_name}(")
        with code.indent():
            for arg in args:
                code.writeline(f"{arg},")
        code.writeline(")")

    def _build_variant_render_spec(self) -> _VariantRenderSpec:
        if self.variant.name != "SCALED_GEMM":
            return _VariantRenderSpec()

        scale_mode_a, swizzle_mode_a, scale_mode_b, swizzle_mode_b = (
            _get_scaled_gemm_modes(
                self.scale_type_a,
                self.swizzle_type_a,
                self.scale_type_b,
                self.swizzle_type_b,
            )
        )
        return _VariantRenderSpec(
            import_lines=(
                "from cutlass_api.library import ScaleMode, ScaleSwizzleMode",
            ),
            helper_kwargs=(
                f"scale_mode_a=ScaleMode.{scale_mode_a.name}",
                f"swizzle_mode_a=ScaleSwizzleMode.{swizzle_mode_a.name}",
                f"scale_mode_b=ScaleMode.{scale_mode_b.name}",
                f"swizzle_mode_b=ScaleSwizzleMode.{swizzle_mode_b.name}",
            ),
        )

    def _build_epilogue_render_spec(self) -> _EpilogueRenderSpec:
        if not self.epilogue_fn_code:
            return _EpilogueRenderSpec()

        if self.variant.name != "GEMM":
            raise NotImplementedError(
                "Epilogue fusion is not yet supported for grouped or scaled GEMM variants"
            )

        epilogue_arg_lines = ["epilogue_fn=_epilogue_fn"]
        epilogue_kwargs = self._render_epilogue_kwargs()
        if epilogue_kwargs:
            epilogue_arg_lines.append(epilogue_kwargs)

        source_hash = hashlib.sha256(self.epilogue_fn_code.encode()).hexdigest()
        return _EpilogueRenderSpec(
            import_lines=("from cutlass_api.arguments import EpilogueArguments",),
            module_block=self.epilogue_fn_code,
            module_lines=(f'_EPILOGUE_FN_SOURCE = "{source_hash}"',),
            setup_arg_lines=tuple(epilogue_arg_lines),
            gemm_arg_kwargs=("epilogue=epi_args",),
            kernel_lookup_kwargs=(
                "epilogue_args=epi_args",
                "epilogue_source=_EPILOGUE_FN_SOURCE",
            ),
            enabled=True,
        )

    def render(self) -> str:
        """Render the Python source for the NVGEMM kernel wrapper."""
        kernel_name_str = self.kernel_metadata["kernel_name"]
        acc_dtype_str = CuteDSLOpOverrides.TORCH_TO_CUTE_DTYPE.get(
            self.accumulator_type, "cutlass.Float32"
        )

        input_tensor_names = [f"in_ptr{i}" for i, _ in enumerate(self.input_nodes)]
        input_params = list(input_tensor_names)
        input_params.append("out_ptr0")
        input_params.extend(self.epilogue_reads)
        if self.workspace_size > 0:
            input_params.append("workspace")
        input_params.append("stream=None")
        params_str = ", ".join(input_params)
        if len(input_tensor_names) == 1:
            input_tensors_expr = f"({input_tensor_names[0]},)"
        else:
            input_tensors_expr = f"({', '.join(input_tensor_names)})"

        workspace_arg = "workspace" if self.workspace_size > 0 else "None"
        var_prefix = self.variant.op_name.upper()
        cache_var = f"_{var_prefix}_compiled_cache"
        kernel_name_var = f"_{var_prefix}_KERNEL_NAME"
        variant_spec = self._build_variant_render_spec()
        epilogue_spec = self._build_epilogue_render_spec()

        code = IndentedBuffer()
        code.writeline("import cutlass")
        code.writeline(
            "from torch._inductor.codegen.nv_universal_gemm.nv_universal_gemm_kernel import ("
        )
        with code.indent():
            code.writeline("_create_gemm_arguments,")
            code.writeline("_create_gemm_cache_key,")
            code.writeline("_lookup_gemm_kernel,")
        code.writeline(")")
        for import_line in (*variant_spec.import_lines, *epilogue_spec.import_lines):
            code.writeline(import_line)
        code.writeline("")

        if epilogue_spec.module_block is not None:
            code.splice(epilogue_spec.module_block, strip=True)
            for line in epilogue_spec.module_lines:
                code.writeline(line)
            code.writeline("")

        code.writeline(f"{kernel_name_var} = {kernel_name_str!r}")
        code.writeline(f"{cache_var} = {{}}")
        code.writeline("")
        code.writeline(f"def {self.kernel_name}_main({params_str}):")
        with code.indent():
            code.writeline(f"global {cache_var}")
            code.writeline(f"input_tensors = {input_tensors_expr}")
            if epilogue_spec.enabled:
                self._write_assign_call(
                    code,
                    "epi_args",
                    "EpilogueArguments",
                    epilogue_spec.setup_arg_lines,
                )
            self._write_assign_call(
                code,
                "args",
                "_create_gemm_arguments",
                (
                    f'variant_name="{self.variant.name}"',
                    "input_tensors=input_tensors",
                    "out=out_ptr0",
                    f"accumulator_type={acc_dtype_str}",
                    *variant_spec.helper_kwargs,
                    *epilogue_spec.gemm_arg_kwargs,
                ),
            )
            code.writeline("")
            self._write_assign_call(
                code,
                "kernel",
                "_lookup_gemm_kernel",
                (kernel_name_var, *epilogue_spec.kernel_lookup_kwargs),
            )
            if epilogue_spec.enabled and self.epilogue_reads:
                aux_arg = (
                    "aux_tensors=("
                    + ", ".join(f"{name}" for name in self.epilogue_reads)
                    + ",)"
                )
                cache_key_args = (
                    f'variant_name="{self.variant.name}"',
                    "input_tensors=input_tensors",
                    f"has_epilogue={epilogue_spec.enabled}",
                    aux_arg,
                )
            else:
                cache_key_args = (
                    f'variant_name="{self.variant.name}"',
                    "input_tensors=input_tensors",
                    f"has_epilogue={epilogue_spec.enabled}",
                )
            self._write_assign_call(
                code,
                "cache_key",
                "_create_gemm_cache_key",
                cache_key_args,
            )
            code.writeline(f"artifact = {cache_var}.get(cache_key)")
            code.writeline("if artifact is None:")
            with code.indent():
                code.writeline("artifact = kernel.compile(args)")
                code.writeline(f"{cache_var}[cache_key] = artifact")
            code.writeline("")
            code.writeline(
                f"kernel.run(args, artifact, stream=stream, workspace={workspace_arg}, assume_supported_args=True)"
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
