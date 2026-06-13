# mypy: allow-untyped-defs
"""
NVIDIA Universal GEMM kernel code generation.

This module generates Python code that calls cutlass_api to execute GEMM operations.
The runtime helpers (_nvgemm_run, _nvgemm_precompile, etc.) are imported by the
generated wrapper at runtime, keeping the generated code thin.
"""

from __future__ import annotations

import hashlib
import importlib
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


# ── Runtime helpers (imported by generated wrapper code at runtime) ───────────


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

    match variant_name:
        case "GROUPED_GEMM":
            a, b, offsets = input_tensors
            return cutlass_api.arguments.GroupedGemmArguments(
                a,
                b,
                out,
                accumulator_type=accumulator_type,
                offsets=offsets,
            )

        case "SCALED_GEMM":
            from cutlass_api.arguments import ScaledTensor

            a, b, scale_a, scale_b = input_tensors
            scaled_a = ScaledTensor(a, scale_a, scale_mode_a, swizzle_mode_a)
            scaled_b = ScaledTensor(b, scale_b, scale_mode_b, swizzle_mode_b)
            return cutlass_api.arguments.GemmArguments(
                scaled_a,
                scaled_b,
                out,
                accumulator_type=accumulator_type,
            )

        case "GEMM":
            a, b = input_tensors
            kwargs: dict[str, Any] = {"accumulator_type": accumulator_type}
            if epilogue is not None:
                kwargs["epilogue"] = epilogue
            return cutlass_api.arguments.GemmArguments(a, b, out, **kwargs)

        case _:
            raise NotImplementedError(f"Unsupported NVGEMM variant: {variant_name}")


def _lookup_gemm_kernel(
    kernel_name: str,
    *,
    epilogue_args: Any | None = None,
    epilogue_source: str = "",
):
    from torch._inductor.codegen.nv_universal_gemm.kernel_cache import (
        get_efc_kernel_with_epilogue,
        get_kernel_by_name,
    )

    if epilogue_args is None:
        kernel = get_kernel_by_name(kernel_name)
        if kernel is None:
            raise RuntimeError(f"Could not find kernel: {kernel_name}")
        return kernel

    kernel = get_efc_kernel_with_epilogue(
        kernel_name, epilogue_args, epilogue_source=epilogue_source
    )
    if kernel is None:
        raise RuntimeError(f"Could not find EFC kernel: {kernel_name}")
    return kernel


def _tensor_sig(t):
    return (t.shape, t.stride(), t.dtype)


def _create_gemm_cache_key(
    input_tensors,
    out,
    *,
    has_epilogue: bool = False,
    aux_tensors: tuple = (),
):
    cache_key = tuple(s for t in input_tensors for s in _tensor_sig(t))
    cache_key = (*cache_key, *_tensor_sig(out))

    if has_epilogue:
        aux_sig = tuple(_tensor_sig(t) for t in aux_tensors)
        return (*cache_key, "epilogue", aux_sig)
    return cache_key


def _unwrap_efc_compiled_obj(compiled_obj):
    """Extract the inner JIT function from an EFC closure for disk serialization."""
    if hasattr(compiled_obj, "__closure__") and compiled_obj.__closure__:
        inner = compiled_obj.__closure__[0].cell_contents
        if hasattr(inner, "export_to_c"):
            return inner
    return compiled_obj


def _rewrap_efc_compiled_obj(compiled_fn, kernel):
    """Reconstruct the EFC wrapped_launch closure from a loaded JIT function."""
    if not hasattr(kernel, "impl") or not hasattr(kernel.impl, "efc"):
        return compiled_fn

    from cutlass_api.providers.cutedsl.gemm.sm100_static_persistent_efc import (
        KernelOperand,
        TensorWrapper,
    )

    def wrapped_launch(a_tensor, b_tensor, stream, *supplemental_args):
        runtime_args = [
            e.runtime_tensor
            if isinstance(e, TensorWrapper)
            else (e.tensor.runtime_tensor if isinstance(e, KernelOperand) else e)
            for e in supplemental_args
        ]
        return compiled_fn(
            a_tensor,
            b_tensor,
            stream,
            kernel.impl.efc.jit.pack_arguments(*runtime_args),
        )

    return wrapped_launch


def _nvgemm_run(
    variant_name: str,
    kernel_name: str,
    input_tensors: tuple,
    out,
    accumulator_type,
    compiled_cache: dict,
    disk_fn_cache: dict,
    module_path: str,
    disk_config_key: tuple,
    *,
    stream=None,
    workspace=None,
    variant_kwargs: dict | None = None,
    epilogue_args=None,
    epilogue_source: str = "",
    has_epilogue: bool = False,
    aux_tensors: tuple = (),
):
    from cutlass_api.artifact import CompiledArtifact

    from torch._inductor.runtime.cutedsl_cache import disk_cache_get, disk_cache_set

    args = _create_gemm_arguments(
        variant_name,
        input_tensors,
        out,
        accumulator_type,
        epilogue=epilogue_args,
        **(variant_kwargs or {}),
    )
    kernel = _lookup_gemm_kernel(
        kernel_name,
        epilogue_args=epilogue_args,
        epilogue_source=epilogue_source,
    )

    cache_key = _create_gemm_cache_key(
        input_tensors,
        out,
        has_epilogue=has_epilogue,
        aux_tensors=aux_tensors,
    )
    dev_idx = input_tensors[0].device.index or 0
    mem_key = (cache_key, dev_idx)
    artifact = compiled_cache.get(mem_key)
    if artifact is None:
        compiled_fn = disk_cache_get(
            disk_fn_cache, module_path, disk_config_key, cache_key, dev_idx
        )
        if compiled_fn is not None:
            compiled_fn = _rewrap_efc_compiled_obj(compiled_fn, kernel)
            artifact = CompiledArtifact(compiled_fn, kernel)
        else:
            artifact = kernel.compile(args)
            disk_cache_set(
                disk_fn_cache,
                module_path,
                disk_config_key,
                cache_key,
                _unwrap_efc_compiled_obj(artifact.compiled_obj),
                dev_idx,
            )
        compiled_cache[mem_key] = artifact

    kernel.run(
        args,
        artifact,
        stream=stream,
        workspace=workspace,
        assume_supported_args=True,
    )


_MAX_ACTIVE_CLUSTERS_MODULES = [
    "cutlass_api.providers.cutedsl.utils",
    "cutlass_api.providers.cutedsl.gemm.sm100_static_persistent",
    "cutlass_api.providers.cutedsl.gemm.sm100_static_persistent_efc",
    "cutlass_api.providers.cutedsl.gemm.sm100_dense_blockscaled_static_persistent",
    "cutlass_api.providers.cutedsl.gemm.sm100_contiguous_offset_2d3d_dense_gemm",
]


def _nvgemm_precompile(
    precompile_shapes: dict,
    precompile_strides: dict,
    precompile_dtypes: dict,
    device_index: int = 0,
    device_capability: tuple | None = None,
    *,
    variant_name: str,
    kernel_name: str,
    accumulator_type,
    compiled_cache: dict,
    disk_fn_cache: dict,
    module_path: str,
    disk_config_key: tuple,
    input_param_names: list[str],
    variant_kwargs: dict | None = None,
    max_active_clusters: int | None = None,
):
    """Precompile an NVGEMM kernel in a subprocess for parallel compilation.

    Precompiles only the base (non-EFC) kernel. EFC kernels produce
    closure-wrapped artifacts that can't be serialized to disk cache.
    """
    import torch
    from torch._inductor.runtime.cutedsl_cache import disk_cache_set
    from torch._subclasses.fake_tensor import FakeTensorMode

    if max_active_clusters is None:
        return

    device = f"cuda:{device_index}"
    with FakeTensorMode():
        tensors = {}
        for name in [*input_param_names, "output"]:
            tensors[name] = torch.empty_strided(
                tuple(precompile_shapes[name]),
                tuple(precompile_strides[name]),
                device=device,
                dtype=getattr(torch, precompile_dtypes[name]),
            )

    input_tensors = tuple(tensors[n] for n in input_param_names)
    out = tensors["output"]

    patched_mods = []

    def mac_fn(*a, **kw):
        return max_active_clusters

    for mod_name in _MAX_ACTIVE_CLUSTERS_MODULES:
        try:
            m = importlib.import_module(mod_name)
        except ImportError:
            continue
        if hasattr(m, "get_max_active_clusters"):
            patched_mods.append((m, m.get_max_active_clusters))
            m.get_max_active_clusters = mac_fn  # pyrefly: ignore [missing-attribute]

    try:
        args = _create_gemm_arguments(
            variant_name,
            input_tensors,
            out,
            accumulator_type,
            **(variant_kwargs or {}),
        )
        kernel = _lookup_gemm_kernel(kernel_name)
        cache_key = _create_gemm_cache_key(input_tensors, out)
        mem_key = (cache_key, device_index)
        if mem_key not in compiled_cache:
            artifact = kernel.compile(args)
            disk_cache_set(
                disk_fn_cache,
                module_path,
                disk_config_key,
                cache_key,
                artifact.compiled_obj,
                device_index,
                device_capability=device_capability,
            )
            compiled_cache[mem_key] = artifact
    finally:
        for m, orig in patched_mods:
            m.get_max_active_clusters = orig  # pyrefly: ignore [missing-attribute]


# ── Kernel wrapper (returned by async_compile.nv_universal_gemm) ─────────────


class NVUniversalGemmKernelWrapper:
    """Wrapper to provide .run() interface for NVIDIA Universal GEMM kernels."""

    def __init__(self, kernel_fn, kernel_path: str | None = None):
        self.kernel_fn = kernel_fn
        self.kernel_path = kernel_path

    def run(self, *args, stream=None, **kwargs):
        """Execute the NVIDIA Universal GEMM kernel."""
        return self.kernel_fn(*args, stream=stream, **kwargs)


# ── Kernel codegen class ─────────────────────────────────────────────────────


class NVUniversalGemmKernel(Kernel):
    """
    Kernel implementation for NVIDIA Universal GEMM.

    Generates a thin Python wrapper that delegates to runtime helpers
    (_nvgemm_run, _nvgemm_precompile) for the actual GEMM execution and
    compilation logic.
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

        for i, input_node in enumerate(input_nodes):
            param_name = f"in_ptr{i}"
            self._template_input_args.append((param_name, input_node))

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
        has_epilogue = bool(self.epilogue_fn_code)

        # Build variant_kwargs dict expression for SCALED_GEMM
        variant_kwargs_expr = "None"
        variant_extra_imports = ""
        if self.variant.name == "SCALED_GEMM":
            sma, swza, smb, swzb = _get_scaled_gemm_modes(
                self.scale_type_a,
                self.swizzle_type_a,
                self.scale_type_b,
                self.swizzle_type_b,
            )
            variant_extra_imports = (
                "from cutlass_api.library import ScaleMode, ScaleSwizzleMode"
            )
            variant_kwargs_expr = (
                "{"
                f'"scale_mode_a": ScaleMode.{sma.name}, '
                f'"swizzle_mode_a": ScaleSwizzleMode.{swza.name}, '
                f'"scale_mode_b": ScaleMode.{smb.name}, '
                f'"swizzle_mode_b": ScaleSwizzleMode.{swzb.name}'
                "}"
            )

        code = IndentedBuffer()

        # -- Module-level imports and constants --
        code.writeline("import cutlass")
        code.writeline(
            "from torch._inductor.codegen.nv_universal_gemm.nv_universal_gemm_kernel import ("
        )
        with code.indent():
            code.writeline("_nvgemm_run,")
            code.writeline("_nvgemm_precompile,")
        code.writeline(")")
        code.writeline("from torch._inductor.utils import _ensure_fp4_dtype_registered")
        code.writeline("_ensure_fp4_dtype_registered()")
        if variant_extra_imports:
            code.writeline(variant_extra_imports)
        if has_epilogue:
            code.writeline("from cutlass_api.arguments import EpilogueArguments")
        code.writeline("")

        # -- Epilogue function definition (must be module-level for cutlass_api) --
        epilogue_fn_code = self.epilogue_fn_code
        if has_epilogue and epilogue_fn_code is not None:
            code.splice(epilogue_fn_code, strip=True)
            epilogue_source_hash = hashlib.sha256(epilogue_fn_code.encode()).hexdigest()
            code.writeline(f'_EPILOGUE_FN_SOURCE = "{epilogue_source_hash}"')
            code.writeline("")

        # -- Module-level state --
        code.writeline(f"_KERNEL_NAME = {kernel_name_str!r}")
        code.writeline("_DISK_CACHE_CONFIG_KEY = (_KERNEL_NAME,)")
        code.writeline("_compiled_cache = {}")
        code.writeline("_disk_fn_cache = {}")
        code.writeline(f"_VARIANT_KWARGS = {variant_kwargs_expr}")
        code.writeline(f"_ACC_TYPE = {acc_dtype_str}")
        input_param_names_repr = repr(input_tensor_names)
        code.writeline(f"_INPUT_PARAM_NAMES = {input_param_names_repr}")
        code.writeline("")

        # -- Main function --
        code.writeline(f"def {self.kernel_name}_main({params_str}):")
        with code.indent():
            # Build epilogue args if needed (user-specific variable names)
            epi_args_expr = "None"
            epi_source_expr = '""'
            aux_tensors_expr = "()"
            if has_epilogue:
                epilogue_kwargs = self._render_epilogue_kwargs()
                epi_kwargs_str = "epilogue_fn=_epilogue_fn"
                if epilogue_kwargs:
                    epi_kwargs_str += f", {epilogue_kwargs}"
                code.writeline(f"epi_args = EpilogueArguments({epi_kwargs_str})")
                epi_args_expr = "epi_args"
                epi_source_expr = "_EPILOGUE_FN_SOURCE"
                if self.epilogue_reads:
                    aux_tensors_expr = "(" + ", ".join(self.epilogue_reads) + ",)"

            code.writeline("_nvgemm_run(")
            with code.indent():
                code.writeline(f'"{self.variant.name}", _KERNEL_NAME,')
                code.writeline(f"{input_tensors_expr}, out_ptr0, _ACC_TYPE,")
                code.writeline(
                    "_compiled_cache, _disk_fn_cache, __file__, _DISK_CACHE_CONFIG_KEY,"
                )
                code.writeline(f"stream=stream, workspace={workspace_arg},")
                code.writeline("variant_kwargs=_VARIANT_KWARGS,")
                code.writeline(f"epilogue_args={epi_args_expr},")
                code.writeline(f"epilogue_source={epi_source_expr},")
                code.writeline(f"has_epilogue={has_epilogue},")
                code.writeline(f"aux_tensors={aux_tensors_expr},")
            code.writeline(")")

        # -- Precompile hook --
        code.writeline("")
        code.writeline(
            f"def {self.kernel_name}_precompile("
            "precompile_shapes, precompile_strides, precompile_dtypes, "
            "device_index=0, device_capability=None, **kwargs):"
        )
        with code.indent():
            code.writeline("_nvgemm_precompile(")
            with code.indent():
                code.writeline(
                    "precompile_shapes, precompile_strides, precompile_dtypes,"
                )
                code.writeline("device_index, device_capability,")
                code.writeline(f'variant_name="{self.variant.name}",')
                code.writeline("kernel_name=_KERNEL_NAME,")
                code.writeline("accumulator_type=_ACC_TYPE,")
                code.writeline(
                    "compiled_cache=_compiled_cache, disk_fn_cache=_disk_fn_cache,"
                )
                code.writeline(
                    "module_path=__file__, disk_config_key=_DISK_CACHE_CONFIG_KEY,"
                )
                code.writeline("input_param_names=_INPUT_PARAM_NAMES,")
                code.writeline("variant_kwargs=_VARIANT_KWARGS,")
                code.writeline("max_active_clusters=kwargs.get('max_active_clusters'),")
            code.writeline(")")

        return code.getvalue()

    def _render_epilogue_kwargs(self) -> str:
        """Render kwargs for EpilogueArguments constructor.

        Skips intermediate stores (write_buffer entries from CutlassEVTCodegen.store()
        in a multi-node epilogue chain) -- those names are not kernel parameters and
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
                # Epilogue may read model parameters/inputs (e.g. bias) that are
                # graph inputs rather than computed buffers.
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
