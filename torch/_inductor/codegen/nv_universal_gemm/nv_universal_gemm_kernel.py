# mypy: allow-untyped-defs
"""
NVIDIA Universal GEMM kernel code generation.

This module generates Python code that calls cutlass_api to execute GEMM operations.
"""

from __future__ import annotations

import functools
import logging
from typing import Any, TYPE_CHECKING

import torch
from torch._inductor.codegen.common import Kernel, WorkspaceArg, WorkspaceZeroMode
from torch._inductor.codegen.cutedsl.cutedsl_op_overrides import CuteDSLOpOverrides
from torch._inductor.codegen.nv_universal_gemm.native_manifest import (
    build_stable_cache_key_inputs,
    default_tensor_descriptor_abi,
    export_compiled_artifact_to_cache,
    module_hooks_for_symbol,
    NVGEMM_NATIVE_C_ABI_VERSION,
    NVGemmNativeArtifact,
    NVGemmNativeManifest,
    NVGemmTensorArgSpec,
    pending_native_artifact,
    with_native_artifact,
)
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
from torch._inductor.kernel.mm_common import load_kernel_template
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

    Generates Python code that calls cutlass_api to execute GEMM operations
    using a Jinja template (nv_universal_gemm.py.jinja).
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

        self._template_input_args: list[tuple[str, Buffer]] = []
        self._seen_input_args: OrderedSet[str] = OrderedSet()

        for i, input_node in enumerate(input_nodes):
            param_name = f"in_ptr{i}"
            self._template_input_args.append((param_name, input_node))
            self._seen_input_args.add(param_name)

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def _get_template():
        from torch._inductor.codegen.common import jinja2_env

        env = jinja2_env()
        assert env is not None, (
            "jinja2 is required for NV Universal GEMM code generation"
        )
        source = load_kernel_template("nv_universal_gemm")
        return env.from_string(source)

    def render(self) -> str:
        """
        Render the NVIDIA Universal GEMM kernel code as a Python source string.

        Uses a Jinja template to generate Python code that:
        1. Looks up the cutlass_api kernel by name from the manifest
        2. Creates GemmArguments with the input/output tensors and accumulator type
        3. Compiles the kernel for the specific tensor shapes/dtypes (cached in memory
           and on disk to avoid recompilation)
        4. Runs the kernel with the compiled artifact and CUDA stream
        """
        from torch._inductor.codegen.nv_universal_gemm.nv_universal_gemm import (
            GemmVariant,
        )

        is_grouped = self.variant == GemmVariant.GROUPED_GEMM
        is_scaled = self.variant == GemmVariant.SCALED_GEMM

        acc_dtype_str = CuteDSLOpOverrides.TORCH_TO_CUTE_DTYPE.get(
            self.accumulator_type, "cutlass.Float32"
        )

        input_params = [f"in_ptr{i}" for i, _ in enumerate(self.input_nodes)]
        input_params.append("out_ptr0")
        if self.workspace_size > 0:
            input_params.append("workspace")
        input_params.append("stream=None")

        template_args: dict[str, Any] = {
            "kernel_name": self.kernel_name,
            "kernel_name_str": self.kernel_metadata["kernel_name"],
            "is_grouped": is_grouped,
            "is_scaled": is_scaled,
            "acc_dtype": acc_dtype_str,
            "params_str": ", ".join(input_params),
            "workspace_arg": "workspace" if self.workspace_size > 0 else "None",
            "input_ptrs": [f"in_ptr{i}" for i, _ in enumerate(self.input_nodes)],
        }

        if is_scaled:
            scale_mode_a, swizzle_mode_a = to_cutlass_scale_mode(
                self.scale_type_a, self.swizzle_type_a
            )
            scale_mode_b, swizzle_mode_b = to_cutlass_scale_mode(
                self.scale_type_b, self.swizzle_type_b
            )
            template_args["scale_mode_a"] = scale_mode_a.name if scale_mode_a else ""
            template_args["swizzle_mode_a"] = (
                swizzle_mode_a.name if swizzle_mode_a else ""
            )
            template_args["scale_mode_b"] = scale_mode_b.name if scale_mode_b else ""
            template_args["swizzle_mode_b"] = (
                swizzle_mode_b.name if swizzle_mode_b else ""
            )

        template = self._get_template()
        return template.render(**template_args)

    def _get_reinterpret_view(self, node) -> ReinterpretView | None:
        """Extract or convert to ReinterpretView from a node, handling all views."""
        while isinstance(node, MutableBox):
            node = node.data
        if isinstance(node, BaseView):
            return ExternKernel.convert_to_reinterpret_view(node)
        return None

    def _concrete_int_tuple(self, values) -> tuple[int, ...]:
        return tuple(int(V.graph.sizevars.simplify(x)) for x in values)

    def _make_compile_tensor(self, node) -> torch.Tensor:
        device = V.graph.get_current_device_or_throw()
        dtype = node.get_dtype()
        return torch.empty_strided(
            self._concrete_int_tuple(node.get_size()),
            self._concrete_int_tuple(node.get_stride()),
            device=device,
            dtype=dtype,
        )

    def _create_compile_args(self, input_tensors, out):
        import cutlass_api

        if self.variant.name == "GROUPED_GEMM":
            a, b, offsets = input_tensors
            return cutlass_api.arguments.GroupedGemmArguments(
                a,
                b,
                out,
                accumulator_type=self.accumulator_type,
                offsets=offsets,
            )
        elif self.variant.name == "SCALED_GEMM":
            from cutlass_api.arguments import ScaledTensor

            scale_mode_a, swizzle_mode_a = to_cutlass_scale_mode(
                self.scale_type_a, self.swizzle_type_a
            )
            scale_mode_b, swizzle_mode_b = to_cutlass_scale_mode(
                self.scale_type_b, self.swizzle_type_b
            )

            a, b, scale_a, scale_b = input_tensors
            scaled_a = ScaledTensor(a, scale_a, scale_mode_a, swizzle_mode_a)
            scaled_b = ScaledTensor(b, scale_b, scale_mode_b, swizzle_mode_b)
            return cutlass_api.arguments.GemmArguments(
                scaled_a,
                scaled_b,
                out,
                accumulator_type=self.accumulator_type,
            )
        else:
            a, b = input_tensors
            return cutlass_api.arguments.GemmArguments(
                a,
                b,
                out,
                accumulator_type=self.accumulator_type,
            )

    def _export_native_artifact(
        self,
        stable_cache_key_inputs: tuple[str, ...],
    ) -> NVGemmNativeArtifact:
        import cutlass_api.config

        from torch._inductor.codegen.nv_universal_gemm.kernel_cache import (
            get_kernel_by_name,
        )
        from torch._inductor.utils import _ensure_fp4_dtype_registered

        _ensure_fp4_dtype_registered()

        kernel = get_kernel_by_name(self.kernel_metadata["kernel_name"])
        if kernel is None:
            raise RuntimeError(
                f"Could not find NVGEMM kernel: {self.kernel_metadata['kernel_name']}"
            )

        input_tensors = tuple(
            self._make_compile_tensor(input_node)
            for _, input_node in self._template_input_args
        )
        out = self._make_compile_tensor(self.output_node)
        args = self._create_compile_args(input_tensors, out)

        global_options = cutlass_api.config.GlobalOptions()
        old_use_tvm_ffi = global_options.use_tvm_ffi
        global_options.use_tvm_ffi = False
        try:
            artifact = kernel.compile(args)
        finally:
            global_options.use_tvm_ffi = old_use_tvm_ffi

        return export_compiled_artifact_to_cache(
            artifact.compiled_obj,
            stable_cache_key_inputs,
        )

    def call_kernel(self, name: str, node=None):
        """
        Generate the kernel call in the wrapper code.

        Similar to CuteDSLTemplateKernel.call_kernel but simplified for NVIDIA Universal GEMM.
        """

        def stringify_shape(seq) -> tuple[str, ...]:
            return tuple(str(V.graph.sizevars.simplify(x)) for x in seq)

        def stringify_offset(node) -> str:
            return str(V.graph.sizevars.simplify(node.get_layout().offset))

        def tensor_arg_spec(name: str, role: str, node) -> NVGemmTensorArgSpec:
            return NVGemmTensorArgSpec(
                name=name,
                role=role,
                dtype=str(V.graph.get_dtype(node.get_name())),
                sizes=stringify_shape(node.get_size()),
                strides=stringify_shape(node.get_stride()),
                offset=stringify_offset(node),
            )

        def build_cpp_wrapper_manifest() -> NVGemmNativeManifest:
            tensor_args: list[NVGemmTensorArgSpec] = []
            for arg_name, input_node in zip(
                call_args[: len(self._template_input_args)],
                (node for _, node in self._template_input_args),
            ):
                tensor_args.append(tensor_arg_spec(arg_name, "input", input_node))

            tensor_args.append(
                NVGemmTensorArgSpec(
                    name=output_name,
                    role="output",
                    dtype=str(V.graph.get_dtype(output_name)),
                    sizes=stringify_shape(self.output_node.get_size()),
                    strides=stringify_shape(self.output_node.get_stride()),
                    offset=stringify_offset(self.output_node),
                )
            )
            if self.workspace_size > 0:
                assert ws is not None
                tensor_args.append(
                    NVGemmTensorArgSpec(
                        name=ws.outer_name,
                        role="workspace",
                        dtype=str(ws.dtype),
                        sizes=(str(self.workspace_size),),
                        strides=("1",),
                        offset="0",
                    )
                )

            tensor_args_tuple = tuple(tensor_args)
            cutlass_kernel_name = self.kernel_metadata["kernel_name"]
            variant = self.variant.op_name
            accumulator_type = str(self.accumulator_type)
            static_options = (
                f"min_cc={self.kernel_metadata.get('min_cc')}",
                f"scale_type_a={self.scale_type_a}",
                f"scale_type_b={self.scale_type_b}",
                f"swizzle_type_a={self.swizzle_type_a}",
                f"swizzle_type_b={self.swizzle_type_b}",
                f"tensor_descriptor_abi_version={NVGEMM_NATIVE_C_ABI_VERSION}",
            )
            stable_cache_key_inputs = build_stable_cache_key_inputs(
                cutlass_kernel_name=cutlass_kernel_name,
                variant=variant,
                accumulator_type=accumulator_type,
                workspace_size=self.workspace_size,
                tensor_args=tensor_args_tuple,
                static_options=static_options,
            )
            native_artifact = self._export_native_artifact(stable_cache_key_inputs)
            return with_native_artifact(
                NVGemmNativeManifest(
                    abi_version=NVGEMM_NATIVE_C_ABI_VERSION,
                    backend="nvgemm",
                    inductor_kernel_name=name,
                    python_entry=f"{name}_main",
                    cutlass_kernel_name=cutlass_kernel_name,
                    variant=variant,
                    accumulator_type=accumulator_type,
                    workspace_size=self.workspace_size,
                    tensor_args=tensor_args_tuple,
                    tensor_descriptor_abi=default_tensor_descriptor_abi(),
                    module_hooks=module_hooks_for_symbol(native_artifact.symbol_prefix),
                    stream_abi="cudaStream_t",
                    native_artifact=pending_native_artifact(
                        native_artifact.symbol_prefix
                    ),
                    stable_cache_key_inputs=stable_cache_key_inputs,
                    export_options={
                        "use_tvm_ffi": False,
                        "export_api": "compiled_obj.export_to_c(directory, file_name, function_prefix)",
                        "object_suffix": ".o",
                        "header_suffix": ".h",
                    },
                ),
                native_artifact,
            )

        def generate_call(*, triton: bool, inductor_meta: dict[str, Any] | None):
            wrapper.generate_kernel_call(
                name,
                call_args,
                triton=triton,
                arg_types=arg_types,
                raw_args=raw_args,
                raw_keys=[None] * len(raw_args),
                inductor_meta=inductor_meta,
            )

        wrapper = V.graph.wrapper_code

        call_args: list[str] = []
        arg_types: list[Any] = []
        raw_args: list[Buffer | ReinterpretView | None] = []

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

        if V.graph.cpp_wrapper:
            generate_call(
                triton=False,
                inductor_meta={
                    "backend": "nvgemm",
                    "manifest": build_cpp_wrapper_manifest().to_dict(),
                },
            )
        else:
            # Python wrapper path: NVGEMM is a PyCodeCache-loaded Python object
            # with a .run(...) method. It is Triton-like only at this level.
            generate_call(triton=True, inductor_meta=None)

        # Deallocate workspace after kernel call
        if ws is not None:
            wrapper.generate_workspace_deallocation(ws)
