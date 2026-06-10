# mypy: allow-untyped-defs
"""
NVIDIA Universal GEMM kernel code generation.

This module generates Python code that calls cutlass_api to execute GEMM operations.
"""

from __future__ import annotations

import functools
import logging
from typing import Any, TYPE_CHECKING

from torch._inductor.codegen.common import Kernel, WorkspaceArg, WorkspaceZeroMode
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
        if env is None:
            raise AssertionError(
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

    def call_kernel(self, name: str, node=None):
        """
        Generate the kernel call in the wrapper code.

        Similar to CuteDSLTemplateKernel.call_kernel but simplified for NVIDIA Universal GEMM.
        """
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
