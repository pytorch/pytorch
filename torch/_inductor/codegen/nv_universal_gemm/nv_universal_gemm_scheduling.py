# mypy: allow-untyped-defs
"""
NVIDIA Universal GEMM scheduling for PyTorch Inductor.
"""

import hashlib
import logging
from collections.abc import Sequence
from typing import cast

from torch._inductor.utils import (
    get_fused_kernel_name,
    get_kernel_metadata,
    Placeholder,
)
from torch.utils._ordered_set import OrderedSet
from ... import config
from ...codecache import code_hash, get_path
from ...ir import NVUniversalGemmBuffer
from ...scheduler import (
    BaseSchedulerNode,
    BaseScheduling,
    FusedSchedulerNode,
    SchedulerNode,
)
from ...virtualized import V
from ..common import BackendFeature, IndentedBuffer


log = logging.getLogger(__name__)

MAIN_SUFFIX = "main"


class NVUniversalGemmScheduling(BaseScheduling):
    """
    Scheduling implementation for NVIDIA Universal GEMM kernels.

    This class is intended to be used in combination with other schedulers,
    and delegated to by CUDACombinedScheduling.
    """

    @classmethod
    def get_backend_features(cls, device) -> OrderedSet[BackendFeature]:
        return OrderedSet()

    @staticmethod
    def is_nv_universal_gemm_template(node: BaseSchedulerNode) -> bool:
        """Check if a node is a NVIDIA Universal GEMM template."""
        return isinstance(node, SchedulerNode) and isinstance(
            node.node, NVUniversalGemmBuffer
        )

    def is_nv_universal_gemm_fused_template(self, node: BaseSchedulerNode) -> bool:
        """Check if a node is a fused NVIDIA Universal GEMM template."""
        return isinstance(
            node, FusedSchedulerNode
        ) and self.is_nv_universal_gemm_template(node)

    def can_fuse_vertical(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        # NVIDIA Universal GEMM templates don't support vertical fusion yet
        return False

    def can_fuse_horizontal(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        # NVIDIA Universal GEMM templates don't support horizontal fusion yet
        return False

    def define_kernel(self, src_code: str, node_schedule) -> str:
        """
        Define a NVIDIA Universal GEMM kernel by writing source code and generating wrapper.

        Based on CuteDSLScheduling.define_kernel.
        """
        wrapper = V.graph.wrapper_code

        # Use the string as the key for caching
        if src_code in wrapper.src_to_kernel:
            return wrapper.src_to_kernel[src_code]

        fused_name = (
            get_fused_kernel_name(node_schedule, config.triton.descriptive_names)
            if config.triton.descriptive_names
            else ""
        )

        kernel_hash = hashlib.sha256(src_code.encode("utf-8")).hexdigest()[:8]
        if fused_name == "fused":
            kernel_name = f"nv_universal_gemm_{kernel_hash}"
        else:
            kernel_name = f"nv_universal_gemm_{fused_name}_{kernel_hash}"

        wrapper.src_to_kernel[src_code] = kernel_name

        src_code = src_code.replace(str(Placeholder.KERNEL_NAME), kernel_name)

        _, _, kernel_path = get_path(code_hash(src_code), "py")

        compile_wrapper = IndentedBuffer()
        compile_wrapper.writeline(
            f"async_compile.nv_universal_gemm({kernel_name!r}, r'''"
        )
        compile_wrapper.splice(src_code, strip=True)
        compile_wrapper.writeline("''')")

        metadata_comment = f"# kernel path: {kernel_path}"
        origins, detailed_origins = get_kernel_metadata(node_schedule, wrapper)
        metadata_comment += "\n" + origins + "\n" + detailed_origins
        wrapper.define_kernel(kernel_name, compile_wrapper.getvalue(), metadata_comment)

        return kernel_name

    def codegen_template(
        self,
        template_node: BaseSchedulerNode,
        epilogue_nodes: Sequence[BaseSchedulerNode],
        prologue_nodes: Sequence[BaseSchedulerNode],
    ):
        """
        Codegen a NVIDIA Universal GEMM template. Currently doesn't support fusion.
        """
        assert self.is_nv_universal_gemm_template(template_node), (
            "Template node passed to NVUniversalGemmScheduling.codegen_template must be a "
            "SchedulerNode that wraps a NVUniversalGemmBuffer"
        )
        # TODO: add support for fusion when needed
        assert not epilogue_nodes, (
            "NVIDIA Universal GEMM doesn't support epilogue fusion yet"
        )
        assert not prologue_nodes, (
            "NVIDIA Universal GEMM doesn't support prologue fusion yet"
        )

        template_node = cast(SchedulerNode, template_node)
        ctb: NVUniversalGemmBuffer = cast(NVUniversalGemmBuffer, template_node.node)

        assert ctb.make_kernel_render is not None
        kernel, render = ctb.make_kernel_render(ctb)
        template_node.mark_run()
        src_code = render()

        with V.set_kernel_handler(kernel):
            node_schedule = [template_node]
            kernel_name = self.define_kernel(src_code, node_schedule)

        self.codegen_comment(node_schedule, kernel_name)
        kernel.call_kernel(kernel_name, ctb)
        V.graph.removed_buffers |= kernel.removed_buffers
        self.free_buffers_in_scheduler()
