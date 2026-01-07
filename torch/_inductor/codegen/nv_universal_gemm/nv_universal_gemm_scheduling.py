# mypy: allow-untyped-defs
"""
NVIDIA Universal GEMM scheduling for PyTorch Inductor.
"""

import hashlib
import logging
from collections.abc import Sequence
from typing import Any, cast

from torch._inductor.utils import (
    get_fused_kernel_name,
    get_kernel_metadata,
    Placeholder,
)
from torch.utils._ordered_set import OrderedSet

from ... import config
from ...codecache import code_hash, get_path
from ...ir import ComputedBuffer, MultiTemplateBuffer, NVUniversalGemmBuffer, Pointwise
from ...scheduler import (
    BaseSchedulerNode,
    BaseScheduling,
    FusedSchedulerNode,
    SchedulerNode,
)
from ...virtualized import V
from ..common import BackendFeature, IndentedBuffer
from ..cuda.cutlass_python_evt import CutlassEVTCodegen
from .nv_universal_gemm import NVUniversalGemmCaller


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
        """Check if a node is a NVIDIA Universal GEMM template.

        Returns True if the node wraps:
        1. A NVUniversalGemmBuffer directly, OR
        2. A MultiTemplateBuffer whose winning (min) choice is an NVUniversalGemmCaller
        """
        if not isinstance(node, SchedulerNode):
            return False

        ir_node = node.node
        if isinstance(ir_node, NVUniversalGemmBuffer):
            return True
        elif isinstance(ir_node, MultiTemplateBuffer):
            # Check if the winning choice would be NVGEMM
            try:
                min_choice, _ = ir_node.get_min_choice()
                return isinstance(min_choice, NVUniversalGemmCaller)
            except Exception:
                return False
        return False

    @staticmethod
    def get_nv_gemm_buffer_from_node(node: BaseSchedulerNode) -> NVUniversalGemmBuffer:
        """Extract NVUniversalGemmBuffer from a scheduler node.

        Works with both direct NVUniversalGemmBuffer and MultiTemplateBuffer
        whose winning choice is NVGEMM.
        """
        assert isinstance(node, SchedulerNode)
        ir_node = node.node

        if isinstance(ir_node, NVUniversalGemmBuffer):
            return ir_node
        elif isinstance(ir_node, MultiTemplateBuffer):
            min_choice, _ = ir_node.get_min_choice()
            assert isinstance(min_choice, NVUniversalGemmCaller)
            # Get the NVUniversalGemmBuffer from the caller's output_node()
            tensor_box = min_choice.output_node()
            return cast(NVUniversalGemmBuffer, tensor_box.data.data)

    def is_nv_universal_gemm_fused_template(self, node: BaseSchedulerNode) -> bool:
        """Check if a node is a fused NVIDIA Universal GEMM template."""
        if not isinstance(node, FusedSchedulerNode):
            return False
        template_node = node.get_template_node()
        return self.is_nv_universal_gemm_template(template_node)

    def can_fuse_vertical(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        """
        Check if node2 can be fused as an epilogue to node1 (NVGEMM template).

        Supports fusing pointwise operations as epilogues.
        """
        if self.is_nv_universal_gemm_template(node1):
            return self._can_fuse_epilogue_impl(
                cast(SchedulerNode, node1),
                [],
                node2,
            )
        elif self.is_nv_universal_gemm_fused_template(node1):
            fnode1 = cast(FusedSchedulerNode, node1)
            template_node = fnode1.get_template_node()
            return self._can_fuse_epilogue_impl(
                template_node,
                self._unwrap_epilogue_nodes(fnode1),
                node2,
            )
        return False

    def _unwrap_epilogue_nodes(
        self, fused_node: FusedSchedulerNode
    ) -> list[BaseSchedulerNode]:
        """Extract epilogue nodes from a fused node."""
        epilogue_nodes = []
        for node in fused_node.snodes:
            if not self.is_nv_universal_gemm_template(node):
                epilogue_nodes.append(node)
        return epilogue_nodes

    def _can_fuse_epilogue_impl(
        self,
        gemm_template_node: SchedulerNode,
        existing_epilogue_nodes: list[BaseSchedulerNode],
        node_to_fuse: BaseSchedulerNode,
    ) -> bool:
        """
        Check if the given node can be fused as an epilogue.

        Supports fusion with Pointwise operations wrapped in ComputedBuffer nodes.
        """
        # Get the IR buffer (could be NVUniversalGemmBuffer or MultiTemplateBuffer)
        ir_node = gemm_template_node.node

        scheduler_nodes_to_fuse = node_to_fuse.get_nodes()

        # Checks on constituent nodes
        for s_node in scheduler_nodes_to_fuse:
            node = s_node.node

            if not isinstance(node, ComputedBuffer):
                log.debug("NVGEMM epilogue fusion: %s is not a ComputedBuffer", node)
                return False
            elif not isinstance(node.data, Pointwise):
                log.debug("NVGEMM epilogue fusion: %s is not a Pointwise op", node)
                return False

            # Size must match the GEMM output
            if node.get_size() != ir_node.get_size():
                log.debug(
                    "NVGEMM epilogue fusion: size mismatch %s vs %s",
                    node.get_size(),
                    ir_node.get_size(),
                )
                return False

        # First epilogue node must read from the GEMM template buffer
        if not existing_epilogue_nodes:
            reads = OrderedSet(rd.name for rd in node_to_fuse.read_writes.reads)
            # Use the original buffer name (works for both NVUniversalGemmBuffer and MultiTemplateBuffer)
            if ir_node.get_name() not in reads:
                log.debug(
                    "NVGEMM epilogue fusion: first epilogue node doesn't read from GEMM output"
                )
                return False

        if node_to_fuse.has_aliasing_or_mutation():
            log.debug("NVGEMM epilogue fusion: node has aliasing or mutation")
            return False
        elif node_to_fuse.is_reduction():
            log.debug("NVGEMM epilogue fusion: reductions not supported")
            return False

        return True

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
        Codegen a NVIDIA Universal GEMM template with optional epilogue fusion.
        """
        log.debug(
            "NVGEMM codegen_template: template_node=%s, epilogue_nodes=%s, prologue_nodes=%s",
            template_node,
            [n.get_name() for n in epilogue_nodes] if epilogue_nodes else [],
            [n.get_name() for n in prologue_nodes] if prologue_nodes else [],
        )
        assert self.is_nv_universal_gemm_template(template_node), (
            "Template node passed to NVUniversalGemmScheduling.codegen_template must be a "
            "SchedulerNode that wraps a NVUniversalGemmBuffer or MultiTemplateBuffer with NVGEMM choice"
        )
        # Prologue fusion is not yet supported
        assert not prologue_nodes, (
            "NVIDIA Universal GEMM doesn't support prologue fusion yet"
        )

        template_node = cast(SchedulerNode, template_node)

        # Get the original buffer name (for epilogue processing - could be MultiTemplateBuffer name)
        original_ir_node = template_node.node
        original_buffer_name = original_ir_node.get_name()

        # Get the NVUniversalGemmBuffer (extract from MultiTemplateBuffer if needed)
        ctb: NVUniversalGemmBuffer = self.get_nv_gemm_buffer_from_node(template_node)

        # Process epilogue nodes if present
        epilogue_fn_code: str | None = None
        epilogue_reads: list[str] = []
        epilogue_writes: list[str] = []
        epilogue_var_renames: dict[str, Any] = {}

        if epilogue_nodes:
            try:
                # Add GEMM output to removed_buffers so it's not treated as a store
                # (it becomes the 'accum' input to the epilogue)
                removed_buffers_with_gemm = V.graph.removed_buffers | {original_buffer_name}

                reads, writes, var_renames, evt_code = CutlassEVTCodegen.ir_to_evt_python_code(
                    original_buffer_name,  # GEMM output node name (may be MultiTemplateBuffer name)
                    list(epilogue_nodes),
                    removed_buffers_with_gemm,
                )
                epilogue_fn_code = evt_code
                epilogue_reads = reads
                epilogue_writes = writes
                epilogue_var_renames = var_renames

                # Mark epilogue nodes as run since they will be fused
                for node in epilogue_nodes:
                    node.mark_run()
                    V.graph.removed_buffers.add(node.get_name())

                log.debug(
                    "NVGEMM epilogue fusion: %d nodes, reads=%s, writes=%s",
                    len(epilogue_nodes),
                    epilogue_reads,
                    epilogue_writes,
                )
            except (NotImplementedError, AssertionError) as e:
                # Fallback: epilogue will run as separate kernel(s)
                log.debug("NVGEMM epilogue fusion failed, falling back: %s", e)
                epilogue_fn_code = None
                epilogue_reads = []
                epilogue_writes = []
                epilogue_var_renames = {}
                # Don't mark epilogue nodes as run - scheduler handles them separately

        assert ctb.make_kernel_render is not None
        kernel, render = ctb.make_kernel_render(
            ctb,
            epilogue_fn_code=epilogue_fn_code,
            epilogue_reads=epilogue_reads,
            epilogue_writes=epilogue_writes,
            epilogue_var_renames=epilogue_var_renames,
        )
        template_node.mark_run()
        src_code = render()

        with V.set_kernel_handler(kernel):
            node_schedule = [template_node]
            # Include epilogue nodes in schedule if they were fused
            if epilogue_fn_code and epilogue_nodes:
                node_schedule.extend(epilogue_nodes)
            kernel_name = self.define_kernel(src_code, node_schedule)

        self.codegen_comment(node_schedule, kernel_name)
        kernel.call_kernel(kernel_name, ctb)
        V.graph.removed_buffers |= kernel.removed_buffers
        self.free_buffers_in_scheduler()
