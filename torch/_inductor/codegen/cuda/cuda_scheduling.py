import logging
from typing import Callable, cast, List

from ... import config, ir
from ...codecache import code_hash, get_path
from ...ir import ComputedBuffer, CUDATemplateBuffer, Pointwise
from ...scheduler import BaseSchedulerNode, FusedSchedulerNode, Scheduler, SchedulerNode
from ...utils import get_fused_kernel_name, get_kernel_metadata
from ...virtualized import V
from ..common import IndentedBuffer
from ..triton import TritonScheduling

log = logging.getLogger(__name__)


def _can_fuse_epilogue(
    cuda_template_buffer: CUDATemplateBuffer,
    epilogue_nodes: List[ir.Buffer],
    additional_node: ir.Buffer,
) -> bool:
    """

    Check if the given node can be fused with the epilogue. At the moment, Kernels
    support fusion with Pointwise operations, wrapped in (named) ComputedBuffer nodes.

    Args:
        cuda_template_buffer : A CUDATemplateBuffer object representing the CUDA template and it's result buffer
        epilogue_nodes : List[ir.Buffer]: The list of already fused epilogue nodes.
        additional_node: The ir.Buffer node to be checked if it can be fused with the epilogue.
    Returns:
    - bool: True if the given node can be fused with the epilogue, False otherwise.

    """
    if not cuda_template_buffer.template.can_fuse_epilogue:
        # The used GEMM op does not support fusing epilogues
        return False

    if not isinstance(additional_node, ComputedBuffer):
        return False
    if not isinstance(additional_node.data, Pointwise):
        return False

    # We can fuse a Pointwise op that depends on the last fused epilogue node
    # if any. If there is no epilogue node yet, it needs to depend on the template
    # node
    node_name = additional_node.name if additional_node.name is not None else additional_node.data.name  # type: ignore[attr-defined] # noqa: B950
    if node_name is None:
        return False

    if len(epilogue_nodes) == 0:
        if cuda_template_buffer.name not in additional_node.get_read_names():
            return False
    else:
        last_epilogue_node = epilogue_nodes[-1]
        last_epilogue_name = (
            last_epilogue_node.name
            if last_epilogue_node.name is not None
            else last_epilogue_node.data.name  # type: ignore[attr-defined]
        )
        if last_epilogue_name not in additional_node.get_read_names():
            return False
    if additional_node.layout != cuda_template_buffer.layout:
        return False
    try:
        from torch._inductor.codegen.cuda.cutlass_epilogue_gen import (
            CutlassEVTEpilogueArgumentFormatter,
            CutlassEVTEpilogueTypeFormatter,
        )

        CutlassEVTEpilogueTypeFormatter.ir_to_evt_string(
            cast(str, cuda_template_buffer.name), "anything", [additional_node]
        )
        CutlassEVTEpilogueArgumentFormatter.ir_to_evt_argument_string(
            cast(str, cuda_template_buffer.name), [additional_node]
        )
    except NotImplementedError as e:
        not_implemented_op = str(e)
        if not_implemented_op.startswith("_op_"):
            not_implemented_op = not_implemented_op[4:]
            log.warning(
                f"Cannot fuse epilogue node {additional_node} into {cuda_template_buffer.name}, likely due to unsupported operation: {not_implemented_op}"  # noqa: G004, B950
            )
            return False
        else:
            # Likely due to unsupported dtype.
            log.warning(
                f"Cannot fuse epilogue node {additional_node} into {cuda_template_buffer.name}. Reason: {not_implemented_op}"  # noqa: G004, B950
            )
            return False
    return True


class CUDASchedulerNode(SchedulerNode):
    """
    A SchedulerNode that represents a single CUDA kernel,
    which may alllow epilogue fusions.
    """

    def __init__(self, scheduler: Scheduler, node: CUDATemplateBuffer, group_fn: Callable):  # type: ignore[type-arg]
        """
        Initializes a new instance of the CUDASchedulerNode class.
        Args:
            scheduler: The Scheduler object that this node belongs to.
            node: The CUDATemplateBuffer object representing the CUDA kernel, fused IRNode epilogues
                    and its inputs and outputs.
            group_fn: A function that returns a group key which determines whether two nodes may be considered
                         for fusion
        """
        assert isinstance(node, CUDATemplateBuffer)
        super().__init__(scheduler, node, group_fn)
        self.group_fn = group_fn  # keeping this to enable cloning during fuse_epilogue

    def can_fuse_epilogue(self, other_node: SchedulerNode) -> bool:
        """
        Determines whether this FusedCUDASchedulerNode can fuse another node as epilogue.
        """
        if other_node.get_device() != self.get_device():
            return False
        return _can_fuse_epilogue(
            cast(CUDATemplateBuffer, self.node), [], other_node.node
        )

    def fuse_epilogue(self, other_node: SchedulerNode) -> FusedSchedulerNode:
        """
        Fuses the current FusedCUDASchedulerNode with another (Epilogue) Scheduler Node
        and returns a new FusedCUDASchedulerNode
        """
        assert self.can_fuse_epilogue(other_node)
        return FusedCUDASchedulerNode(self, self.scheduler, [other_node])  # type: ignore[arg-type]


_cuda_epilogue_fusion_counter: int = (
    0  # Used by unit tests to verify fusions are happening / not happening
)


class FusedCUDASchedulerNode(FusedSchedulerNode):
    def __init__(
        self,
        cuda_scheduler_node: CUDASchedulerNode,
        scheduler: Scheduler,
        epilogue_scheduler_nodes: List[SchedulerNode],
    ):
        """
        Initializes a FusedCUDASchedulerNode object. This object represents a CUDA kernel with fused epilogues
        on the Scheduler level. Apart from the FusedSchedulerNode baseclass functionality,
        it is mostly a wrapper around the CUDASchedulerNode and the CUDATemplateBuffer
        that it contains.

        Args:
            cuda_scheduler_node (CUDASchedulerNode): The CUDA scheduler node that will be fused.
            scheduler (Scheduler): The overall scheduler object.
            epilogue_scheduler_nodes (List[SchedulerNode]): A list of scheduler nodes that are fused as epilogue.
        """
        global _cuda_epilogue_fusion_counter
        _cuda_epilogue_fusion_counter += 1
        assert isinstance(cuda_scheduler_node, CUDASchedulerNode)
        assert (
            cuda_scheduler_node not in epilogue_scheduler_nodes
        ), "template node should not be in snodes"
        assert cuda_scheduler_node.is_template()
        super().__init__(scheduler, [cuda_scheduler_node] + epilogue_scheduler_nodes)
        self.template_node = cuda_scheduler_node

    def get_cuda_scheduler_node(self) -> CUDASchedulerNode:
        """
        Returns the CUDASchedulerNode that is the template for this FusedCUDASchedulerNode.
        """
        return self.template_node

    def get_epilogue_nodes(self) -> List[SchedulerNode]:
        """
        Returns the list of epilogue nodes that are fused with the template node.
        """
        return self.snodes[1:]

    def can_fuse_epilogue(self, other_node: SchedulerNode) -> bool:
        """
        Determines whether this FusedCUDASchedulerNode can fuse another node as epilogue.
        """
        if other_node.get_device() != self.get_device():
            return False
        return _can_fuse_epilogue(
            cast(CUDATemplateBuffer, self.get_cuda_scheduler_node().node),
            [n.node for n in self.snodes[1:]],
            other_node.node,
        )

    def fuse_epilogue(self, other_node: SchedulerNode) -> FusedSchedulerNode:
        """
        Fuses the current FusedCUDASchedulerNode with another (Epilogue) Scheduler Node
        and returns a new FusedCUDASchedulerNode
        """
        assert self.can_fuse_epilogue(other_node)
        return FusedCUDASchedulerNode(
            self.template_node, self.scheduler, self.snodes + [other_node]
        )


class CUDAScheduling(TritonScheduling):
    """
    Final codegen for CUDAKernels
    """

    def define_kernel(self, src_code: str, node_schedule) -> str:
        wrapper = V.graph.wrapper_code
        if src_code in wrapper.src_to_kernel:
            kernel_name = wrapper.src_to_kernel[src_code]
        else:
            fused_name = (
                get_fused_kernel_name(node_schedule, config.triton.descriptive_names)
                if config.triton.descriptive_names
                else ""
            )
            kernel_name = "_".join(["cuda", fused_name, wrapper.next_kernel_suffix()])
            # use the original src_code as the key
            wrapper.src_to_kernel[src_code] = kernel_name
            src_code = src_code.replace("KERNEL_NAME", kernel_name)

            _, _, kernel_path = get_path(code_hash(src_code), "py")

            compile_wrapper = IndentedBuffer()
            compile_wrapper.writeline("async_compile.cuda(r'''")
            compile_wrapper.splice(src_code, strip=True)
            compile_wrapper.writeline("''', 'so')")

            metadata_comment = f"# kernel path: {kernel_path}"
            origins, detailed_origins = get_kernel_metadata(node_schedule, wrapper)
            metadata_comment += "\n" + origins + "\n" + detailed_origins
            wrapper.define_kernel(
                kernel_name, compile_wrapper.getvalue(), metadata_comment
            )
        return kernel_name

    def codegen_template(
        self, template_node: CUDASchedulerNode, epilogue_nodes: List[BaseSchedulerNode]
    ):
        """
        Codegen a CUDA template, possibly with fused epilogues
        """
        assert isinstance(
            template_node, CUDASchedulerNode
        ), "Template node passed to CUDAScheduler.codegen_template must be a CUDASchedulerNode."
        _, (numel, rnumel) = template_node.group
        assert rnumel == 1
        ctb: CUDATemplateBuffer = template_node.node
        epilogue_ir_nodes: List[ir.Buffer] = [n.node for n in epilogue_nodes]
        assert all(
            isinstance(n, ir.ComputedBuffer) for n in epilogue_ir_nodes
        ), "Epilogue nodes must all be instances of ir.ComputedBuffer"
        kernel, render = ctb.make_kernel_render(ctb, epilogue_nodes=epilogue_ir_nodes)
        with kernel:
            for node in [template_node, *epilogue_nodes]:
                node.mark_run()
            src_code = render()

        with V.set_kernel_handler(kernel):
            node_schedule = [template_node, *epilogue_nodes]
            kernel_name = self.define_kernel(src_code, node_schedule)
        self.codegen_comment(node_schedule)
        kernel.call_kernel(kernel_name, ctb, epilogue_ir_nodes)
        V.graph.removed_buffers |= kernel.removed_buffers
        self.scheduler.free_buffers()
