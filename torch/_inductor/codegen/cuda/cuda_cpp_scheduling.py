import logging
from typing import cast, List, Set

from ...._dynamo.utils import counters

from ... import config, ir
from ...codecache import code_hash, CUDACodeCache, get_path

from ...exc import CUDACompileError
from ...ir import ComputedBuffer, CUDATemplateBuffer, Pointwise
from ...scheduler import (
    BaseSchedulerNode,
    BaseScheduling,
    FusedSchedulerNode,
    Scheduler,
    SchedulerNode,
)
from ...utils import get_fused_kernel_name, get_kernel_metadata, sympy_product
from ...virtualized import V
from ..common import IndentedBuffer

from .cutlass_epilogue_gen import CUTLASSEVTOpNotImplementedError

log = logging.getLogger(__name__)


class CUDACPPScheduling(BaseScheduling):
    """
    Partial Scheduling implementation for CUDA C++ Kernels.
    This class is intended to be used in combination with TritonScheduling,
    and delegated to by CUDACombinedScheduling.

    It handles fusion decisions and CUDA C++ specific template code generation.
    """

    def __init__(self, scheduler: Scheduler):
        super().__init__()
        self.scheduler = scheduler

    def group_fn(self, sizes):
        return tuple(V.graph.sizevars.simplify(sympy_product(s)) for s in sizes)

    def is_cuda_cpp_template(self, node: BaseSchedulerNode) -> bool:
        return isinstance(node, SchedulerNode) and isinstance(
            node.node, CUDATemplateBuffer
        )

    def is_cuda_cpp_fused_template(self, node: BaseSchedulerNode) -> bool:
        return isinstance(node, FusedSchedulerNode) and self.is_cuda_cpp_template(
            node.get_template_node()
        )

    def _can_fuse_epilogue_impl(
        self,
        cuda_template_buffer: CUDATemplateBuffer,
        epilogue_nodes: List[ir.IRNode],
        additional_node: ir.IRNode,
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
        if not isinstance(cuda_template_buffer, CUDATemplateBuffer):
            return False
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
        node_name = additional_node.get_computed_buffer_name()
        if node_name is None:
            return False
        if len(epilogue_nodes) == 0:
            if cuda_template_buffer.name not in additional_node.get_read_names():
                return False
        else:
            last_epilogue_node = epilogue_nodes[-1]
            assert isinstance(last_epilogue_node, ir.ComputedBuffer)  # for mypy
            last_epilogue_name = (
                last_epilogue_node.name
                if last_epilogue_node.name is not None
                else last_epilogue_node.data.name  # type: ignore[attr-defined]
            )
            if last_epilogue_name not in additional_node.get_read_names():
                return False

        template_buffer_names: Set[str] = cuda_template_buffer.get_read_names()
        fused_reading_buffer_names: Set[str] = set(template_buffer_names)

        for epilogue_node in epilogue_nodes:
            fused_reading_buffer_names.update(epilogue_node.get_read_names())

        # We need to remove all reads which were written as intermediate results
        fused_written_names = set()
        fused_written_names.add(cuda_template_buffer.get_name())
        for epilogue_node in epilogue_nodes:
            fused_written_names.add(epilogue_node.get_name())
        fused_reading_buffer_names -= fused_written_names

        # TODO: So far we only support 3 tensor arguments for the buffer. A, B, and C ( = Bias OR arbitrary arg )
        # Additional ( auxiliary ) EVT inputs would require non-zero workspace memory and a change of
        # the Kernel call signature.
        assert (
            len(fused_reading_buffer_names) <= 3
        ), "Only 3 tensor arguments are supported for the buffer."
        after_fuse_reading_buffers = (
            fused_reading_buffer_names.union(additional_node.get_read_names())
            - fused_written_names
        )
        if len(after_fuse_reading_buffers) > len(fused_reading_buffer_names):
            # Check that the layout of the additional input is compatible
            added_names = after_fuse_reading_buffers - fused_reading_buffer_names
            for added_name in added_names:
                added_node = V.graph.get_buffer(added_name)
                from torch._inductor.codegen.cuda.cuda_template import CUDATemplate

                template: CUDATemplate = cuda_template_buffer.template
                check_layouts = [n.layout for n in template.input_nodes[:2]] + [
                    added_node.layout
                ]
                if not template.are_inputs_layout_compatible(
                    [n.layout for n in template.input_nodes[:2]] + [added_node.layout]
                ):
                    log.warning(
                        f"Cannot fuse epilogue node {additional_node} into {cuda_template_buffer.name}, since the layouts (A,B,C)={check_layouts} are not compatible"
                    )
                    return False
        try:
            from torch._inductor.codegen.cuda.cutlass_epilogue_gen import (
                CutlassEVTEpilogueArgumentFormatter,
                CutlassEVTEpilogueTypeFormatter,
            )

            CutlassEVTEpilogueTypeFormatter.ir_to_evt_string(
                cast(str, cuda_template_buffer.name),
                "anything",
                [additional_node],
                gemm_output_layout=cuda_template_buffer.layout,
            )
            CutlassEVTEpilogueArgumentFormatter.ir_to_evt_argument_string(
                cast(str, cuda_template_buffer.name),
                [additional_node],
                dry_run=True,
                gemm_output_layout=cuda_template_buffer.layout,
            )
        except CUTLASSEVTOpNotImplementedError as e:
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
        compilation_result = self.try_fused_template_compilation(
            cuda_template_buffer, epilogue_nodes + [additional_node]
        )
        if not compilation_result:
            log.warning(
                f"Cannot fuse epilogue node {additional_node} into {cuda_template_buffer.name}, due to compilation failure, this most likely means that the fused kernel would require too much shared memory."
            )
            return False
        try:
            # If retuning is enabled, let's try to run the Kernel
            cuda_template_buffer.retune(epilogue_nodes + [additional_node])
        except NoValidChoicesError:
            log.warning(
                f"Cannot fuse epilogue node {additional_node} into {cuda_template_buffer.name}, retuning did not return any viable kernel choices. This can indicate that the shared memory requirement would be too high."
            )
            return False
        return True

    @staticmethod
    def _unwrap_epilogue_nodes(fused_node: FusedSchedulerNode) -> List[ir.IRNode]:
        nodes = fused_node.get_nodes()
        template_node = fused_node.get_template_node()
        nodes.remove(template_node)
        return [n.node for n in nodes]

    def can_fuse_vertical(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        if self.is_cuda_cpp_template(node1) and isinstance(node2, SchedulerNode):
            return self._can_fuse_epilogue_impl(
                cast(CUDATemplateBuffer, node1.node), [], node2.node
            )
        elif self.is_cuda_cpp_fused_template(node1) and isinstance(
            node2, SchedulerNode
        ):
            fnode1 = cast(FusedSchedulerNode, node1)
            return self._can_fuse_epilogue_impl(
                fnode1.get_template_node().node,
                self._unwrap_epilogue_nodes(fnode1),
                node2.node,
            )
        return False

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

    def try_fused_template_compilation(self, ctb, epilogue_ir_nodes) -> bool:
        # Try codegen and see if we can compile the generated source
        # this is the only reliable way to detect whether we would use too much
        # shared memory for the fused kernel
        if not all(isinstance(n, ir.ComputedBuffer) for n in epilogue_ir_nodes):
            return False
        kernel, render = ctb.make_kernel_render(ctb, epilogue_nodes=epilogue_ir_nodes)
        with kernel:
            src_code = render()
        try:
            CUDACodeCache.compile(src_code, "so")
        except CUDACompileError as e:
            log.debug(e, exc_info=False)
            return False
        return True

    def codegen_template(
        self, template_node: BaseSchedulerNode, epilogue_nodes: List[SchedulerNode]
    ):
        """
        Codegen a CUDA template, possibly with fused epilogues
        """
        counters["inductor"]["cuda_epilogue_fusion_counter"] += len(epilogue_nodes)
        assert self.is_cuda_cpp_template(
            template_node
        ), "Template node passed to CUDAScheduler.codegen_template must be a SchedulerNode that wraps a CUDATemplateBuffer"
        template_node = cast(SchedulerNode, template_node)
        _, (numel, rnumel) = template_node.group
        assert rnumel == 1
        ctb: CUDATemplateBuffer = cast(CUDATemplateBuffer, template_node.node)
        epilogue_ir_nodes: List[ir.Buffer] = [n.node for n in epilogue_nodes]
        assert all(
            isinstance(n, ir.ComputedBuffer) for n in epilogue_ir_nodes
        ), "Epilogue nodes must all be instances of ir.ComputedBuffer"
        ctb.retune(
            epilogue_ir_nodes
        )  # Retune for the epilogue nodes, if enabled ( cached )
        kernel, render = ctb.make_kernel_render(ctb, epilogue_nodes=epilogue_ir_nodes)
        with kernel:
            for node in [template_node, *epilogue_nodes]:
                node.mark_run()
            src_code = render()

        with V.set_kernel_handler(kernel):
            node_schedule = [template_node, *epilogue_nodes]
            kernel_name = self.define_kernel(src_code, node_schedule)
        kernel.call_kernel(kernel_name, ctb, epilogue_ir_nodes)
        V.graph.removed_buffers |= kernel.removed_buffers
        self.scheduler.free_buffers()
