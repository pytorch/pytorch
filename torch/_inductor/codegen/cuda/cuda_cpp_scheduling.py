# mypy: allow-untyped-defs
import logging
from typing import cast, Sequence

from ...._dynamo.utils import counters
from ... import config
from ...codecache import code_hash, get_path
from ...ir import CUDATemplateBuffer
from ...scheduler import BaseSchedulerNode, BaseScheduling, Scheduler, SchedulerNode
from ...utils import get_fused_kernel_name, get_kernel_metadata, sympy_product
from ...virtualized import V
from ..common import IndentedBuffer


log = logging.getLogger(__name__)


class CUDACPPScheduling(BaseScheduling):
    """
    Partial Scheduling implementation for CUDA C++ Kernels.
    This class is intended to be used in combination with TritonScheduling,
    and delegated to by CUDACombinedScheduling.

    It handles fusion decisions and CUDA C++ specific template code generation.
    """

    def __init__(self, scheduler: Scheduler) -> None:
        super().__init__()
        self.scheduler = scheduler

    @classmethod
    def get_backend_features(cls, device):
        return {}

    def group_fn(self, sizes):
        return tuple(V.graph.sizevars.simplify(sympy_product(s)) for s in sizes)

    @staticmethod
    def is_cuda_cpp_template(node: BaseSchedulerNode) -> bool:
        return isinstance(node, SchedulerNode) and isinstance(
            node.node, CUDATemplateBuffer
        )

    def can_fuse_vertical(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
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

    def codegen_template(
        self,
        template_node: BaseSchedulerNode,
        epilogue_nodes: Sequence[BaseSchedulerNode],
    ):
        """
        Codegen a CUDA template, possibly with fused epilogues
        """
        counters["inductor"]["cuda_epilogue_fusion_counter"] += len(epilogue_nodes)
        assert self.is_cuda_cpp_template(
            template_node
        ), "Template node passed to CUDAScheduler.codegen_template must be a SchedulerNode that wraps a CUDATemplateBuffer"
        template_node = cast(SchedulerNode, template_node)
        _, (_, rnumel) = template_node.group
        assert rnumel == 1
        ctb: CUDATemplateBuffer = cast(CUDATemplateBuffer, template_node.node)
        kernel, render = ctb.make_kernel_render(ctb)
        with kernel:
            template_node.mark_run()
            src_code = render()

        with V.set_kernel_handler(kernel):
            node_schedule = [template_node]
            kernel_name = self.define_kernel(src_code, node_schedule)

        # debug printing values of intermediate tensors
        _, call_args, arg_signatures, _ = kernel.args.python_argdefs()
        debug_printer_manager = V.graph.wrapper_code.debug_printer
        debug_printer_manager.set_printer_args(
            call_args, kernel_name, arg_signatures, kernel
        )
        with debug_printer_manager:
            kernel.call_kernel(kernel_name, ctb)

        V.graph.removed_buffers |= kernel.removed_buffers
        self.scheduler.free_buffers()
