# mypy: allow-untyped-defs
import logging
from collections.abc import Sequence
from typing import cast

from torch.utils._ordered_set import OrderedSet

from ...._dynamo.utils import counters
from ... import config
from ...codecache import code_hash, get_path
from ...ir import Buffer, ComputedBuffer, CUDATemplateBuffer, IRNode, Pointwise
from ...scheduler import (
    BaseSchedulerNode,
    BaseScheduling,
    FusedSchedulerNode,
    SchedulerNode,
)
from ...utils import get_fused_kernel_name, get_kernel_metadata, sympy_product
from ...virtualized import V
from ..common import BackendFeature, IndentedBuffer


log = logging.getLogger(__name__)


class CUDACPPScheduling(BaseScheduling):
    """
    Partial Scheduling implementation for CUDA C++ Kernels.
    This class is intended to be used in combination with TritonScheduling,
    and delegated to by CUDACombinedScheduling.

    It handles fusion decisions and CUDA C++ specific template code generation.
    """

    @classmethod
    def get_backend_features(cls, device) -> OrderedSet[BackendFeature]:
        return OrderedSet()

    def group_fn(self, sizes):
        return tuple(V.graph.sizevars.simplify(sympy_product(s)) for s in sizes)

    @staticmethod
    def is_cuda_cpp_template(node: BaseSchedulerNode) -> bool:
        return isinstance(node, SchedulerNode) and isinstance(
            node.node, CUDATemplateBuffer
        )

    def is_cuda_cpp_fused_template(self, node: BaseSchedulerNode) -> bool:
        return isinstance(node, FusedSchedulerNode) and self.is_cuda_cpp_template(node)

    def can_fuse_vertical(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        if self.is_cuda_cpp_template(node1) and isinstance(node2, SchedulerNode):
            assert node1.node, "node1.node should not be None"
            assert node2.node, "node2.node should not be None"
            return self._can_fuse_epilogue_impl(
                cast(CUDATemplateBuffer, node1.node),
                [],
                node2.node,  # type: ignore[arg-type]
            )
        elif self.is_cuda_cpp_fused_template(node1) and isinstance(
            node2, SchedulerNode
        ):
            assert node1.node, "node1.node should not be None"
            assert node2.node, "node2.node should not be None"
            fnode1 = cast(FusedSchedulerNode, node1)
            return self._can_fuse_epilogue_impl(
                fnode1.get_template_node(),  # type: ignore[arg-type]
                self._unwrap_epilogue_nodes(fnode1),
                node2.node,  # type: ignore[arg-type]
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
            compile_wrapper.writeline(
                f"''', 'so', aot_compile={str(V.graph.aot_mode)})"
            )

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
        prologue_nodes: Sequence[BaseSchedulerNode],
    ):
        """
        Codegen a CUDA template, possibly with fused epilogues
        """
        counters["inductor"]["cuda_epilogue_fusion_counter"] += len(epilogue_nodes)
        assert self.is_cuda_cpp_template(template_node), (
            "Template node passed to CUDAScheduler.codegen_template must be a SchedulerNode that wraps a CUDATemplateBuffer"
        )
        template_node = cast(SchedulerNode, template_node)
        _, (_numel, rnumel) = template_node.group
        assert rnumel == 1
        ctb: CUDATemplateBuffer = cast(CUDATemplateBuffer, template_node.node)
        epilogue_ir_nodes: list[Buffer] = [n.node for n in epilogue_nodes]  # type: ignore[misc]
        assert all(isinstance(n, ComputedBuffer) for n in epilogue_ir_nodes), (
            "Epilogue nodes must all be instances of ir.ComputedBuffer"
        )
        kernel, render = ctb.make_kernel_render(ctb, epilogue_nodes=epilogue_ir_nodes)

        with kernel:
            for node in [template_node, *epilogue_nodes]:
                node.mark_run()
            src_code = render()

        with V.set_kernel_handler(kernel):
            node_schedule = [template_node, *epilogue_nodes]
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
        self.free_buffers_in_scheduler()

    @staticmethod
    def _unwrap_epilogue_nodes(fused_node: FusedSchedulerNode) -> list[IRNode]:
        nodes = list(fused_node.get_nodes())
        template_node = fused_node.get_template_node()
        assert all(n.node is not None for n in nodes), (
            "All epilogue nodes should have an IRNode"
        )
        return cast(
            list[IRNode], [n.node for n in nodes if n.node is not template_node]
        )

    def _can_fuse_epilogue_impl(
        self,
        cuda_template_buffer: CUDATemplateBuffer,
        epilogue_nodes: list[IRNode],
        additional_node: IRNode,
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
        # if not cuda_template_buffer.template.can_fuse_epilogue:
        #    # The used GEMM op does not support fusing epilogues
        #    return False
        if not isinstance(additional_node, ComputedBuffer):
            return False
        if not isinstance(additional_node.data, Pointwise):
            return False
        # We can fuse a Pointwise op that depends on the last fused epilogue node
        # if any. If there is no epilogue node yet, it needs to depend on the template
        # node
        node_name = additional_node.get_computed_buffer_name()  # type: ignore[attr-defined]
        if node_name is None:
            return False

        if len(epilogue_nodes) == 0:
            if cuda_template_buffer.name not in additional_node.get_read_names():
                return False
        else:
            last_epilogue_node = epilogue_nodes[-1]
            assert isinstance(last_epilogue_node, ComputedBuffer)  # for mypy
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
            from torch._inductor.codegen.cuda.cutlass_python_evt import (
                CutlassEVTCodegen,
            )

            CutlassEVTCodegen.ir_to_evt_python_code(
                cast(str, cuda_template_buffer.name), epilogue_nodes + [additional_node]
            )

        except NotImplementedError as e:
            not_implemented_op = str(e)
            if not_implemented_op.startswith("_op_"):
                not_implemented_op = not_implemented_op[4:]
                log.warning(
                    f"Cannot fuse epilogue node {additional_node} into {cuda_template_buffer.name}, likely due to unsupported operation: {not_implemented_op}"  # noqa: G004, B950
                )
                return False
            else:  # Likely due to unsupported dtype.
                log.warning(
                    f"Cannot fuse epilogue node {additional_node} into {cuda_template_buffer.name}. Reason: {not_implemented_op}"  # noqa: G004, B950
                )
                return False

        return True
