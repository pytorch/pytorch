# mypy: allow-untyped-defs
from collections.abc import Sequence
from typing import cast

from torch._inductor import ir
from torch.utils._ordered_set import OrderedSet

from ..scheduler import BaseSchedulerNode, BaseScheduling, SchedulerNode
from ..virtualized import V
from .common import BackendFeature


class QuackGemmEpilogueScheduling(BaseScheduling):
    @classmethod
    def get_backend_features(cls, device) -> OrderedSet[BackendFeature]:
        return OrderedSet()

    @staticmethod
    def is_quack_gemm_epilogue_template(node: BaseSchedulerNode) -> bool:
        return isinstance(node, SchedulerNode) and isinstance(
            node.node, ir.QuackGemmEpilogueTemplateBuffer
        )

    def can_fuse_vertical(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        return False

    def can_fuse_horizontal(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        return False

    def codegen_template(
        self,
        template_node: BaseSchedulerNode,
        epilogue_nodes: Sequence[BaseSchedulerNode],
        prologue_nodes: Sequence[BaseSchedulerNode],
    ) -> None:
        assert not epilogue_nodes
        assert not prologue_nodes
        assert self.is_quack_gemm_epilogue_template(template_node)

        template_node = cast(SchedulerNode, template_node)
        qtb = cast(ir.QuackGemmEpilogueTemplateBuffer, template_node.node)
        template_node.mark_run()

        wrapper = V.graph.wrapper_code
        wrapper.add_import_once(
            "from quack.gemm_epilogue_interface import gemm_epilogue"
        )
        if not hasattr(wrapper, "quack_gemm_epilogue_defs"):
            wrapper.quack_gemm_epilogue_defs = OrderedSet()
        if qtb.epilogue_name not in wrapper.quack_gemm_epilogue_defs:
            wrapper.header.splice(qtb.epilogue_source)
            wrapper.quack_gemm_epilogue_defs.add(qtb.epilogue_name)

        input_args = [input.codegen_reference() for input in qtb.inputs]
        wrapper.writeline(
            f"{qtb.get_name()} = gemm_epilogue("
            f"{input_args[0]}, {input_args[1]}, "
            f"{qtb.epilogue_name}, {qtb.epilogue_name!r})"
        )
        self.free_buffers_in_scheduler()
