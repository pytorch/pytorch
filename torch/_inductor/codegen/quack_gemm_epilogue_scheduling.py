# mypy: allow-untyped-defs
from collections.abc import Sequence
from typing import Any, cast

from torch._inductor import ir
from torch.utils._ordered_set import OrderedSet

from ..scheduler import BaseSchedulerNode, BaseScheduling, SchedulerNode
from ..virtualized import V
from .common import BackendFeature


class QuackGemmEpilogueScheduling(BaseScheduling):
    """Schedules generated QuACK split-K template calls."""

    @classmethod
    def get_backend_features(cls, device) -> OrderedSet[BackendFeature]:
        return OrderedSet()

    @staticmethod
    def is_quack_gemm_epilogue_template(node: BaseSchedulerNode) -> bool:
        return isinstance(node, SchedulerNode) and isinstance(
            node.node, ir.QuackSplitKTemplateBuffer
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
        template_node.mark_run()
        self._codegen_split_k_template(
            cast(ir.QuackSplitKTemplateBuffer, template_node.node)
        )
        self.free_buffers_in_scheduler()

    def _codegen_split_k_template(self, qtb: ir.QuackSplitKTemplateBuffer) -> None:
        wrapper = V.graph.wrapper_code
        wrapper.add_import_once("from quack.gemm_interface import gemm as quack_gemm")
        input_args = [cast(Any, input).codegen_reference() for input in qtb.inputs]
        k_split = qtb.k_split
        wrapper.writeline(
            f"{qtb.get_name()} = quack_gemm("
            f"{input_args[0]}.reshape({input_args[0]}.shape[0], {k_split}, "
            f"{input_args[0]}.shape[1] // {k_split}).permute(1, 0, 2), "
            f"{input_args[1]}.reshape({k_split}, {input_args[0]}.shape[1] // {k_split}, "
            f"{input_args[1]}.shape[1]), out_dtype=torch.float32, tuned=False)"
        )
