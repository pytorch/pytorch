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
            node.node,
            (ir.QuackGemmEpilogueTemplateBuffer, ir.QuackSplitKTemplateBuffer),
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

        wrapper = V.graph.wrapper_code
        if isinstance(template_node.node, ir.QuackSplitKTemplateBuffer):
            qtb = cast(ir.QuackSplitKTemplateBuffer, template_node.node)
            wrapper.add_import_once(
                "from quack.gemm_interface import gemm as quack_gemm"
            )
            input_args = [input.codegen_reference() for input in qtb.inputs]
            k_split = qtb.k_split
            wrapper.writeline(
                f"{qtb.get_name()} = quack_gemm("
                f"{input_args[0]}.reshape({input_args[0]}.shape[0], {k_split}, "
                f"{input_args[0]}.shape[1] // {k_split}).permute(1, 0, 2), "
                f"{input_args[1]}.reshape({k_split}, {input_args[0]}.shape[1] // {k_split}, "
                f"{input_args[1]}.shape[1]), out_dtype=torch.float32, tuned=False)"
            )
            self.free_buffers_in_scheduler()
            return

        qtb = cast(ir.QuackGemmEpilogueTemplateBuffer, template_node.node)
        wrapper.add_import_once(
            "from quack.gemm_epilogue_interface import gemm_epilogue"
        )
        if not hasattr(wrapper, "quack_gemm_epilogue_defs"):
            wrapper.quack_gemm_epilogue_defs = OrderedSet()
        if qtb.epilogue_name not in wrapper.quack_gemm_epilogue_defs:
            wrapper.header.splice(qtb.epilogue_source)
            wrapper.quack_gemm_epilogue_defs.add(qtb.epilogue_name)

        input_args = [input.codegen_reference() for input in qtb.inputs]
        if qtb.gemm_op in ("mm", "bmm"):
            call_args = [input_args[0], input_args[1]]
            call_kwargs = ""
        elif qtb.gemm_op == "scaled_mm":
            call_args = [input_args[0], input_args[1]]
            call_kwargs = (
                f", scale_a={input_args[2]}, scale_b={input_args[3]}, "
                f"out_dtype={qtb.out_dtype!r}"
            )
        elif qtb.gemm_op == "grouped_mm":
            call_args = [input_args[0], input_args[1]]
            call_kwargs = f", offs={input_args[2]}, out_dtype={qtb.out_dtype!r}"
        else:
            call_args = [input_args[1], input_args[2]]
            call_kwargs = f", C={input_args[0]}, alpha={qtb.alpha!r}, beta={qtb.beta!r}"
        wrapper.writeline(
            f"{qtb.get_name()} = gemm_epilogue("
            f"{call_args[0]}, {call_args[1]}, "
            f"{qtb.epilogue_name}, {qtb.epilogue_name!r}{call_kwargs})"
        )
        self.free_buffers_in_scheduler()
