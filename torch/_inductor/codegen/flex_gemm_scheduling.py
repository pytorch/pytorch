# mypy: allow-untyped-defs
from collections.abc import Sequence
from typing import Any, cast

from torch._inductor import ir
from torch.utils._ordered_set import OrderedSet

from ..scheduler import BaseSchedulerNode, BaseScheduling, SchedulerNode
from ..virtualized import V
from .common import BackendFeature


class FlexGemmScheduling(BaseScheduling):
    @classmethod
    def get_backend_features(cls, device) -> OrderedSet[BackendFeature]:
        return OrderedSet()

    @staticmethod
    def is_flex_gemm_template(node: BaseSchedulerNode) -> bool:
        return isinstance(node, SchedulerNode) and isinstance(
            node.node, ir.FlexGemmEpilogueTemplateBuffer
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
        assert self.is_flex_gemm_template(template_node)

        template_node = cast(SchedulerNode, template_node)
        template_buffer = cast(ir.FlexGemmEpilogueTemplateBuffer, template_node.node)
        template_node.mark_run()
        self._codegen_template(template_buffer)
        self.free_buffers_in_scheduler()

    def _codegen_template(
        self, template_buffer: ir.FlexGemmEpilogueTemplateBuffer
    ) -> None:
        wrapper = V.graph.wrapper_code
        wrapper.add_import_once(
            "from torch._inductor.kernel.flex_gemm.runtime import gemm_epilogue as flex_gemm_epilogue"
        )
        config = template_buffer.config
        wrapper.header.splice(config.epilogue_source)
        call_args, call_kwargs = self._gemm_call_args(
            [cast(Any, input).codegen_reference() for input in template_buffer.inputs],
            config,
        )
        call_kwargs += f", tuned=False, epilogue_source={config.epilogue_source!r}"
        wrapper.writeline(
            f"{template_buffer.get_name()} = flex_gemm_epilogue("
            f"{', '.join(call_args)}, {config.epilogue_name}, "
            f"{config.epilogue_name!r}{call_kwargs})"
        )

    def _gemm_call_args(
        self, input_args: list[str], config: ir.FlexGemmEpilogueConfig
    ) -> tuple[list[str], str]:
        out_dtype = (
            "" if config.out_dtype is None else f", out_dtype={config.out_dtype!r}"
        )
        if config.gemm_op == "mm":
            return [input_args[0], input_args[1]], out_dtype
        if config.gemm_op == "addmm":
            return [input_args[1], input_args[2]], (
                f", C={input_args[0]}, alpha={config.alpha!r}, beta={config.beta!r}"
                f"{out_dtype}"
            )
        raise NotImplementedError(f"unsupported FlexGEMM op: {config.gemm_op}")
