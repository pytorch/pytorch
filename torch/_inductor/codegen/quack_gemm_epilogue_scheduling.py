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

        if isinstance(template_node.node, ir.QuackSplitKTemplateBuffer):
            self._codegen_split_k_template(template_node.node)
        else:
            self._codegen_gemm_epilogue_template(
                cast(ir.QuackGemmEpilogueTemplateBuffer, template_node.node)
            )
        self.free_buffers_in_scheduler()

    def _codegen_split_k_template(self, qtb: ir.QuackSplitKTemplateBuffer) -> None:
        wrapper = V.graph.wrapper_code
        wrapper.add_import_once("from quack.gemm_interface import gemm as quack_gemm")
        input_args = [input.codegen_reference() for input in qtb.inputs]
        k_split = qtb.k_split
        wrapper.writeline(
            f"{qtb.get_name()} = quack_gemm("
            f"{input_args[0]}.reshape({input_args[0]}.shape[0], {k_split}, "
            f"{input_args[0]}.shape[1] // {k_split}).permute(1, 0, 2), "
            f"{input_args[1]}.reshape({k_split}, {input_args[0]}.shape[1] // {k_split}, "
            f"{input_args[1]}.shape[1]), out_dtype=torch.float32, tuned=False)"
        )

    def _codegen_gemm_epilogue_template(
        self, qtb: ir.QuackGemmEpilogueTemplateBuffer
    ) -> None:
        wrapper = V.graph.wrapper_code
        config = qtb.config
        wrapper.add_import_once(
            "from quack.gemm_epilogue_interface import gemm_epilogue"
        )
        if not hasattr(wrapper, "quack_gemm_epilogue_defs"):
            wrapper.quack_gemm_epilogue_defs = OrderedSet()
        if config.epilogue_name not in wrapper.quack_gemm_epilogue_defs:
            wrapper.header.splice(config.epilogue_source)
            wrapper.quack_gemm_epilogue_defs.add(config.epilogue_name)

        input_args = [input.codegen_reference() for input in qtb.inputs]
        call_args, call_kwargs = self._gemm_call_args(input_args, config)
        call_kwargs += self._epilogue_kwargs(input_args, config)
        call_kwargs += self._local_reduce_kwargs(input_args, config)
        call_kwargs += f", tuned={config.tuned!r}, epilogue_source={config.epilogue_source!r}"
        wrapper.writeline(
            f"{qtb.get_name()} = gemm_epilogue("
            f"{call_args[0]}, {call_args[1]}, "
            f"{config.epilogue_name}, {config.epilogue_name!r}{call_kwargs})"
        )

    def _epilogue_kwargs(
        self, input_args: list[str], config: ir.QuackGemmEpilogueConfig
    ) -> str:
        epilogue_args = [input_args[i] for i in config.epilogue_arg_indices]
        kwargs = ""
        if epilogue_args:
            kwargs = f", epilogue_args=({', '.join(epilogue_args)},)"
            kwargs += f", epilogue_arg_kinds={config.epilogue_arg_kinds!r}"
        if config.main_output_transform is not None:
            kwargs += f", main_output_transform={config.main_output_transform!r}"
            if config.main_output_transform_group is not None:
                kwargs += (
                    f", main_output_transform_group={config.main_output_transform_group!r}"
                )
        if config.concat_layout:
            kwargs += f", concat_layout={config.concat_layout!r}"
        return kwargs

    def _local_reduce_kwargs(
        self, input_args: list[str], config: ir.QuackGemmEpilogueConfig
    ) -> str:
        kwargs = ""
        if config.aux_out_index is not None:
            kwargs += f", aux_out={input_args[config.aux_out_index]}"
        if config.local_reduce_group is None:
            return kwargs

        kwargs += (
            f", local_reduce_group={config.local_reduce_group!r}, "
            f"local_reduce_dim={config.local_reduce_dim!r}"
        )
        if config.local_reduce_out_index is not None:
            kwargs += f", local_reduce_out={input_args[config.local_reduce_out_index]}"
        if config.local_reduce_op is not None:
            kwargs += f", local_reduce_op={config.local_reduce_op!r}"
        if config.local_reduce_scale != 1.0:
            kwargs += f", local_reduce_scale={config.local_reduce_scale!r}"
        if config.local_reduce_op == "mx_e8m0_scale" or config.local_reduce_max_power != 8:
            kwargs += f", local_reduce_max_power={config.local_reduce_max_power!r}"
        if config.local_reduce_feeds_main:
            kwargs += ", local_reduce_feeds_main=True"
        if config.local_reduce_source_from_epilogue:
            kwargs += ", local_reduce_source_from_epilogue=True"
        return kwargs

    def _gemm_call_args(
        self, input_args: list[str], config: ir.QuackGemmEpilogueConfig
    ) -> tuple[list[str], str]:
        if config.gemm_op in ("mm", "bmm"):
            out_dtype_kwargs = (
                "" if config.out_dtype is None else f", out_dtype={config.out_dtype!r}"
            )
            return [input_args[0], input_args[1]], out_dtype_kwargs
        if config.gemm_op == "scaled_mm":
            return [input_args[0], input_args[1]], (
                f", scale_a={input_args[2]}, scale_b={input_args[3]}, "
                f"out_dtype={config.out_dtype!r}"
            )
        if config.gemm_op == "grouped_mm":
            return [input_args[0], input_args[1]], (
                f", offs={input_args[2]}, out_dtype={config.out_dtype!r}"
            )

        out_dtype_kwargs = (
            "" if config.out_dtype is None else f", out_dtype={config.out_dtype!r}"
        )
        return [input_args[1], input_args[2]], (
            f", C={input_args[0]}, alpha={config.alpha!r}, beta={config.beta!r}"
            f"{out_dtype_kwargs}"
        )
