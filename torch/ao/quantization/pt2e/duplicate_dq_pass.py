import logging
import operator

import torch
from torch.ao.quantization.pt2e.utils import (
    _filter_sym_size_users,
    _is_valid_annotation,
)
from torch.fx.node import map_arg
from torch.fx.passes.infra.pass_base import PassBase, PassResult


logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

__all__ = ["DuplicateDQPass"]

_QUANTIZE_OPS = [
    torch.ops.quantized_decomposed.quantize_per_tensor.default,
    torch.ops.quantized_decomposed.quantize_per_tensor.tensor,
    torch.ops.quantized_decomposed.quantize_per_channel.default,
]

_DEQUANTIZE_OPS = [
    torch.ops.quantized_decomposed.dequantize_per_tensor.default,
    torch.ops.quantized_decomposed.dequantize_per_tensor.tensor,
    torch.ops.quantized_decomposed.dequantize_per_channel.default,
]


def _maybe_duplicate_dq(
    gm: torch.fx.GraphModule, dq_node: torch.fx.Node, user: torch.fx.Node
) -> None:
    annotation = user.meta.get("quantization_annotation", None)
    if not _is_valid_annotation(annotation):  # type: ignore[arg-type]
        return
    with gm.graph.inserting_after(dq_node):
        new_node = gm.graph.node_copy(dq_node)

        def maybe_replace_node(n: torch.fx.Node) -> torch.fx.Node:
            if n == dq_node:
                return new_node
            else:
                return n

        new_args = map_arg(user.args, maybe_replace_node)
        new_kwargs = map_arg(user.kwargs, maybe_replace_node)
        user.args = new_args  # type: ignore[assignment]
        user.kwargs = new_kwargs  # type: ignore[assignment]


class DuplicateDQPass(PassBase):
    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        for node in graph_module.graph.nodes:
            if node.op == "call_function" and node.target in _DEQUANTIZE_OPS:
                dq_users = _filter_sym_size_users(node)
                if len(dq_users) <= 1:
                    continue
                # Do not duplicate dq for dynamic quantization
                # Pattern: choose_qparam - getitem - q - dq
                q_node = node.args[0]
                if q_node.op == "call_function" and q_node.target in _QUANTIZE_OPS:
                    getitem_node = q_node.args[1]
                    if (
                        isinstance(getitem_node, torch.fx.node.Node)
                        and getitem_node.op == "call_function"
                        and getitem_node.target is operator.getitem
                    ):
                        choose_qparam_node = getitem_node.args[0]
                        if (
                            isinstance(choose_qparam_node, torch.fx.node.Node)
                            and choose_qparam_node.op == "call_function"
                            and choose_qparam_node.target
                            == torch.ops.quantized_decomposed.choose_qparams.tensor
                        ):
                            continue
                for user in dq_users:
                    _maybe_duplicate_dq(graph_module, node, user)
        graph_module.graph.eliminate_dead_code()
        graph_module.recompile()
        return PassResult(graph_module, True)
