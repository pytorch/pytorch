import logging

import torch
from torch._export.pass_base import _ExportPassBase

from torch.ao.quantization.pt2e.utils import (
    _filter_sym_size_users,
    _is_valid_annotation,
)

from torch.fx.node import map_arg
from torch.fx.passes.infra.pass_base import PassResult


logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

__all__ = ["DuplicateDQPass"]

_DEQUANTIZE_OPS = [
    torch.ops.quantized_decomposed.dequantize_per_tensor.default,
    torch.ops.quantized_decomposed.dequantize_per_tensor.tensor,
    torch.ops.quantized_decomposed.dequantize_per_channel.default,
]


"""
This API is specifically to replace a specific use of producer node
ATM, it uses internal API __update_args_kwargs, but ideally we mvoe
this API to fx.Node.replace_use_with...
"""


def _copy_and_replace_use(
    gm: torch.fx.GraphModule, producer: torch.fx.Node, user: torch.fx.Node
):
    with gm.graph.inserting_after(producer):
        new_node = gm.graph.node_copy(producer)

        def maybe_replace_node(n: torch.fx.Node) -> torch.fx.Node:
            if n == producer:
                return new_node
            else:
                return n

        new_args = map_arg(user.args, maybe_replace_node)
        new_kwargs = map_arg(user.kwargs, maybe_replace_node)
        user.args = new_args
        user.kwargs = new_kwargs


def _maybe_duplicate_dq(
    gm: torch.fx.GraphModule, producer: torch.fx.Node, user: torch.fx.Node
):
    annotation = user.meta.get("quantization_annotation", None)
    if not _is_valid_annotation(annotation):
        return
    _copy_and_replace_use(gm, producer, user)


class DuplicateDQPass(_ExportPassBase):
    """
    kk
    """

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        for node in graph_module.graph.nodes:
            if node.op == "call_function" and node.target in _DEQUANTIZE_OPS:
                dq_users = _filter_sym_size_users(node)
                if len(dq_users) <= 1:
                    continue
                for user in dq_users:
                    _maybe_duplicate_dq(graph_module, node, user)
        graph_module.graph.eliminate_dead_code()
        graph_module.recompile()
        return PassResult(graph_module, True)
