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

__all__ = ["DuplicateDynamicQuantChainPass"]

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


def _replace_input_node_with_new_node(node, input_node, new_node):
    def maybe_replace_node(n: torch.fx.Node) -> torch.fx.Node:
        if n == input_node:
            return new_node
        else:
            return n

    new_args = map_arg(node.args, maybe_replace_node)
    new_kwargs = map_arg(node.kwargs, maybe_replace_node)
    node.args = new_args
    node.kwargs = new_kwargs


def _replicate_chose_qparam_nodes_for_q_dq(
    gm: torch.fx.GraphModule, chose_qparams_node, get_item_node_1, get_item_node_2
):
    if (
        (
            chose_qparams_node.target
            != torch.ops.quantized_decomposed.choose_qparams.tensor
        )
        or (get_item_node_1.target != operator.getitem)
        or (get_item_node_2.target != operator.getitem)
    ):
        raise RuntimeError(
            f"Expecting choose_qparams.tensor and getitem nodes but got {chose_qparams_node}, {get_item_node_1}, {get_item_node_2}"
        )

    users = list(get_item_node_1.users.copy())
    q_dq_pair = []
    for user in users:
        if user.target in _QUANTIZE_OPS:
            if len(user.users) != 1:
                raise RuntimeError(f"Node {user} has more than one user")
            dq_node = list(user.users)[0]
            if dq_node.target not in _DEQUANTIZE_OPS:
                raise RuntimeError(
                    f"Node {user}'s use must be a dequantize op but got {dq_node}:{dq_node.target}"
                )
            q_dq_pair.append((user, dq_node))

    for (q_node, dq_node) in q_dq_pair:
        with gm.graph.inserting_after(get_item_node_1):
            new_get_item_node_1 = gm.graph.node_copy(get_item_node_1)
            new_get_item_node_2 = gm.graph.node_copy(get_item_node_2)
            new_chose_qparams_node = gm.graph.node_copy(chose_qparams_node)
            _replace_input_node_with_new_node(
                new_get_item_node_1, chose_qparams_node, new_chose_qparams_node
            )
            _replace_input_node_with_new_node(
                new_get_item_node_2, chose_qparams_node, new_chose_qparams_node
            )

            _replace_input_node_with_new_node(
                q_node, get_item_node_1, new_get_item_node_1
            )
            _replace_input_node_with_new_node(
                dq_node, get_item_node_1, new_get_item_node_1
            )
            _replace_input_node_with_new_node(
                q_node, get_item_node_2, new_get_item_node_2
            )
            _replace_input_node_with_new_node(
                dq_node, get_item_node_2, new_get_item_node_2
            )

    gm.graph.eliminate_dead_code()
    gm.recompile()


def _replicate_node_for_each_user(gm: torch.fx.GraphModule, node: torch.fx.Node):
    users = list(node.users.copy())
    for user in users:
        with gm.graph.inserting_after(node):
            new_node = gm.graph.node_copy(node)
            _replace_input_node_with_new_node(user, node, new_node)

    gm.graph.eliminate_dead_code()
    gm.recompile()


def _maybe_duplicate_dynamic_quantize_chain(
    gm: torch.fx.GraphModule,
    chose_qparams_node,
    get_item_node_1,
    get_item_node_2,
    q_node,
    dq_node: torch.fx.Node,
):
    num_dq_users = len(dq_node.users)
    dq_node_users = list(dq_node.users.copy())
    for user in dq_node_users:
        annotation = user.meta.get("quantization_annotation", None)
        if not _is_valid_annotation(annotation):
            return
        with gm.graph.inserting_after(dq_node):
            new_node = gm.graph.node_copy(dq_node)
            _replace_input_node_with_new_node(user, dq_node, new_node)

    gm.graph.eliminate_dead_code()
    gm.recompile()
    if len(q_node.users) != num_dq_users:
        raise RuntimeError(
            f"Expected {num_dq_users} users of {q_node}, but got {len(q_node.users)}"
        )
    _replicate_node_for_each_user(gm, q_node)

    # *2 because scale/zp are used both with q and dq nodes
    if len(get_item_node_1.users) != num_dq_users * 2:
        raise RuntimeError(
            f"Expected {num_dq_users} users of {get_item_node_1}, but got {len(get_item_node_1.users)}"
        )
    if len(get_item_node_2.users) != num_dq_users * 2:
        raise RuntimeError(
            f"Expected {num_dq_users} users of {get_item_node_2}, but got {len(get_item_node_2.users)}"
        )
    _replicate_chose_qparam_nodes_for_q_dq(
        gm, chose_qparams_node, get_item_node_1, get_item_node_2
    )


class DuplicateDynamicQuantChainPass(PassBase):
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
                    getitem_1_node = q_node.args[1]
                    getitem_2_node = q_node.args[2]
                    if (
                        isinstance(getitem_1_node, torch.fx.node.Node)
                        and getitem_1_node.op == "call_function"
                        and getitem_1_node.target == operator.getitem
                    ):
                        choose_qparam_node = getitem_1_node.args[0]
                        if (
                            isinstance(choose_qparam_node, torch.fx.node.Node)
                            and choose_qparam_node.op == "call_function"
                            and choose_qparam_node.target
                            == torch.ops.quantized_decomposed.choose_qparams.tensor
                        ):
                            _maybe_duplicate_dynamic_quantize_chain(
                                graph_module,
                                choose_qparam_node,
                                getitem_1_node,
                                getitem_2_node,
                                q_node,
                                node,
                            )
                            continue
        graph_module.graph.eliminate_dead_code()
        graph_module.recompile()
        return PassResult(graph_module, True)
