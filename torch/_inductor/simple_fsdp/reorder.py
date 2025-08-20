# mypy: ignore-errors
from collections import defaultdict
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils._ordered_set import OrderedSet

from .. import config, ir, scheduler
from ..utils import is_collective


class NodeType(IntEnum):
    ALL_GATHER = 0
    COMPUTE = 1
    REDUCE_SCATTER = 2
    AG_WAIT = 3
    RS_WAIT = 4


def compute_node_users(
    snodes: List["scheduler.BaseSchedulerNode"],
) -> Tuple[
    Dict["scheduler.BaseSchedulerNode", OrderedSet["scheduler.BaseSchedulerNode"]],
    Dict["scheduler.BaseSchedulerNode", OrderedSet["scheduler.BaseSchedulerNode"]],
]:
    """
    Compute the inverse users and users of each node
    """
    buf_to_snode: Dict[str, scheduler.BaseSchedulerNode] = {}
    for node in snodes:
        if isinstance(node, scheduler.FusedSchedulerNode):
            for x in node.snodes:
                for buf in x.get_outputs():
                    buf_to_snode[buf.get_name()] = node

        for buf in node.get_outputs():
            buf_to_snode[buf.get_name()] = node

    inverse_users = {}
    keys = list(buf_to_snode.keys())
    for node in snodes:
        dep_list = []
        for dep in node.unmet_dependencies:
            if dep.name in keys:
                dep_list.append(buf_to_snode[dep.name])
        inverse_users.update({node: OrderedSet(dep_list)})

    node_users: Dict[
        scheduler.BaseSchedulerNode, OrderedSet[scheduler.BaseSchedulerNode]
    ] = defaultdict(OrderedSet)
    for node, node_inverse_users in inverse_users.items():
        for inverse_user in node_inverse_users:
            node_users[inverse_user].add(node)

    return inverse_users, node_users


def _check_ir_node_fsdp(ir_node: "ir.Operation") -> bool:
    """
    Determine if the AG/RS node is for FSDP or TP
    """
    if config.simplefsdp.simplefsdp_only:
        return True

    is_fsdp = False
    ir_node_origins = list(getattr(ir_node, "origins", None))

    if len(ir_node_origins) == 0:
        # bucketed AG and RS doesn't have origins, but they are created by FSDP
        is_fsdp = True

    for n in ir_node_origins:
        meta_data = n.meta.get("stack_trace", {})
        # TODO(ruisizhang123): hack to get FSDP node (the FSDP AG/RS are created from torch_spmd)
        if "parametrization" in meta_data:
            is_fsdp = True
    return is_fsdp


def _get_ir_node_type(ir_node: "ir.Operation") -> NodeType:
    """
    Determine the type of a ir node
    """
    if isinstance(ir_node, ir._WaitKernel):
        # Determine if the wait node is waiting for ALL_GATHER or REDUCE_SCATTER
        ir_op_overload = getattr(ir_node.inputs[0], "op_overload", None)
        if (
            ir_op_overload == torch.ops._c10d_functional.all_gather_into_tensor.default
            and _check_ir_node_fsdp(ir_node.inputs[0])
        ):
            return NodeType.AG_WAIT
        elif (
            ir_op_overload == torch.ops._c10d_functional.reduce_scatter_tensor.default
            and _check_ir_node_fsdp(ir_node.inputs[0])
        ):
            return NodeType.RS_WAIT
    if isinstance(ir_node, ir._CollectiveKernel):
        # Determine if the collective kernel is for ALL_GATHER or REDUCE_SCATTER
        ir_op_overload = getattr(ir_node, "op_overload", None)
        if is_collective(
            ir_node, op=torch.ops._c10d_functional.all_gather_into_tensor.default
        ) and _check_ir_node_fsdp(ir_node):
            return NodeType.ALL_GATHER
        elif is_collective(
            ir_node, op=torch.ops._c10d_functional.reduce_scatter_tensor.default
        ) and _check_ir_node_fsdp(ir_node):
            return NodeType.REDUCE_SCATTER

    if isinstance(ir_node, ir.FallbackKernel):
        python_kernel_name = ir_node.python_kernel_name
        if (
            python_kernel_name == "torch.ops._c10d_functional.wait_tensor.default"
            and ir_node.inputs[0].inputs[0].python_kernel_name
            == "torch.ops._c10d_functional.reduce_scatter_tensor.default"
        ):
            return NodeType.RS_WAIT
        elif (
            python_kernel_name == "torch.ops._c10d_functional.wait_tensor.default"
            and ir_node.inputs[0].inputs[0].python_kernel_name
            == "torch.ops._c10d_functional.all_gather_into_tensor_out.default"
        ):
            return NodeType.AG_WAIT
        elif (
            python_kernel_name
            == "torch.ops._c10d_functional.reduce_scatter_tensor.default"
        ):
            return NodeType.REDUCE_SCATTER
        elif (
            python_kernel_name
            == "torch.ops._c10d_functional.all_gather_into_tensor_out.default"
        ):
            return NodeType.ALL_GATHER
    return NodeType.COMPUTE


def get_node_type(node: "scheduler.BaseSchedulerNode") -> NodeType:
    """
    Determine the NodeType of a node
    """
    if isinstance(node, scheduler.FusedSchedulerNode):
        # Only compute nodes are fused
        return NodeType.COMPUTE

    if isinstance(node, scheduler.GroupedSchedulerNode):
        # [Only for bucketing]: newly created AG and RS are grouped as GroupedSchedulerNode
        child_nodes_type = [
            _get_ir_node_type(n) for n in [node.snodes[0].node, node.snodes[-2].node]
        ]
        if child_nodes_type[0] in [NodeType.AG_WAIT, NodeType.RS_WAIT]:
            return child_nodes_type[0]
        elif child_nodes_type[1] in [NodeType.ALL_GATHER, NodeType.REDUCE_SCATTER]:
            return child_nodes_type[1]
        else:
            return NodeType.COMPUTE

    return _get_ir_node_type(node.node)


def reorder_all_gather(
    snodes: List["scheduler.BaseSchedulerNode"],
    all_gather_before_last_wait: Optional[bool] = True,
) -> List["scheduler.BaseSchedulerNode"]:
    """
    Reorder All Gather and Wait in the forward/backward pass;
    1. all_gather_before_last_wait set to True: all_gather_i is reordered before wait_i-1
    2. all_gather_before_last_wait set to False: all_gather_i is reordered after wait_i-1
    """
    result_list: List[scheduler.BaseSchedulerNode] = []
    all_gather_list: List[scheduler.BaseSchedulerNode] = []
    node_to_type: Dict[scheduler.BaseSchedulerNode, int] = {}
    inverse_users, node_users = compute_node_users(snodes)

    for node in snodes:
        node_to_type[node] = get_node_type(node)
    snodes.reverse()
    for idx, node in enumerate(snodes):
        node_type = node_to_type[node]
        if node_type in [NodeType.REDUCE_SCATTER, NodeType.COMPUTE, NodeType.RS_WAIT]:
            # we do not reorder reduce scatter and compute node
            if node not in result_list and node not in all_gather_list:
                result_list.append(node)
        elif node_type == NodeType.ALL_GATHER:
            # gather i-th all gather node and its dependencies
            all_gather_list.append(node)
            inverse_user = list(inverse_users[node])
            inverse_user = [
                n for n in inverse_user if node_to_type[n] == NodeType.COMPUTE
            ]
            if len(inverse_user) > 0:
                all_gather_list.extend(inverse_user)
        elif node_type == NodeType.AG_WAIT:
            if not all_gather_before_last_wait and len(all_gather_list) > 0:
                assert node_to_type[snodes[idx + 1]] == NodeType.ALL_GATHER
                # move i-th all gather node and its dependencies after (i-1)-th wait node (bc this is a reverse list)
                result_list.extend(all_gather_list)
                all_gather_list = []

            result_list.append(node)

            if all_gather_before_last_wait and len(all_gather_list) > 0:
                assert node_to_type[snodes[idx + 1]] == NodeType.ALL_GATHER
                # move i-th all gather node and its dependencies before (i-1)-th wait node (bc this is a reverse list)
                result_list.extend(all_gather_list)
                all_gather_list = []
    if len(all_gather_list) > 0:
        result_list.extend(all_gather_list)
    result_list.reverse()

    return result_list


def reorder_reduce_scatter(
    snodes: List["scheduler.BaseSchedulerNode"],
    front_node: Optional["scheduler.BaseSchedulerNode"] = None,
) -> List["scheduler.BaseSchedulerNode"]:
    """
    Reorder Reduce Scatter and Wait in the backward pass
    reorder wait_i_rs before reduce_scatter_i+1
    """
    result_list: List[scheduler.BaseSchedulerNode] = []
    wait_list: List[scheduler.BaseSchedulerNode] = []
    node_to_type: Dict[scheduler.BaseSchedulerNode, int] = {}
    inverse_users, node_users = compute_node_users(snodes)
    types = []
    for node in snodes:
        node_to_type[node] = get_node_type(node)
        types.append(get_node_type(node))
    for idx, node in enumerate(snodes):
        node_type = node_to_type[node]
        if node_type in [NodeType.ALL_GATHER, NodeType.COMPUTE, NodeType.AG_WAIT]:
            if node not in result_list and node not in wait_list:
                result_list.append(node)
        elif node_type == NodeType.RS_WAIT:
            assert node_to_type[snodes[idx - 1]] == NodeType.REDUCE_SCATTER
            # gather wait node after reduce scatter
            wait_list.append(node)
            node_user = node_users[node]
            node_user = [n for n in node_user if node_to_type[n] == NodeType.COMPUTE]
            wait_list.extend(node_user)
        elif node_type == NodeType.REDUCE_SCATTER:
            if len(wait_list) > 0:
                # move the i-th wait node before (i+1)-th reduce scatter node
                result_list.extend(wait_list)
                wait_list = []
            # add reduce scatter node
            result_list.append(node)

    if len(wait_list) > 0:
        result_list.extend(wait_list)

    return result_list
