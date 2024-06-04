import operator
from dataclasses import dataclass
from typing import cast, List, Set, Tuple, Union

import torch

from ..pattern_matcher import (
    CallFunction,
    Ignored,
    KeywordArg,
    ListOf,
    MULTIPLE,
    PatternMatcherPass,
    register_graph_pattern,
)

aten = torch.ops.aten
patterns = PatternMatcherPass()


def _is_backward(graph: torch.fx.Graph) -> bool:
    placeholders = []
    for node in graph.nodes:
        if node.op != "placeholder":
            break
        placeholders.append(node)
    return not all(node.name.startswith("primal") for node in placeholders)


def _compute_mm_arithmetic_intensity(M: int, N: int, K: int) -> float:
    return M * N * K / (M * K + N * K + M * N)


def _filter_nodes_by_target(nodes: List[torch.fx.Node], target) -> List[torch.fx.Node]:
    return [x for x in nodes if x.target == target]


def _find_ancestors(node: torch.fx.Node) -> Set[torch.fx.Node]:
    ancestors = set()
    ancestors.add(node)
    cur_nodes = [node]
    while len(cur_nodes) > 0:
        new_nodes = []
        for node in cur_nodes:
            for inp in node.all_input_nodes:
                if inp not in ancestors:
                    ancestors.add(inp)
                    new_nodes.append(inp)
        cur_nodes = new_nodes
    return {node for node in ancestors if node.op != "placeholder"}


def _can_schedule_y_before_x(
    x: torch.fx.Node, y: torch.fx.Node
) -> Tuple[bool, Set[torch.fx.Node]]:
    """
    Check if y can be reordered before x and return the ancestors of y
    (inclusive).
    """
    y_ancestors = _find_ancestors(y)
    if x in y_ancestors:
        return False, y_ancestors

    return True, y_ancestors


@dataclass
class _2DMatmul:
    node: torch.fx.Node
    B_node: torch.fx.Node
    B_node_ancestors: Set[torch.fx.Node]

    def replace_with(self, new_node: torch.fx.Node) -> None:
        """
        Replace the matmul with the new node.
        """
        self.node.replace_all_uses_with(new_node)


@dataclass
class _NDMatmul:
    nodes: List[torch.fx.Node]
    B_node: torch.fx.Node
    B_node_ancestors: Set[torch.fx.Node]

    def replace_with(self, new_node: torch.fx.Node) -> None:
        """
        Replace the matmul with the new node.

        ND-matmul is a sequence of reshape -> mm -> reshape in the graph. The
        second reshape node is replaced with `new_node`.

        In addition, we ensure that the original mm node ends up with zero
        users by replacing it with a reverse reshape of `new_node`.
        """
        graph = new_node.graph
        assert len(self.nodes) == 3
        mm_node = self.nodes[1]
        output_reshape_node = self.nodes[2]

        assert mm_node.target == aten.mm.default
        assert output_reshape_node.target == aten.reshape.default

        output_reshape_node.replace_all_uses_with(new_node)
        if len(mm_node.users) > 1:
            with graph.inserting_after(new_node):
                new_mm_node = graph.call_function(
                    aten.reshape.default,
                    args=(new_node, list(mm_node.meta["val"].shape)),
                )
            mm_node.replace_all_uses_with(new_mm_node)


def _find_consumer_matmuls(node: torch.fx.Node) -> List[Union[_2DMatmul, _NDMatmul]]:
    """
    Find the matmuls that use `node` as the lhs argument.
    This function effective normalizes 2D and ND matmuls.
    """
    matmuls: List[Union[_2DMatmul, _NDMatmul]] = []

    for user in node.users:
        # ND matmuls
        if user.target == aten.reshape.default:
            for mm_node in user.users:
                if mm_node.target != aten.mm.default:
                    continue

                B_node = cast(torch.fx.Node, mm_node.args[1])
                can_schedule, B_node_ancestors = _can_schedule_y_before_x(user, B_node)
                if not can_schedule:
                    continue

                for reshape_node in mm_node.users:
                    if reshape_node.target != aten.reshape.default:
                        continue

                    matmul_out_shape = torch.Size(
                        [
                            *node.meta["val"].shape[:-1],
                            B_node.meta["val"].shape[-1],
                        ]
                    )
                    if reshape_node.meta["val"].shape != matmul_out_shape:
                        continue

                    matmuls.append(
                        _NDMatmul(
                            nodes=[user, mm_node, reshape_node],
                            B_node=B_node,
                            B_node_ancestors=B_node_ancestors,
                        )
                    )
        # 2D matmuls
        elif user.target == aten.mm.default:
            B_node = cast(torch.fx.Node, user.args[1])
            can_schedule, B_node_ancestors = _can_schedule_y_before_x(user, B_node)
            if not can_schedule:
                continue

            matmuls.append(
                _2DMatmul(
                    node=user,
                    B_node=B_node,
                    B_node_ancestors=B_node_ancestors,
                ),
            )
    return matmuls


def _find_all_gather_node_from_match(match) -> Tuple[torch.fx.Node, torch.fx.Node]:
    """
    Processes match for ZeroDimAllGather and NonZeroDimAllGather. Returns the
    all-gather node (all_gather_into_tensor.default) and the all-gather result
    node (wait_tensor.default for gather_dim == 0 and aten.cat.default for
    gather_dim == 1). This function effectively normalizes zero-dim and
    non-zero-dim all_gather_tensor.
    """
    # gather_dim == 0
    if len(match.nodes) == 2:
        return match.nodes[0], match.nodes[1]
    # gather_dim == 1
    ag_node = _filter_nodes_by_target(
        match.nodes,
        torch.ops._c10d_functional.all_gather_into_tensor.default,
    )[0]
    ag_res_node = _filter_nodes_by_target(
        match.nodes,
        aten.cat.default,
    )[0]
    shard_node = ag_node.args[0]
    return ag_node, ag_res_node


def fuse_all_gather_matmul_zero_dim(match, shard, group_name):
    fuse_all_gather_matmul(match, shard, 0, group_name)


def fuse_all_gather_matmul(match, shard, gather_dim, group_name):
    """
    Fused the pattern

        A = all_gather_tensor(A_shard, gather_dim, group_name)
        C_0 = torch.matmul(A, B_0)
        C_1 = torch.matmul(A, B_1)
        C_2 = torch.matmul(A, B_2)
        ...

    into

        A, Cs = torch.ops.cuda_p2p.fused_all_gather_matmul(
            A_shard, [B_0, B_1, B_2, ...], gather_dim, group_name,
        )
    """
    if (
        not torch.distributed.is_available()
        or not torch.distributed.is_nccl_available()
    ):
        return

    c10d = torch.ops._c10d_functional
    from torch.distributed._cuda_p2p import is_cuda_p2p_group
    from torch.distributed.distributed_c10d import _resolve_process_group

    if gather_dim >= len(shard.meta["val"].shape) - 1:
        # Decomposing the matmul on the K dimension is not supported
        return

    if not is_cuda_p2p_group(_resolve_process_group(group_name)):
        return

    # Normalize zero-dim and non-zero-dim all_gather_tensor
    ag_node, ag_res_node = _find_all_gather_node_from_match(match)

    # Find consumer matmuls for eligible for fusion
    matmuls = _find_consumer_matmuls(ag_res_node)
    if len(matmuls) == 0:
        return

    shard_node = ag_node.args[0]
    B_nodes = [matmul.B_node for matmul in matmuls]

    # Fuse the all_gather_tensor with the eligible matmuls
    graph = ag_node.graph
    with graph.inserting_before(ag_node):
        fused_node = graph.call_function(
            torch.ops.cuda_p2p.fused_all_gather_matmul.default,
            args=(shard_node, B_nodes, gather_dim, group_name),
        )
        new_ag_node = graph.call_function(
            operator.getitem,
            args=(fused_node, 0),
        )
        new_out_nodes = graph.call_function(
            operator.getitem,
            args=(fused_node, 1),
        )
        for idx, matmul in enumerate(matmuls):
            new_out_node = graph.call_function(
                operator.getitem,
                args=(new_out_nodes, idx),
            )
            matmul.replace_with(new_out_node)
        ag_res_node.replace_all_uses_with(new_ag_node)

    # Raise ancestors of B that are topologically ordered between ag_res_node
    # and the matmul above fused_node. _find_consumer_matmuls guarantees that
    # ag_res_node is not an ancestor of B.
    order = {node: idx for idx, node in enumerate(graph.nodes)}
    nodes_to_raise = sorted(
        {x for matmul in matmuls for x in matmul.B_node_ancestors},
        key=lambda x: order[x],
    )
    for node in nodes_to_raise:
        if order[node] > order[fused_node]:
            fused_node.prepend(node)

    graph.eliminate_dead_code()
    return


def fuse_matmul_reduce_scatter_zero_dim(match, rs_input, reduce_op, group_name):
    fuse_matmul_reduce_scatter(match, rs_input, reduce_op, 0, group_name)


def fuse_matmul_reduce_scatter(match, rs_input, reduce_op, scatter_dim, group_name):
    """
    Fused the pattern

        reduce_scatter_tensor(A @ B, scatter_dim, group_name)

    into

        torch.ops.cuda_p2p.fused_matmul_reduce_scatter(
            A, B, scatter_dim, group_name,
        )
    """
    if (
        not torch.distributed.is_available()
        or not torch.distributed.is_nccl_available()
    ):
        return

    c10d = torch.ops._c10d_functional
    from torch.distributed._cuda_p2p import is_cuda_p2p_group
    from torch.distributed.distributed_c10d import _resolve_process_group

    if not is_cuda_p2p_group(_resolve_process_group(group_name)):
        return

    # Currently fused_matmul_reduce_scatter doesn't return the matmul result,
    # so we can't apply the fusion if the matmul result is used by multiple
    # users. This is not a fundamental limitation of the fused op and can be
    # addressed if needed.
    if len(rs_input.users) != 1:
        return

    # 2D matmul
    if rs_input.target == aten.mm.default:
        A_node, B_node = rs_input.args[0], rs_input.args[1]
    # ND matmul
    elif rs_input.target == aten.reshape.default:
        mm_node = rs_input.args[0]
        if mm_node.target != aten.mm.default or len(mm_node.users) != 1:
            return

        A_node, B_node = mm_node.args[0], mm_node.args[1]
        if A_node.target != aten.reshape.default:
            return
        A_node = A_node.args[0]
    # Not matmul
    else:
        return

    rs_res_node = _filter_nodes_by_target(match.nodes, c10d.wait_tensor.default)[0]
    if not _can_schedule_y_before_x(rs_res_node, B_node):
        return

    graph = rs_res_node.graph
    with graph.inserting_before(rs_res_node):
        fused_node = graph.call_function(
            torch.ops.cuda_p2p.fused_matmul_reduce_scatter.default,
            args=(A_node, B_node, reduce_op, scatter_dim, group_name),
        )
        rs_res_node.replace_all_uses_with(fused_node)

    order = {node: idx for idx, node in enumerate(graph.nodes)}
    nodes_to_raise = sorted(
        _find_ancestors(B_node),
        key=lambda x: order[x],
    )
    for node in nodes_to_raise:
        if order[node] > order[fused_node]:
            fused_node.prepend(node)

    graph.eliminate_dead_code()


def _register_passes():
    if (
        not torch.distributed.is_available()
        or not torch.distributed.is_nccl_available()
    ):
        return

    c10d = torch.ops._c10d_functional

    # Matches funcol.all_gather_tensor with gather_dim == 0
    ZeroDimAllGather = CallFunction(
        c10d.wait_tensor.default,
        CallFunction(
            c10d.all_gather_into_tensor.default,
            KeywordArg("shard"),
            Ignored(),
            KeywordArg("group_name"),
        ),
    )

    # Matches funcol.all_gather_tensor with gather_dim > 0
    # NOTE: this pattern may need to be updated if funcol.all_gather_tensor changes
    NonZeroDimAllGather = CallFunction(
        aten.cat.default,
        ListOf(
            CallFunction(
                operator.getitem,
                CallFunction(
                    aten.split.Tensor,
                    CallFunction(
                        c10d.wait_tensor.default,
                        CallFunction(
                            c10d.all_gather_into_tensor.default,
                            KeywordArg("shard"),
                            Ignored(),
                            KeywordArg("group_name"),
                        ),
                    ),
                    Ignored(),
                    _users=MULTIPLE,
                ),
                Ignored(),
            ),
        ),
        KeywordArg("gather_dim"),
        _users=MULTIPLE,
    )

    register_graph_pattern(
        ZeroDimAllGather,
        pass_dict=patterns,
    )(fuse_all_gather_matmul_zero_dim)

    register_graph_pattern(
        NonZeroDimAllGather,
        pass_dict=patterns,
    )(fuse_all_gather_matmul)

    # Matches funcol.reduce_scatter_tensor with scatter_dim == 0
    ZeroDimReduceScatter = CallFunction(
        c10d.wait_tensor.default,
        CallFunction(
            c10d.reduce_scatter_tensor.default,
            KeywordArg("rs_input"),
            KeywordArg("reduce_op"),
            Ignored(),
            KeywordArg("group_name"),
        ),
    )

    # Matches funcol.reduce_scatter_tensor with scatter_dim > 0
    # NOTE: this pattern may need to be updated if funcol.reduce_scatter_tensor
    # changes
    NonZeroDimReduceScatter = CallFunction(
        c10d.wait_tensor.default,
        CallFunction(
            c10d.reduce_scatter_tensor.default,
            CallFunction(
                aten.cat.default,
                ListOf(
                    CallFunction(
                        operator.getitem,
                        CallFunction(
                            aten.split.Tensor,
                            KeywordArg("rs_input"),
                            Ignored(),
                            KeywordArg("scatter_dim"),
                            _users=MULTIPLE,
                        ),
                        Ignored(),
                    )
                ),
            ),
            KeywordArg("reduce_op"),
            Ignored(),
            KeywordArg("group_name"),
        ),
    )

    register_graph_pattern(
        ZeroDimReduceScatter,
        pass_dict=patterns,
    )(fuse_matmul_reduce_scatter_zero_dim)

    register_graph_pattern(
        NonZeroDimReduceScatter,
        pass_dict=patterns,
    )(fuse_matmul_reduce_scatter)


_register_passes()
