# mypy: allow-untyped-defs
import operator
from collections import defaultdict
from dataclasses import dataclass
from typing import cast, Dict, List, Set, Tuple, Union

import torch
from .. import config, inductor_prims
from ..pattern_matcher import (
    CallFunction,
    Ignored,
    KeywordArg,
    ListOf,
    Match,
    MULTIPLE,
    PatternExpr,
    PatternMatcherPass,
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


def _get_tensor(node: torch.fx.Node) -> torch.Tensor:
    val = node.meta["val"]
    assert isinstance(val, torch.Tensor)
    return val


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
class _AllGatherMatch:
    match: Match
    shard_node: torch.fx.Node
    ag_node: torch.fx.Node
    res_node: torch.fx.Node
    gather_dim: int
    group_name: str


def find_all_gather_patterns(graph: torch.fx.Graph):
    c10d = torch.ops._c10d_functional

    # Matches funcol.all_gather_tensor with gather_dim == 0
    zero_dim_all_gather_pattern = CallFunction(
        c10d.wait_tensor.default,
        CallFunction(
            c10d.all_gather_into_tensor.default,
            KeywordArg("shard"),
            Ignored(),
            KeywordArg("group_name"),
        ),
    )

    # Matches funcol.all_gather_tensor with gather_dim > 0
    non_zero_dim_all_gather_pattern = CallFunction(
        aten.cat.default,
        ListOf(
            CallFunction(
                operator.getitem,
                CallFunction(
                    aten.split.Tensor,
                    zero_dim_all_gather_pattern,
                    Ignored(),
                    _users=MULTIPLE,
                ),
                Ignored(),
            ),
        ),
        KeywordArg("gather_dim"),
    )

    all_gathers = []
    visited_waits = set()
    for node in reversed(graph.nodes):
        if node.target == aten.cat.default:
            if match := non_zero_dim_all_gather_pattern.match(node):
                assert isinstance(match, Match)
                ag_match = _AllGatherMatch(
                    match=match,
                    shard_node=match.kwargs["shard"],
                    ag_node=match.nodes[0],
                    res_node=node,
                    gather_dim=match.kwargs["gather_dim"],
                    group_name=match.kwargs["group_name"],
                )
                visited_waits.add(match.nodes[1])
                all_gathers.append(ag_match)
        elif node.target == c10d.wait_tensor.default:
            if node in visited_waits:
                continue
            if match := zero_dim_all_gather_pattern.match(node):
                assert isinstance(match, Match)
                ag_match = _AllGatherMatch(
                    match=match,
                    shard_node=match.kwargs["shard"],
                    ag_node=match.nodes[0],
                    res_node=node,
                    gather_dim=0,
                    group_name=match.kwargs["group_name"],
                )
                all_gathers.append(ag_match)
    return list(reversed(all_gathers))


@dataclass
class _ReduceScatterMatch:
    match: Match
    input_node: torch.fx.Node
    rs_node: torch.fx.Node
    res_node: torch.fx.Node
    reduce_op: str
    scatter_dim: int
    group_name: str


def find_reduce_scatter_patterns(graph: torch.fx.Graph):
    c10d = torch.ops._c10d_functional

    def reduce_scatter_template(inp: PatternExpr):
        return CallFunction(
            c10d.wait_tensor.default,
            CallFunction(
                c10d.reduce_scatter_tensor.default,
                inp,
                KeywordArg("reduce_op"),
                Ignored(),
                KeywordArg("group_name"),
            ),
        )

    # Matches funcol.reduce_scatter_tensor with scatter_dim == 0
    zero_dim_reduce_scatter_pattern = reduce_scatter_template(KeywordArg("input"))

    # Matches funcol.reduce_scatter_tensor with scatter_dim > 0
    non_zero_dim_reduce_scatter_pattern = reduce_scatter_template(
        CallFunction(
            aten.cat.default,
            ListOf(
                CallFunction(
                    operator.getitem,
                    CallFunction(
                        aten.split.Tensor,
                        KeywordArg("input"),
                        Ignored(),
                        KeywordArg("scatter_dim"),
                        _users=MULTIPLE,
                    ),
                    Ignored(),
                )
            ),
        ),
    )

    reduce_scatters = []
    for node in reversed(graph.nodes):
        if node.target == c10d.wait_tensor.default:
            if match := non_zero_dim_reduce_scatter_pattern.match(node):
                assert isinstance(match, Match)
                reduce_scatters.append(
                    _ReduceScatterMatch(
                        match=match,
                        input_node=match.kwargs["input"],
                        rs_node=match.nodes[-2],
                        res_node=node,
                        reduce_op=match.kwargs["reduce_op"],
                        scatter_dim=match.kwargs["scatter_dim"],
                        group_name=match.kwargs["group_name"],
                    )
                )
            elif match := zero_dim_reduce_scatter_pattern.match(node):
                assert isinstance(match, Match)
                reduce_scatters.append(
                    _ReduceScatterMatch(
                        match=match,
                        input_node=match.kwargs["input"],
                        rs_node=match.nodes[0],
                        res_node=node,
                        reduce_op=match.kwargs["reduce_op"],
                        scatter_dim=0,
                        group_name=match.kwargs["group_name"],
                    )
                )
    return list(reversed(reduce_scatters))


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
                    args=(new_node, list(_get_tensor(mm_node).shape)),
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

                B_node = mm_node.args[1]
                assert isinstance(B_node, torch.fx.Node)
                can_schedule, B_node_ancestors = _can_schedule_y_before_x(user, B_node)
                if not can_schedule:
                    continue

                for reshape_node in mm_node.users:
                    if reshape_node.target != aten.reshape.default:
                        continue

                    matmul_out_shape = torch.Size(
                        [
                            *_get_tensor(node).shape[:-1],
                            _get_tensor(B_node).shape[-1],
                        ]
                    )
                    if _get_tensor(reshape_node).shape != matmul_out_shape:
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


def fuse_all_gather_matmul(all_gather: _AllGatherMatch) -> None:
    """
    Fused the pattern

        A = all_gather_tensor(A_shard, gather_dim, group_name)
        C_0 = torch.matmul(A, B_0)
        C_1 = torch.matmul(A, B_1)
        C_2 = torch.matmul(A, B_2)
        ...

    into

        A, Cs = torch.ops.symm_mem.fused_all_gather_matmul(
            A_shard, [B_0, B_1, B_2, ...], gather_dim, group_name,
        )
    """
    if (
        not torch.distributed.is_available()
        or not torch.distributed.is_nccl_available()
    ):
        return

    c10d = torch.ops._c10d_functional
    from torch.distributed._symmetric_memory import (
        is_symm_mem_enabled_for_group,
        restride_A_shard_for_fused_all_gather_matmul,
    )

    shard_node, ag_node, ag_res_node, gather_dim, group_name = (
        all_gather.shard_node,
        all_gather.ag_node,
        all_gather.res_node,
        all_gather.gather_dim,
        all_gather.group_name,
    )

    if not is_symm_mem_enabled_for_group(group_name):
        return

    if gather_dim >= len(_get_tensor(shard_node).shape) - 1:
        # Decomposing the matmul on the K dimension is not supported
        return

    # Find consumer matmuls for eligible for fusion
    matmuls = _find_consumer_matmuls(ag_res_node)
    if len(matmuls) == 0:
        return

    B_nodes = [matmul.B_node for matmul in matmuls]

    # Fuse the all_gather_tensor with the eligible matmuls
    graph = ag_node.graph
    with graph.inserting_before(ag_node):
        if "val" in shard_node.meta:
            restrided = restride_A_shard_for_fused_all_gather_matmul(
                _get_tensor(shard_node),
                gather_dim,
            )
            shard_node = graph.call_function(
                inductor_prims.force_stride_order,
                args=(shard_node, restrided.stride()),
            )

        fused_node = graph.call_function(
            torch.ops.symm_mem.fused_all_gather_matmul.default,
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


def fuse_matmul_reduce_scatter(reduce_scatter: _ReduceScatterMatch) -> None:
    """
    Fused the pattern

        reduce_scatter_tensor(A @ B, scatter_dim, group_name)

    into

        torch.ops.symm_mem.fused_matmul_reduce_scatter(
            A, B, scatter_dim, group_name,
        )
    """
    if (
        not torch.distributed.is_available()
        or not torch.distributed.is_nccl_available()
    ):
        return

    c10d = torch.ops._c10d_functional
    from torch.distributed._symmetric_memory import (
        is_symm_mem_enabled_for_group,
        restride_A_for_fused_matmul_reduce_scatter,
    )

    input_node, rs_node, rs_res_node, reduce_op, scatter_dim, group_name = (
        reduce_scatter.input_node,
        reduce_scatter.rs_node,
        reduce_scatter.res_node,
        reduce_scatter.reduce_op,
        reduce_scatter.scatter_dim,
        reduce_scatter.group_name,
    )

    if not is_symm_mem_enabled_for_group(group_name):
        return

    # Currently fused_matmul_reduce_scatter doesn't return the matmul result,
    # so we can't apply the fusion if the matmul result is used by multiple
    # users. This is not a fundamental limitation of the fused op and can be
    # addressed if needed.
    if len(input_node.users) != 1:
        return

    # 2D matmul
    if input_node.target == aten.mm.default:
        A_node, B_node = input_node.args[0], input_node.args[1]
    # ND matmul
    elif input_node.target == aten.reshape.default:
        mm_node = input_node.args[0]
        assert isinstance(mm_node, torch.fx.Node)
        if mm_node.target != aten.mm.default or len(mm_node.users) != 1:
            return

        A_node, B_node = mm_node.args[0], mm_node.args[1]
        assert isinstance(A_node, torch.fx.Node)
        if A_node.target != aten.reshape.default:
            return
        A_node = A_node.args[0]
    # Not matmul
    else:
        return

    assert isinstance(A_node, torch.fx.Node)
    assert isinstance(B_node, torch.fx.Node)

    if not _can_schedule_y_before_x(rs_res_node, B_node):
        return

    graph = rs_res_node.graph
    with graph.inserting_before(rs_res_node):
        if "val" in A_node.meta:
            restrided = restride_A_for_fused_matmul_reduce_scatter(
                _get_tensor(A_node),
                scatter_dim,
            )
            A_node = graph.call_function(
                inductor_prims.force_stride_order,
                args=(A_node, restrided.stride()),
            )

        fused_node = graph.call_function(
            torch.ops.symm_mem.fused_matmul_reduce_scatter.default,
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


def _get_node_to_ancestors(
    graph: torch.fx.Graph,
) -> Dict[torch.fx.Node, Set[torch.fx.Node]]:
    """
    Compute the ancestors for all nodes in a graph.
    """
    node_to_ancestors = defaultdict(set)
    for node in graph.nodes:
        node_to_ancestors[node] = set(node.all_input_nodes)
        for dep in node.all_input_nodes:
            node_to_ancestors[node] |= node_to_ancestors[dep]

    return node_to_ancestors


def _get_collective_to_overlappable_nodes(
    graph: torch.fx.Graph,
) -> Dict[torch.fx.Node, List[torch.fx.Node]]:
    """
    For each collective in the graph, find nodes that are neither ancestors nor
    descendants of the collective.
    """

    def is_collective(node) -> bool:
        # Only consider all-gather and reduce-scatter in the context of
        # micro-pipeline TP.
        return node.target in [
            torch.ops._c10d_functional.all_gather_into_tensor.default,
            torch.ops._c10d_functional.reduce_scatter_tensor.default,
        ]

    node_to_ancestors = _get_node_to_ancestors(graph)
    collective_to_overlappable_nodes = defaultdict(list)
    for node in graph.nodes:
        if not is_collective(node):
            continue
        for x in graph.nodes:
            if (
                node not in node_to_ancestors[x]
                and x not in node_to_ancestors[node]
                and x.op == "call_function"
            ):
                collective_to_overlappable_nodes[node].append(x)

    return collective_to_overlappable_nodes


def _get_unexposed_collectives(graph: torch.fx.Graph) -> List[torch.fx.Node]:
    """
    Find all unexposed collectives in the graph.

    Because we don't have the runtime estimate, this function is a rough
    estimation using the following strong/hand-wavy assumptions:

    - Only a predefined set of "compute intensive" operation can hide a collective.
    - Any "compute intensive" operation can hide exactly one collective.
    """

    def _is_compute_intensive(node: torch.fx.Node) -> bool:
        return node.target in [torch.ops.aten.mm.default]

    collective_to_overlapping_candidates = defaultdict(list)
    available_nodes = set()
    collective_to_overlappable_nodes = _get_collective_to_overlappable_nodes(graph)
    for collective, overlappable_nodes in collective_to_overlappable_nodes.items():
        candidates = [x for x in overlappable_nodes if _is_compute_intensive(x)]
        collective_to_overlapping_candidates[collective] = candidates
        available_nodes |= set(candidates)

    unexposed_collectives = []
    for (
        collective,
        overlapping_candidates,
    ) in collective_to_overlapping_candidates.items():
        # Each collective consumes exactly one overlapping candidate
        for x in overlapping_candidates:
            if x in available_nodes:
                unexposed_collectives.append(collective)
                available_nodes.remove(x)
                break
    return unexposed_collectives


def micro_pipeline_tp_pass(graph: torch.fx.Graph):
    all_gathers = find_all_gather_patterns(graph)
    reduce_scatters = find_reduce_scatter_patterns(graph)

    # When a collective can be hidden through either simple overlapping or
    # micro-pipeline TP, we prefer simple overlapping to avoid the overhead
    # associated with decomposition. If reorder_for_compute_comm_overlap is
    # enabled, we identify collectives that can be hidden through simple
    # overlapping and exclude them from micro-pipeline TP candidates.
    if config.reorder_for_compute_comm_overlap:
        unexposed_collectives = _get_unexposed_collectives(graph)
        all_gathers = [x for x in all_gathers if x.ag_node not in unexposed_collectives]
        reduce_scatters = [
            x for x in reduce_scatters if x.rs_node not in unexposed_collectives
        ]

    for all_gather in all_gathers:
        fuse_all_gather_matmul(all_gather)

    for reduce_scatter in reduce_scatters:
        fuse_matmul_reduce_scatter(reduce_scatter)
