from __future__ import annotations

import heapq
from collections import Counter, defaultdict
from typing import Any, Optional

import torch
import torch.fx as fx
from torch._dynamo.graph_deduplication import _stable_topological_sort
from torch._inductor.fx_passes.bucketing import (
    is_all_gather_into_tensor as is_all_gather,
    is_reduce_scatter_tensor as is_reduce_scatter,
    is_wait_tensor,
    merge_all_gather_bucket,
    merge_reduce_scatter_bucket,
)
from torch._inductor.fx_passes.overlap_preserving_bucketer import (
    bucket_key,
    OverlapPreservingBucketer,
)
from torch._inductor.fx_passes.overlap_scheduling import (
    CollectiveInfo,
    is_compute_node,
    OverlapScheduler,
)
from torch.utils._ordered_set import OrderedSet

from .graph_view import get_subgraph_by_path, GraphView, make_graph_view


class ManualOverlapPreservingBucketer(OverlapPreservingBucketer):
    """
    Buckets collective operations based on user specifications.
    The actual bucket happens in bucket_collectives, where all-gathers/reduce-scatters in
        `nodes` will be buckted one single all-gather/reduce-scatter.
    """

    def __init__(
        self, node_users: dict[fx.Node, OrderedSet[fx.Node]], *args: Any, **kwargs: Any
    ):
        super().__init__(*args, **kwargs)
        self.node_users = node_users
        self.wait_to_node_map: dict[fx.Node, fx.Node] = defaultdict()

    def _check_recursive_dep(
        self,
        node: fx.Node,
        target_op: str,
        dep_dict: dict[torch.fx.Node, OrderedSet[torch.fx.Node]],
    ) -> bool:
        """
        Check if the node is directly used for fetch parameters/gradients

        TODO (ruisizhang123): currently, we assume the node only pre-fetch/update one parameter/gradient
            We should handle multiple parameters/gradients update case by checking if there are non closure
            computes along the path from primal/output to coll_node
        """
        deps: OrderedSet[fx.Node] = dep_dict[node]
        seen_target_op = 0
        for d in deps:
            if d.op == target_op:
                seen_target_op += 1

        return seen_target_op == 1

    def _bucket_group(self, coll_nodes: list[fx.Node]) -> None:
        assert len(coll_nodes) > 0, "bucketed coll_nodes should have nonzero node"

        waits = [self.collective_info[n].wait_node for n in coll_nodes]
        # Use earliest wait insertion point
        first_wait = min(waits, key=lambda w: self.node_idx[w])
        # Find insertion location
        first = coll_nodes[0]
        next_node = first
        while next_node in coll_nodes:
            next_node = next_node.next

        if is_all_gather(first):
            new_nodes, replacements = merge_all_gather_bucket(
                self.graph,
                coll_nodes,
                wait_insertion_point=first_wait,
                insert_before=next_node,
                mode="custom_ops",
            )
        elif is_reduce_scatter(first):
            new_nodes, replacements = merge_reduce_scatter_bucket(
                self.graph,
                coll_nodes,
                wait_insertion_point=first_wait,
                insert_before=next_node,
                mode="custom_ops",
            )
        else:
            raise ValueError(
                "bucket non all_gather/reduce_scatter node is not supported"
            )

        # Identify the new wait and start
        new_waits = [n for n in new_nodes if is_wait_tensor(n)]
        assert len(new_waits) == 1, f"Expected exactly one new wait, got {new_waits}"
        new_wait = new_waits[0]
        new_start = new_wait.args[0]
        assert isinstance(new_start, fx.Node)

        node_type = (
            "bucketed_all_gather" if is_all_gather(first) else "bucketed_reduce_scatter"
        )
        for n in new_nodes:
            n.meta["nn_module_stack"] = coll_nodes[0].meta.get("nn_module_stack", "")
            n.meta["fwd_nn_module_stack"] = coll_nodes[0].meta.get(
                "fwd_nn_module_stack", ""
            )
            if n == new_wait:
                node_type = node_type + "_wait"
            n.meta["manual_bucket_node_type"] = node_type
            if "wait" in node_type:
                self.wait_to_node_map[n] = new_wait

    def manual_bucket_collectives(self, nodes: list[fx.Node]) -> None:
        """
        Bucket all all-gather/reduce-scatter nodes from nodes into one all-gather/reduce-scatter.
        """
        # Filter out valid collectives
        collectives = [n for n in nodes if n in self.collective_info]
        if collectives == []:
            return
        grouped_collectives: dict[object, OrderedSet[fx.Node]] = defaultdict(OrderedSet)
        for node in collectives:
            key = bucket_key(node)
            if not (is_all_gather(node) or is_reduce_scatter(node)):
                continue
            # We only want to bucket all-gather/reduce-scatter that
            # 1. all_gather that have ancestors dependent only on input placeholder(parameters)
            # 2. reduce scatter that the wait user node is returned as output(gradients)
            if is_all_gather(node) and not self._check_recursive_dep(
                node, "placeholder", self.node_ancestors
            ):
                continue
            if is_reduce_scatter(node) and not self._check_recursive_dep(
                self.collective_info[node].wait_node, "output", self.node_users
            ):
                continue
            if key is not None:
                grouped_collectives[key].add(node)

        for key, nodes in grouped_collectives.items():  # type: ignore[arg-type]
            self._bucket_group(list(nodes))


class ManualOverlapScheduler(OverlapScheduler):
    """
    Scheduler that manual buckets and reorders collective nodes based on module_bucket_plans
    """

    def __init__(
        self,
        gm: fx.GraphModule,
        module_bucket_plans: list[list[str] | str],
        insert_overlap_deps: bool,
    ):
        super().__init__(
            gm,
            max_in_flight_gb=0.0,
            max_compute_pre_fetch=0,
            collective_bucketing=True,
            insert_overlap_deps=insert_overlap_deps,
            compute_overlap_multipler=0.0,
            max_coll_distance=0,
            custom_runtime_estimation=None,
            collective_estimator="analytical",
        )
        self.module_bucket_plans = module_bucket_plans
        self.nodes_in_subgraph: list[list[fx.Node]] = []

        self.node_users: dict[fx.Node, OrderedSet[fx.Node]] = self._collect_node_users()
        self.bucketer = ManualOverlapPreservingBucketer(
            graph=self.graph,
            collective_info=self.collective_info,
            node_ancestors=self.node_ancestors,
            node_users=self.node_users,
            scheduled=OrderedSet(self.graph.nodes),
        )
        self.insert_overlap_deps = insert_overlap_deps

    def _identify_collectives(self) -> None:
        """Identify all collective operations."""
        for node in self.nodes:
            if is_wait_tensor(node):
                start = node.args[0]
                info = CollectiveInfo(
                    start_node=start,
                    wait_node=node,
                    size_bytes=0,
                    estimated_time_ms=0,
                    exposed_time_ms=0,
                )
                self.collective_info[start] = info
                self.wait_to_start[node] = start
                self.unscheduled_collectives.add(start)

    def run(self) -> torch.fx.GraphModule:
        """Entry point to run the manual bucket algorithm"""
        # Bucket collectives in each bucket_module
        self._manual_bucket_collectives()

        # Reorder collectives with last/next bucket_module
        self._manual_reorder_graph()

        return self.gm

    def _manual_reorder_graph(self) -> None:
        """
        Reorder nodes in the FX graph to enforce manual overlap dependencies.

        Enforce:
        - all_gather_start_i depends on all_gather_wait_(i-1)
        - reduce_scatter_wait_i must happen before reduce_scatter_start_(i+1)
        """
        delayed_rs_nodes: list[fx.Node] = []
        overlap_deps: dict[fx.Node, OrderedSet[fx.Node]] = defaultdict(OrderedSet)

        # schedule reduce scatter normally in self._schedule
        while self.ready:
            _, node = heapq.heappop(self.ready)
            node_type = node.meta.get("manual_bucket_node_type", "")

            if node in self.scheduled:
                continue

            if node_type == "bucketed_reduce_scatter":
                # Ensure all delayed waits execute before this reduce_scatter
                for delayed in delayed_rs_nodes:
                    self._schedule(delayed)
                    overlap_deps[delayed].add(node)
                delayed_rs_nodes.clear()

            elif node_type == "bucketed_reduce_scatter_wait":
                # Defer until next reduce_scatter
                delayed_rs_nodes.append(node)
                continue
            self._schedule(node)

        for delayed in delayed_rs_nodes:
            self._schedule(delayed)

        self.scheduled = OrderedSet(reversed(list(self.scheduled)))
        picked_ag: list[fx.Node] = []
        last_compute: Optional[fx.Node] = None

        for node in self.scheduled:
            node_type = node.meta.get("manual_bucket_node_type", "")
            if node_type == "bucketed_all_gather":
                picked_ag.append(node)
                continue

            if node_type == "bucketed_all_gather_wait":
                # Connect corresponding all_gather_wait -> all_gather edges
                if picked_ag:
                    for ag in picked_ag:
                        overlap_deps[self.bucketer.wait_to_node_map[node]].add(ag)
                picked_ag.clear()
            if is_compute_node(node):
                last_compute = node

        if last_compute is not None and not bool(
            OrderedSet(picked_ag) & OrderedSet(self.node_ancestors[last_compute])
        ):
            for ag in picked_ag:
                overlap_deps[last_compute].add(ag)

        _stable_topological_sort(self.graph, overlap_deps)
        self.graph.lint()

        if self.insert_overlap_deps:
            from torch._inductor.fx_passes.control_dependencies import (
                preserve_node_ordering,
            )

            preserve_node_ordering(self.graph, overlap_deps)

    def _manual_bucket_collectives(self) -> None:
        """Bucket nodes in each module_bucket from module_bucket_plans."""
        self._obtain_nodes_in_subgraph()
        for i, nodes in enumerate(self.nodes_in_subgraph):
            self.bucketer.manual_bucket_collectives(nodes=nodes)

        _stable_topological_sort(self.graph, {})
        self.graph.lint()
        self.nodes = list(self.graph.nodes)
        self.in_degree = Counter(user for node in self.nodes for user in node.users)

    def _collect_node_users(self) -> dict[fx.Node, OrderedSet[fx.Node]]:
        """Collect all users for each node."""
        node_users: dict[fx.Node, OrderedSet[fx.Node]] = defaultdict(OrderedSet)
        for node in self.nodes:
            for output_node in list(node.users.keys()):
                node_users[node].add(output_node)
                node_users[node] |= node_users[output_node]
        return node_users

    def _schedule(self, node: fx.Node) -> None:
        """Schedule a node."""
        assert node not in self.scheduled
        assert all(n in self.scheduled for n in node.all_input_nodes)
        self.scheduled.add(node)
        for user in node.users:
            self.in_degree[user] -= 1
            if self.in_degree[user] == 0:
                heapq.heappush(self.ready, ((), user))

    def _obtain_nodes_in_subgraph(self) -> None:
        """
        Obtain nodes in each subgraph from module_bucket_plans
        """
        graph_view: GraphView | None = make_graph_view(self.graph)
        if graph_view is None:
            return

        for module in self.module_bucket_plans:
            subgraph_view = get_subgraph_by_path(graph_view, module)
            self.nodes_in_subgraph.append(subgraph_view)

        all_subgraph_nodes = [
            node for sublist in self.nodes_in_subgraph for node in sublist
        ]
        unique_subgraph_nodes = list(OrderedSet(all_subgraph_nodes))
        assert len(all_subgraph_nodes) <= len(unique_subgraph_nodes), (
            f"Overlapping FX nodes detected across subgraphs in `module_bucket_plans`. "
            f"Expected disjoint node sets but found "
            f"{len(all_subgraph_nodes) - len(unique_subgraph_nodes)} duplicated node(s)."
        )


def manual_overlap_bucketing(
    gm: torch.fx.GraphModule,
    module_bucket_plans: list[list[str] | str],
    insert_overlap_deps: bool = False,
) -> torch.fx.GraphModule:
    """Schedule nodes based on user specifications in module_bucket_plans
    The manual overlapping consists of two steps:
    Step 1: bucket all-gather/reduce-scatter in each module in module_bucket_plans
    Step 2: reorder all-gather to overlap with last module_bucket &
        reorder reduce-scatter to overlap with next module_bucket
    TODO(ruisizhang123): allow users to explicitly specify which
        module_bucket they want to overlap.

    Args:
        gm: input graph module to optimize.
        module_bucket_plans: user specified FQNs
    """
    # decode abbreviated FQNs to actual FQNs
    overlapped_gm = ManualOverlapScheduler(
        gm, module_bucket_plans, insert_overlap_deps
    ).run()
    overlapped_gm.recompile()
    return overlapped_gm
