from __future__ import annotations

import heapq
import itertools
import re
from collections import Counter, defaultdict
from typing import Any, Callable, Optional, Union

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


def _get_module_stack(node: fx.Node) -> list[tuple[str, type[Any]]]:
    if node.meta.get("nn_module_stack", "") == "":
        if node.meta.get("fwd_nn_module_stack", "") != "":
            return list(node.meta["fwd_nn_module_stack"].values())
        return []
    return list(node.meta["nn_module_stack"].values())


def _addindent(s_: str, num_spaces: int) -> str:
    s: list[str] = s_.split("\n")
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first: str = s.pop(0)
    s: list[str] = [(num_spaces * " ") + line for line in s]
    joint_s: str = "\n".join(s)
    joint_s = first + "\n" + joint_s
    return joint_s


class Container:
    def __init__(self, name: str, klass: type[Any]) -> None:
        self.name: str = name
        self.klass: type[Any] = klass
        self.data: list[fx.Node] = []
        self.unique_nodes: OrderedSet[fx.Node] = OrderedSet()
        self.children: dict[str, Container] = {}

    def add(self, data: fx.Node) -> None:
        if data not in self.unique_nodes:
            self.data.append(data)
            self.unique_nodes.add(data)

    def get_child(
        self, module_stack: str, klass: Optional[type[Any]] = None
    ) -> Container:
        if module_stack not in self.children:
            new_stack = Container(module_stack, klass or self.klass)
            self.children[module_stack] = new_stack
        return self.children[module_stack]

    def __getitem__(self, name: str) -> Container:
        return self.children[name]

    def __getattr__(self, name: str) -> Container:
        return self.children[name]

    def __repr__(self) -> str:
        child_lines: list[str] = []
        for name, child in self.children.items():
            mod_str = repr(child)
            mod_str = _addindent(mod_str, 2)
            child_lines.append(f"({name}): {mod_str}")
        main_str = f"{self.klass.__name__}("
        if child_lines:
            main_str += "\n  " + "\n  ".join(child_lines) + "\n"
        main_str += ")"
        return main_str

    def graph_view(self) -> fx.Graph:
        return _make_subgraph(self.data)


def _clean_stack_name(val: str) -> str:
    name: str = (
        val.replace("L['self']", "Model")
        .replace("_modules['", "")
        .replace("['", ".")
        .replace("']", "")
    )
    return name


def _find_key_nodes(nodes: list[fx.Node]) -> tuple[list[fx.Node], list[fx.Node]]:
    root = []
    outputs = []
    nodes_set = OrderedSet(nodes)
    for node in nodes:
        for x in node.all_input_nodes:
            if x not in nodes_set:
                root.append(x)
        if all(x not in nodes_set for x in node.users):
            outputs.append(node)
    return root, outputs


def _make_subgraph(nodes: list[fx.Node]) -> fx.Graph:
    placeholders, outputs = _find_key_nodes(nodes)

    new_graph = torch.fx.Graph()
    env: dict[fx.Node, fx.Node] = {}

    def env_lookup(x: fx.Node) -> fx.Node:
        assert x in env, f"Dependent node {x} not in env when creating downstream node"
        return env[x]

    def node_copy(
        node: fx.Node, arg_transform: Callable[[fx.Node], fx.Node]
    ) -> fx.Node:
        if node not in env:
            new_node = new_graph.node_copy(node, arg_transform=arg_transform)
            env[node] = new_node
        else:
            new_node = env[node]
        return new_node

    for node in placeholders:
        env[node] = new_graph.placeholder(node.name)

    for node in nodes:
        if node in placeholders:
            continue
        else:
            new_node = node_copy(node, env_lookup)
            new_node.meta = node.meta.copy()

    out_node = [env[x] for x in outputs]
    new_graph.output(out_node)
    return new_graph


def _is_root(stack: str) -> bool:
    return stack == ""


def make_graph_view(graph: fx.Graph) -> Container:
    """
    Code from: https://github.com/meta-pytorch/autoparallel/pull/158

    Make a graph view from the fx.Graph. This is a tree structure that
    represents the module hierarchy of the graph, and enables us to
    easily find the nodes that belong to each module, and gives a slightly
    easier way of visualize different parts of the graph by extracting
    subgraphs that belong to a particular module FQN.

    For example, if we have the following model with module hierarchy:

    Transformer(
        (tok_embeddings): Embedding(128256, 4096)
        (layers): ModuleDict(
            (0): TransformerBlock(
            (attention): Attention(
                (wq): Linear(in_features=4096, out_features=4096, bias=False)
                (wk): Linear(in_features=4096, out_features=1024, bias=False)
                (wv): Linear(in_features=4096, out_features=1024, bias=False)
                (wo): Linear(in_features=4096, out_features=4096, bias=False)
                (sdpa): ScaledDotProductAttention()
            )
            (feed_forward): FeedForward(
                (w1): Linear(in_features=4096, out_features=14336, bias=False)
                (w2): Linear(in_features=14336, out_features=4096, bias=False)
                (w3): Linear(in_features=4096, out_features=14336, bias=False)
            )
            (attention_norm): RMSNorm((4096,), eps=1e-05, elementwise_affine=True)
            (ffn_norm): RMSNorm((4096,), eps=1e-05, elementwise_affine=True)
            )
        )
        (norm): RMSNorm((4096,), eps=1e-05, elementwise_affine=True)
        (output): Linear(in_features=4096, out_features=128256, bias=False)
    )

    Then we can get a GraphView for the fx.Graph that enables us to do

    graph_view = make_graph_view(graph)
    subgraph = graph_view.layers["0"].attention.graph_view()

    where subgraph is a fx.Graph that contains all the nodes that belong to
    Transformer.layers['0'].attention, and whose inputs are all inputs to this
    region of the graph, and whose outputs are all outputs of this region of
    the graph. This returns a new graph with new nodes, so we shouldn't use it
    for graph manipulations, but it is useful to visualize what a particular
    part of a larger graph looks like.

    Additionally, you can also query the original nodes in that region with
    `graph_view.layers['0'].attention.data`, which returns a list of all the
    nodes that belong to Transformer.layers['0'].attention.
    """
    nodes: list[fx.Node] = list(graph.nodes)
    nodes_by_module_stack_root: Container | None = None
    for node in nodes:
        for module_stack, module_class in _get_module_stack(node):
            module_stack = _clean_stack_name(module_stack)
            nodes_by_module_stack: Container | None = nodes_by_module_stack_root
            for name in module_stack.split("."):
                if nodes_by_module_stack is None:
                    nodes_by_module_stack = Container(name, module_class)
                    nodes_by_module_stack_root = nodes_by_module_stack
                if _is_root(module_stack):
                    new_stack: Container = nodes_by_module_stack
                else:
                    new_stack = nodes_by_module_stack.get_child(name, module_class)
                nodes_by_module_stack = new_stack
                nodes_by_module_stack.add(node)

    assert nodes_by_module_stack_root is not None, "empty node in the graph"
    return nodes_by_module_stack_root


def decode_module(module_bucket_plans: list[str]) -> list[list[str] | str]:
    """
    Convert abbreviated FQNs to the actual FQNs.
    Currently, we support the decoding of these abbreviations:
    (1) layers.[0-2] -> [layers.0], [layers.1], [layers.2]
        (layers are split three separate buckets)
    (2) norm+output -> [norm, output]
        (norm and output are in one bucket)
    """
    full_plan: list[list[str] | str] = []
    for module_name in module_bucket_plans:
        if "+" in module_name:
            full_plan.append(module_name.split("+"))
            continue
        match = re.search(r"\[(\d+)-(\d+)\]", module_name)
        if not match:
            full_plan.append(module_name)
        else:
            start, end = map(int, match.groups())
            prefix = module_name[: match.start()]
            suffix = module_name[match.end() :]
            full_plan.extend([f"{prefix}{i}{suffix}" for i in range(start, end + 1)])
    return full_plan


def get_subgraph_by_path(
    graph_view: Container, paths: Union[str, list[str]]
) -> list[fx.Node]:
    """
    Get subgraph by path(s).
    Args:
        graph_view (object): Root graph view object.
        paths (str or list of str): Path(s) to subgraph.
    Returns:
        object: Subgraph object or node.
    """

    def get_node_by_path(node: Container, path: str) -> Container:
        for p in path.split("."):
            if p in node.children:
                node = node.children[p]
            else:
                return Container("", object)
        return node

    if isinstance(paths, list):
        nodes = list(
            itertools.chain.from_iterable(
                get_node_by_path(graph_view, p).data for p in paths
            )
        )
        return nodes
    else:
        node = get_node_by_path(graph_view, paths)
        return node.data


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

        for key, nodes in grouped_collectives.items():
            # resort by "FQNs"
            nodes_ = list(nodes)
            # fqns = [
            #     (
            #         list(node.meta["nn_module_stack"].values())[-2][0]
            #         if "nn_module_stack" in node.meta
            #         else list(node.meta["fwd_nn_module_stack"].values())[-2][0]
            #         # else ""  # only hit this for embedding layer, which is a 1-param bucket
            #     )
            #     for node in nodes
            # ]
            # idxs = sorted(range(len(fqns)), key=lambda i: fqns[i])
            # nodes_ = [nodes_[i] for i in idxs]
            # fqns = [fqns[i] for i in idxs]
            # print(fqns)
            self._bucket_group(nodes_)


class ManualOverlapScheduler(OverlapScheduler):
    """
    Scheduler that manual buckets and reorders collective nodes based on module_bucket_plans
    """

    def __init__(self, gm: fx.GraphModule, module_bucket_plans: list[list[str] | str]):
        super().__init__(
            gm,
            max_in_flight_gb=0.0,
            max_compute_pre_fetch=0,
            collective_bucketing=True,
            insert_overlap_deps=True,
            compute_overlap_multipler=0.0,
            max_coll_distance=0,
            custom_runtime_estimation=None,
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
        graph_view: Container = make_graph_view(self.graph)
        for module in self.module_bucket_plans:
            subgraph_view = get_subgraph_by_path(graph_view.children["Model"], module)
            self.nodes_in_subgraph.append(subgraph_view)


def manual_overlap_bucketing(
    gm: torch.fx.GraphModule,
    module_bucket_plans: list[str],
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
    module_bucket_plans = decode_module(module_bucket_plans)
    overlapped_gm = ManualOverlapScheduler(gm, module_bucket_plans).run()
    overlapped_gm.recompile()
    return overlapped_gm
