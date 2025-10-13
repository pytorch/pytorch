import re
import itertools
from collections import defaultdict

import torch
import torch.fx as fx

from torch._inductor.fx_passes.overlap_preserving_bucketer import OverlapPreservingBucketer
from torch._inductor.fx_passes.overlap_scheduling import OverlapScheduler, is_compute_node
from torch._inductor.fx_passes.control_dependencies import preserve_node_ordering
from torch._inductor.fx_passes.bucketing import (
    is_all_gather_into_tensor as is_all_gather,
    is_reduce_scatter_tensor as is_reduce_scatter,
    is_wait_tensor,
    merge_all_gather_bucket,
    merge_reduce_scatter_bucket,
)
from torch.utils._ordered_set import OrderedSet
from torch._dynamo.graph_deduplication import _stable_topological_sort


def _get_module_stack(node):
    if "nn_module_stack" not in node.meta:
        if 'fwd_nn_module_stack' in node.meta:
           return list(node.meta['fwd_nn_module_stack'].values())
        return []
    return list(node.meta["nn_module_stack"].values())


def _addindent(s_, num_spaces):
    s = s_.split("\n")
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(num_spaces * " ") + line for line in s]
    s = "\n".join(s)
    s = first + "\n" + s
    return s


class Container:
    def __init__(self, name, klass):
        self.name = name
        self.klass = klass
        self.data = []
        self.unique_nodes = set()
        # self.children = defaultdict(Container)
        self.children = {}

    def add(self, data):
        if data not in self.unique_nodes:
            self.data.append(data)
            self.unique_nodes.add(data)

    def get_child(self, module_stack, klass=None):
        if module_stack not in self.children:
            new_stack = Container(module_stack, klass)
            self.children[module_stack] = new_stack
        return self.children[module_stack]

    def __getitem__(self, name):
        return self.children[name]

    def __getattr__(self, name):
        return self.children[name]

    def __repr__(self):
        child_lines = []
        for name, child in self.children.items():
            mod_str = repr(child)
            mod_str = _addindent(mod_str, 2)
            child_lines.append("(" + name + "): " + mod_str)
        main_str = self.klass.__name__ + "("
        if child_lines:
            main_str += "\n  " + "\n  ".join(child_lines) + "\n"
        main_str += ")"
        return main_str

    def graph_view(self):
        return _make_subgraph(self.data)


def _clean_stack_name(val):
    # TODO: is this still needed?
    name = (
        val.replace("L['self']", "Model")
        .replace("_modules['", "")
        .replace("['", ".")
        .replace("']", "")
    )
    return name


def _find_key_nodes(nodes):
    root = []
    outputs = []
    nodes_set = set(nodes)
    for node in nodes:
        for x in node.all_input_nodes:
            if x not in nodes_set:
                root.append(x)
        if all(x not in nodes_set for x in node.users):
            outputs.append(node)
    return root, outputs


def _make_subgraph(nodes):
    placeholders, outputs = _find_key_nodes(nodes)

    new_graph = torch.fx.Graph()
    env = {}

    # pyre-ignore
    def env_lookup(x: torch.fx.Node) -> torch.fx.Node:
        assert x in env, f"Dependent node {x} not in env when creating downstream node"
        return env[x]

    # pyre-ignore
    def node_copy(node, arg_transform) -> torch.fx.Node:
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


def _is_root(stack):
    return stack == ""


def make_graph_view(graph):
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

    >>> graph_view = make_graph_view(graph)
    >>> subgraph = graph_view.layers['0'].attention.graph_view()

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
    nodes = list(graph.nodes)
    nodes_by_module_stack_root = None
    last_module_name = None
    for node in nodes:
        # TODO: handle cases where there is no module stack (i.e., loop is empty and node is not added)
        for module_stack, module_class in _get_module_stack(node):
            module_stack = _clean_stack_name(module_stack)
            nodes_by_module_stack = nodes_by_module_stack_root
            for name in module_stack.split("."):
                if nodes_by_module_stack is None:
                    nodes_by_module_stack = Container(name, module_class)
                    nodes_by_module_stack_root = nodes_by_module_stack
                if _is_root(module_stack):
                    new_stack = nodes_by_module_stack
                else:
                    new_stack = nodes_by_module_stack.get_child(name, module_class)
                nodes_by_module_stack = new_stack
                nodes_by_module_stack.add(node)

    return nodes_by_module_stack_root


def decode_module(bucketing_plan: list[str]) -> list[str]:
    """
    Convert user defined FQNs to the actual list of manuals to perform bucketing.
    """
    full_plan = []
    for module_name in bucketing_plan:
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


def get_subgraph_by_path(graph_view, paths):
    """
    Get subgraph by path(s).
    Args:
        graph_view (object): Root graph view object.
        paths (str or list of str): Path(s) to subgraph.
    Returns:
        object: Subgraph object or node.
    """
    def get_node_by_path(node, path):
        for p in path.split("."):
            if p in node.children:
                node = node.children[p]
            else:
                return Container("", "")
        return node
    if isinstance(paths, list):
        nodes = list(itertools.chain.from_iterable(
            get_node_by_path(graph_view, p).data for p in paths
        ))
        return nodes
    else:
        node = get_node_by_path(graph_view, paths)
        return node.data


class ManualOverlapPreservingBucketer(OverlapPreservingBucketer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def bucket_collectives(self, nodes) -> None:
        """
        Manually bucket all rs/ag/wait nodes from self.manual_nodes into one bucket.
        """

        from torch._inductor.fx_passes.bucketing import (
            merge_all_gather_bucket,
            merge_reduce_scatter_bucket,
        )

        # Filter out valid collective *starts*
        collectives = [n for n in nodes if n in self.collective_info]
        if collectives is []:
            return

        # Partition into groups by type
        ag_nodes = [n for n in collectives if is_all_gather(n)]
        rs_nodes = [n for n in collectives if is_reduce_scatter(n)]

        def _bucket_group(coll_nodes: list[fx.Node]):
            if not coll_nodes:
                return {}

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
            else:
                new_nodes, replacements = merge_reduce_scatter_bucket(
                    self.graph,
                    coll_nodes,
                    wait_insertion_point=first_wait,
                    insert_before=next_node,
                    mode="custom_ops",
                )

            # Identify the new wait and start
            new_waits = [n for n in new_nodes if is_wait_tensor(n)]
            assert len(new_waits) == 1, f"Expected exactly one new wait, got {new_waits}"
            new_wait = new_waits[0]
            new_start = new_wait.args[0]
            assert isinstance(new_start, fx.Node)

            # Build overlap dependency constraints
            overlap_deps: dict[fx.Node, OrderedSet[fx.Node]] = defaultdict(OrderedSet)
            for coll in coll_nodes:
                info = self.collective_info[coll]
                if info.hiding_node and not info.is_exposed:
                    overlap_deps[info.hiding_node].add(new_start)
                    overlap_deps[new_wait].add(info.hiding_node)
            return overlap_deps

        # Bucket each type group
        deps_ag = _bucket_group(ag_nodes)
        deps_rs = _bucket_group(rs_nodes)

        # Merge dependency dicts
        combined_deps: dict[fx.Node, OrderedSet[fx.Node]] = defaultdict(OrderedSet)
        for d in (deps_ag, deps_rs):
            for k, v in d.items():
                combined_deps[k].update(v)

        # Re-toposort and preserve ordering
        _stable_topological_sort(self.graph, combined_deps)
        self._preserve_dependencies_with_tokens(combined_deps)

        self.graph.lint()

class ManualOverlapScheduler(OverlapScheduler):
    """
    Scheduler that manual reorders and buckets collective nodes based on user specifications (nodes_in_subgraph)
    """
    def __init__(self, gm: fx.GraphModule, nodes_in_subgraph: list[fx.Node]):
        super().__init__(gm)  # use default parameters from OverlapScheduler
        self.nodes_in_subgraph = nodes_in_subgraph
        for node in list(self.graph.nodes):
            self._schedule(node)

    def run(self) -> torch.fx.GraphModule:
        """Run the manual scheduling from nodes_in_subgraph"""
        # Bucket collective
        if torch._inductor.config.test_configs.aten_fx_overlap_preserving_bucketing:
            self._manual_bucket_collectives()
        elif torch._inductor.config.test_configs.aten_fx_overlap_insert_overlap_deps:
            # If not bucketing, add effect tokens to preserve hiding dependencies
            self._add_effect_tokens_for_overlap()

        # Reorder collectives
        self._manual_reorder_graph()

        return self.gm

    def _manual_reorder_graph(self) -> None:
        deps = defaultdict(OrderedSet)
        num_subgraphs = len(self.nodes_in_subgraph)
        for i in range(num_subgraphs):
            sub_nodes = self.nodes_in_subgraph[i]
            all_gather_start = next((n for n in sub_nodes if "all_gather" in n.name), None)
            all_gather_wait = next((n for n in sub_nodes if "wait" in n.name and "gather" in n.name), None)
            reduce_scatter_start = next((n for n in sub_nodes if "reduce_scatter" in n.name), None)
            reduce_scatter_wait = next((n for n in sub_nodes if "wait" in n.name and "scatter" in n.name), None)

            # enforce all_gather_i after all_gather_wait_{i-1}
            if i > 0:
                prev_nodes = self.nodes_in_subgraph[i - 1]
                prev_all_gather_wait = next(
                    (n for n in prev_nodes if "wait" in n.name and "gather" in n.name),
                    None,
                )
                if all_gather_start and prev_all_gather_wait:
                    deps[all_gather_start].add(prev_all_gather_wait)

            # enforce reduce_scatter_wait_i before reduce_scatter_{i+1}
            if i < num_subgraphs - 1:
                next_nodes = self.nodes_in_subgraph[i + 1]
                next_reduce_scatter_start = next(
                    (n for n in next_nodes if "reduce_scatter" in n.name),
                    None,
                )
                if reduce_scatter_wait and next_reduce_scatter_start:
                    deps[next_reduce_scatter_start].add(reduce_scatter_wait)

        if deps:
            preserve_node_ordering(self.graph, deps)
            self.graph.lint()

    def _manual_bucket_collectives(self) -> None:
        bucketer = ManualOverlapPreservingBucketer(
            graph=self.graph,
            collective_info=self.collective_info,
            node_ancestors=self.node_ancestors,
            scheduled=self.scheduled,
            max_bucket_memory_gb=1.0,
            max_coll_distance=self.max_node_distance,
        )
        for i, nodes in enumerate(self.nodes_in_subgraph):
            bucketer.bucket_collectives(nodes=nodes)


def manual_overlap_bucketing(
    gm: torch.fx.GraphModule,
    buckted_module: list[str],
) -> torch.fx.GraphModule:
    """Schedule nodes based on user specifications in buckted_module

    Args:
        gm: Input graph module to optimize.
        buckted_module: user specified plans
    """
    ## Step 1: Get the subgraphs from each bucketed module
    buckted_module = decode_module(buckted_module)
    graph_view = make_graph_view(gm.graph)
    nodes_in_subgraph = []
    for module in buckted_module:
        subgraph_view = get_subgraph_by_path(graph_view.children["Model"], module)
        nodes_in_subgraph.append(subgraph_view)

    ## Step 2: Do bucketing and reordering over each subgraph
    return ManualOverlapScheduler(
        gm, nodes_in_subgraph
    ).run().recompile()
