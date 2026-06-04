from __future__ import annotations

import itertools
import logging
import re
from typing import Any, TYPE_CHECKING

import torch.fx as fx  # noqa: TC001
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet


if TYPE_CHECKING:
    from collections.abc import Callable

log = logging.getLogger(__name__)


def _get_module_stack(node: fx.Node) -> list[tuple[str, type[Any]]]:
    nn_stack = node.meta.get("nn_module_stack", "")
    if nn_stack:
        return list(nn_stack.values())

    fwd_nn_stack = node.meta.get("fwd_nn_module_stack", "")
    if fwd_nn_stack:
        return list(fwd_nn_stack.values())

    return []


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


class GraphView:
    """
    A hierarchical class for organizing and managing torch.fx nodes by their module stack.

    This class provides a tree-like structure where each node in the hierarchy corresponds
    to a module or submodule in a traced FX graph. Each `GraphView` instance can hold a list
    of FX nodes (`self.data`) belonging to that module scope, maintain a unique set of nodes
    (`self.unique_nodes`), and manage its child containers (`self.children`).

    Attributes:
        name (str): The name of the module or container scope.
        klass (type[Any]): The class type associated with this module/container.
        data (list[fx.Node]): A list of FX graph nodes belonging to this module.
        unique_nodes (OrderedSet[fx.Node]): A deduplicated set of nodes to ensure no duplicates.
        children (dict[str, GraphView]): A mapping of child module names to their corresponding GraphView instances.
    """

    def __init__(self, name: str, klass: type[Any]) -> None:
        self.name: str = name
        self.klass: type[Any] = klass
        self.data: list[fx.Node] = []
        self.unique_nodes: OrderedSet[fx.Node] = OrderedSet()
        self.children: dict[str, GraphView] = {}

    def add(self, data: fx.Node) -> None:
        if data not in self.unique_nodes:
            self.data.append(data)
            self.unique_nodes.add(data)

    def get_child(self, module_stack: str, klass: type[Any] | None = None) -> GraphView:
        if module_stack not in self.children:
            new_stack = GraphView(module_stack, klass or self.klass)
            self.children[module_stack] = new_stack
        return self.children[module_stack]

    def __getitem__(self, name: str) -> GraphView:
        return self.children[name]

    def __getattr__(self, name: str) -> GraphView:
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


def _clean_stack_name(stack_name: str) -> str:
    """
    Clean up FX node's nn_module_stack metadata string to a dot-separated path.

    L['self'] is replaced with L as a fixed anchor for the root of the network.

    Examples:
        Input: "L['self']._modules['layers']['0']._modules['attention']"
        Output: "L.layers.0.attention"

        Input: "L['self'].networks.1.conv"
        Output: "L.networks.1.conv"
    """
    cleaned = re.sub(r"^L\['self'\]\.?", "", stack_name)
    parts = re.findall(r"\['([^']+)'\]", cleaned)
    suffix = ".".join(parts) if parts else cleaned
    return f"L.{suffix}" if suffix else "L"


def _is_root(stack: str) -> bool:
    return stack == ""


def _strip_instance_suffix(name: str) -> str:
    """Strip the _N uniqueness suffix added by FX to node names (e.g. convolution_1 -> convolution)."""
    return re.sub(r"_\d+$", "", name)


def _outermost_prefix(stack: Any) -> str:
    """Return the cleaned outermost (block-level) module path from an nn_module_stack."""
    return _clean_stack_name(next(iter(stack.values()))[0])


def get_fused_kernel_module_fqn(scheduler_nodes: Any) -> str | None:
    """
    Return a human-readable FQN annotation for a fused kernel.

    Uses V.graph.fx_fqn_map — built once during lowering in graph.py:run_node —
    to map FX node names to FQN strings, e.g.
    "convolution_1" -> "L.networks.1.conv.convolution".

    Two-pass hybrid algorithm:

    Pass 1 — anchor prefixes via origin_node:
        For each snode, origin_node is the direct FX node whose run_node
        call created the IR buffer.  It is never transitively accumulated,
        so it gives an unambiguous block identity.  Look it up in fqn_map
        to get the anchor FQN, then read its nn_module_stack outermost
        entry to get the block-level prefix (e.g. "L.networks.3").

    Pass 2 — collect FQNs from origins with prefix filter:
        Walk all origins across every snode.  origins is transitively
        accumulated (cascading history from upstream blocks), but each
        entry is looked up in fqn_map and filtered by the anchor prefix.
        This correctly captures inline ops (relu, add) that share the
        same block but were inlined into a parent buffer and never became
        their own snodes, while excluding cascaded history from upstream
        blocks.
    """
    fqn_map: dict[str, str] = getattr(V.graph, "fx_fqn_map", {})
    log.debug(
        "[fqn_trace] get_fused_kernel_module_fqn entry: fqn_map size=%d snodes=%d",
        len(fqn_map),
        len(scheduler_nodes),
    )

    # Pass 1: derive block anchor prefixes from each snode's origin_node.
    # origin_node is direct (not accumulated), giving clean block identity.
    # Fallback: when origin_node is absent or a placeholder (not in fqn_map),
    # walk origins to find the first non-placeholder op in fqn_map.
    anchor_prefixes: OrderedSet[str] = OrderedSet()
    for snode in scheduler_nodes:
        if snode.node is None:
            continue
        buf_name = getattr(snode.node, "name", None)
        origin = snode.node.get_origin_node()
        origin_name = origin.name if origin is not None else None
        anchor_fqn = fqn_map.get(origin_name) if origin_name else None
        log.debug(
            "[fqn_trace] pass1 snode=%s buf_name=%s origin_node=%s op=%s in_fqn_map=%s",
            snode, buf_name, origin_name,
            origin.op if origin is not None else "None",
            anchor_fqn is not None,
        )

        if anchor_fqn:
            stack = origin.meta.get("nn_module_stack")
            prefix = _outermost_prefix(stack) if stack else None
            log.debug(
                "[fqn_trace] pass1 snode=%s buf_name=%s origin_node=%s op=%s "
                "anchor_fqn=%s outermost_prefix=%s",
                snode, buf_name, origin_name, origin.op, anchor_fqn, prefix,
            )
            if prefix:
                anchor_prefixes.add(prefix)
            continue

        # origin_node absent or not in fqn_map.
        # For placeholders: scan FX consumers (users) — they identify which
        # block uses this parameter, not the upstream producers in origins.
        # For None origin_node: fall back to scanning origins.
        if origin is not None and origin.op == "placeholder":
            fallback_source = origin.users
            fallback_kind = "user"
        else:
            fallback_source = snode.node.origins
            fallback_kind = "origin"
        log.debug(
            "[fqn_trace] pass1 snode=%s buf_name=%s origin_node=%s op=%s "
            "not in fqn_map — scanning %ss for fallback anchor",
            snode, buf_name, origin_name,
            origin.op if origin is not None else "None",
            fallback_kind,
        )
        for fx_node in fallback_source:
            fallback_fqn = fqn_map.get(fx_node.name)
            if fallback_fqn:
                stack = fx_node.meta.get("nn_module_stack")
                prefix = _outermost_prefix(stack) if stack else None
                log.debug(
                    "[fqn_trace] pass1 snode=%s buf_name=%s fallback_%s=%s op=%s "
                    "fallback_fqn=%s outermost_prefix=%s",
                    snode, buf_name, fallback_kind, fx_node.name, fx_node.op,
                    fallback_fqn, prefix,
                )
                if prefix:
                    anchor_prefixes.add(prefix)
                break
        else:
            log.debug(
                "[fqn_trace] pass1 snode=%s buf_name=%s no anchor found in %ss",
                snode, buf_name, fallback_kind,
            )

    log.debug("[fqn_trace] pass1 complete: anchor_prefixes=%s", list(anchor_prefixes))

    if not anchor_prefixes:
        log.debug("[fqn_trace] get_fused_kernel_module_fqn: result=None (no anchors)")
        return None

    # Pass 2: walk all origins across every snode (the transitively accumulated
    # set), look each up in fqn_map, and include only those whose FQN prefix
    # matches an anchor.  This captures inline ops (e.g. relu, add inlined into
    # a parent buffer) while rejecting cascaded history from upstream blocks.
    extern_fqns: set[str] = getattr(V.graph, "fx_extern_fqns", set())
    module_names: OrderedSet[str] = OrderedSet()
    for snode in scheduler_nodes:
        if snode.node is None:
            continue
        buf_name = getattr(snode.node, "name", None)
        for fx_node in snode.node.origins:
            fqn = fqn_map.get(fx_node.name)
            if not fqn:
                log.debug(
                    "[fqn_trace] pass2 snode=%s buf_name=%s fx_node=%s op=%s "
                    "skipped (not in fqn_map)",
                    snode, buf_name, fx_node.name, fx_node.op,
                )
                continue
            if fqn in extern_fqns:
                log.debug(
                    "[fqn_trace] pass2 snode=%s buf_name=%s fx_node=%s "
                    "fqn=%s skipped (claimed by extern kernel)",
                    snode, buf_name, fx_node.name, fqn,
                )
                continue
            if not any(fqn == p or fqn.startswith(p + ".") for p in anchor_prefixes):
                log.debug(
                    "[fqn_trace] pass2 snode=%s buf_name=%s fx_node=%s "
                    "fqn=%s excluded (prefix not in anchors=%s)",
                    snode, buf_name, fx_node.name, fqn, list(anchor_prefixes),
                )
                continue
            log.debug(
                "[fqn_trace] pass2 snode=%s buf_name=%s fx_node=%s fqn=%s included",
                snode, buf_name, fx_node.name, fqn,
            )
            module_names.add(fqn)

    result = " + ".join(module_names) if module_names else None
    log.debug("[fqn_trace] get_fused_kernel_module_fqn: result=%s", result)
    return result


def make_graph_view(
    graph: fx.Graph,
    module_stack_fn: Callable[[fx.Node], list[tuple[str, type[Any]]]] | None = None,
) -> GraphView | None:
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
    subgraph = get_subgraph_by_path(graph_view, "layers.0")

    where subgraph contains all the nodes that belong to this region

    module_stack_fn: Optional callable for extracting module hierarchy information from nodes.

        Signature: Callable[[fx.Node], list[tuple[str, type[Any]]]]

        Takes an FX node and returns a list of (module_path, module_class) tuples representing
        the nested module hierarchy for that node, ordered from outermost to innermost scope.

        - module_path (str): Dot-separated path identifying the module in the hierarchy
          (e.g., "layers.0.attention.wq")
        - module_class (type): The Python class type of the module

        This enables custom logic for determining module membership, useful for:
        - Graphs without standard nn_module_stack metadata
        - Filtering or grouping nodes by custom criteria

        Example of getting the module stack from annotation:

        def module_stack_fn(node):
            module_stack = node.meta.get("custom", {}).get("module_path", "")
            return [(module_stack, torch.nn.Module)]

        If None, defaults to extracting from node.meta["nn_module_stack"] or
        node.meta["fwd_nn_module_stack"].
    """

    def nn_module_stack_meta(node: fx.Node) -> list[tuple[str, type[Any]]]:
        result = []
        for module_stack, module_class in _get_module_stack(node):
            module_stack = _clean_stack_name(module_stack)
            result.append((module_stack, module_class))
        return result

    if module_stack_fn is None:
        module_stack_fn = nn_module_stack_meta
    nodes: list[fx.Node] = list(graph.nodes)
    nodes_by_module_stack_root: GraphView | None = None
    for node in nodes:
        for module_stack, module_class in module_stack_fn(node):
            nodes_by_module_stack: GraphView | None = nodes_by_module_stack_root
            for name in module_stack.split("."):
                if nodes_by_module_stack is None:
                    nodes_by_module_stack = GraphView(name, module_class)
                    nodes_by_module_stack_root = nodes_by_module_stack
                if _is_root(module_stack):
                    new_stack: GraphView = nodes_by_module_stack
                else:
                    new_stack = nodes_by_module_stack.get_child(name, module_class)
                nodes_by_module_stack = new_stack
                nodes_by_module_stack.add(node)

    return nodes_by_module_stack_root


def get_subgraph_by_path(
    graph_view: GraphView, paths: str | list[str]
) -> list[fx.Node]:
    """
    Get subgraph by path(s).
    Args:
        graph_view (object): Root graph view object.
        paths (str or list of str): Path(s) to subgraph.
    Returns:
        list[fx.Node]: fx nodes belong to the subgraph
    """

    def get_node_by_path(node: GraphView, path: str) -> GraphView:
        for p in path.split("."):
            if p in node.children:
                node = node.children[p]
            else:
                return GraphView("", object)
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
