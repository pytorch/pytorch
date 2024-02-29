import copy
import itertools
import logging
from typing import Any, Dict, List, Set

import onnx


def build_uses_and_sources(onnx_graph: "onnx.GraphProto"):  # type: ignore[name-defined]
    """Build a dictionary of uses and sources for each node in the model.

    Consider a graph,
      x -> node0 -> y -> node1 -> z
    , where x, y, and z are tensor names, and node0 and node1 are node names.
    Its uses dictionary and sources dictionary respectively are:
      uses = {"x": [node0], "y": [node1]}
      sources = {"y": node0, "z": node1}
    Feeding a graph to this function returns the corresponding uses and
    sources dictionaries.
    """

    # If uses = {"x": [node0, node1], "y": [node2], ...},
    # "x" is used by "node0" and "node1", and "y" is used by "node2".
    uses: Dict[str, List["onnx.NodeProto"]] = {}  # type: ignore[name-defined]
    # If sources={"a": node0, "b": node1, ...},
    # "a" is produced by "node0", and "b" is produced by "node1".
    # Model inputs and initializers do not have sources.
    sources: Dict[str, "onnx.NodeProto"] = {}  # type: ignore[name-defined]
    for node in onnx_graph.node:
        # FIXME: If and Loop are not supported.
        if node.op_type in {"If", "Loop"}:
            logger.warning(
                f"Sub-graphs in node (name={node.name}, op_type={node.op_type}) is "
                "ignored in uses and sources."
            )
        for input_name in node.input:
            if input_name not in uses:
                uses[input_name] = []
            # Value called `input_name` is used by node called `node.name`.
            uses[input_name].append(node)
        for output_name in node.output:
            assert output_name not in sources, (
                f"Output name {output_name} cannot have multiple sources. "
                f"Existing source of {output_name} is {sources[output_name]}, "
                f"and we find another source {node}."
            )
            sources[output_name] = node
    return uses, sources


def build_upstream_nodes_and_downstream_nodes(onnx_graph: "onnx.GraphProto"):  # type: ignore[name-defined]
    """Build upstream nodes and downstream nodes for each node in the model.

    Consider a graph,
      w (graph input) -> node0 -> x -> node1 -> y -> node2 -> z (graph output)
    , where w, x, y, and z are tensor names, and node0, node1, and node 2
    are node names. Its upstream nodes and downstream nodes respectively are:
      upstream_nodes = {"node0": [], "node1": [node0], "node2": [node1]}
      downstream_nodes = {"node0": [node1], "node1": [node2], "node2": []}
    Feeding a graph to this function returns the corresponding upstream nodes and
    downstream nodes dictionaries.
    """

    uses, sources = build_uses_and_sources(onnx_graph)
    # If upstream_nodes = {node2: [node0, node1], node3: [node0, node2], ...},
    # node2 consumes some of node0's and node1's outputs, and
    # node3 consumes some of node0's and node2's outputs.
    upstream_nodes: Dict[str, List["onnx.NodeProto"]] = {}  # type: ignore[name-defined]
    # If downstream_nodes = {node0: [node1, node2], node1: [node3, node4], ...},
    # node0's output is consumed by node1 and node2, and
    # node1's output is consumed by node3 and node4.
    downstream_nodes: Dict[str, List["onnx.NodeProto"]] = {}  # type: ignore[name-defined]

    for node in onnx_graph.node:
        upstream_nodes_ = []
        for input_name in node.input:
            if input_name in sources:
                upstream_nodes_.append(sources[input_name])
        downstream_nodes_ = []
        for output_name in node.output:
            if output_name in uses:
                downstream_nodes_.extend(uses[output_name])
        upstream_nodes[node.name] = upstream_nodes_
        downstream_nodes[node.name] = downstream_nodes_

    return upstream_nodes, downstream_nodes


def search_for_root(onnx_graph: "onnx.GraphProto"):  # type: ignore[name-defined]
    """Search for the root node of the graph.

    For a graph like
      w -> node0 -> x -> node1 -> y -> node2 -> z,
    where w, x, y, and z are tensor names, and node0, node1, and node 2
    are node names. The root node is node0. Generally speaking,
    the root node is a node that consumes the graph's inputs or initializers.
    """
    _, sources = build_uses_and_sources(onnx_graph)
    # Find the root node of the graph
    # A root node is a node consumes the graph's inputs or initializers;
    # that is, some of the node's inputs don't have source node.

    root_nodes = []
    for node in onnx_graph.node:
        if all(input_name not in sources for input_name in node.input):
            # If some node's input doesn't have source node, then
            # that input must be a graph input or initializer. So, the node
            # is a root node.
            root_nodes.append(node)

    return root_nodes


logger = logging.getLogger(__name__)
# Uncomment the following lines to print out development info.
# logging.basicConfig(level=logging.DEBUG)
# logger.setLevel(logging.DEBUG)


def search_for_pattern(
    main_graph: "onnx.GraphProto",  # type: ignore[name-defined]
    pattern_graph: "onnx.GraphProto",  # type: ignore[name-defined]
    skipped_main_node_names: Set[str],
    pattern_upstreams_nodes: Dict[str, List["onnx.NodeProto"]],  # type: ignore[name-defined]
    pattern_downstream_nodes: Dict[str, List["onnx.NodeProto"]],  # type: ignore[name-defined]
    main_upstreams_nodes: Dict[str, List["onnx.NodeProto"]],  # type: ignore[name-defined]
    main_downstream_nodes: Dict[str, List["onnx.NodeProto"]],  # type: ignore[name-defined]
):
    """Search for a pattern in the main graph.

    Arguments:
    - main_graph: The main graph to search for the pattern.
    - pattern_graph: The pattern graph to search for in the main graph.
    - skipped_main_node_names: A set of node names in the main graph that
      should not be used to match the pattern. This is used for avoiding
      matching nodes that have been fused into another sub-graph.
    - pattern_upstreams_nodes: Upstream nodes for each node in the pattern graph.
      This is the 1st output of `build_upstream_nodes_and_downstream_nodes(pattern_graph)`.
    - pattern_downstream_nodes: Downstream nodes for each node in the pattern graph.
      This is the 2nd output of `build_upstream_nodes_and_downstream_nodes(pattern_graph)`.
    - main_upstreams_nodes: Upstream nodes for each node in the main graph.
      This is the 1st output of `build_upstream_nodes_and_downstream_nodes(main_graph)`.
    - main_downstream_nodes: Downstream nodes for each node in the main graph.
      This is the 2nd output of `build_upstream_nodes_and_downstream_nodes(main_graph)`.

    Assume main graph is
      w -> node0 (op_type: Relu) -> x -> node1 (op_type: Relu) -> y -> node2 (op_type: Neg) -> z
      w1 -> node3 (op_type: Relu) -> x1 -> node4 (op_type: Relu) -> y1 -> node5 (op_type: Neg) -> z1
    and pattern graph is
      a -> nodea (op_type: Relu) -> b -> nodeb (op_type: Relu) -> c
    To encode the matched sub-graphs, this function returns a pair of dictionaries
      pattern_to_main_node_bindings = [
        {
          "nodea": "node0",
          "nodeb": "node1",
        },
        {
          "nodea": "node3",
          "nodeb": "node4",
        },
      }
    and
      pattern_to_main_value_bindings = [
        {
          "a": "w",
          "b": "x",
          "c": "y",
        },
        {
          "a": "w1",
          "b": "x1",
          "c": "y1",
        },
      }

    """
    pattern_root_nodes = search_for_root(pattern_graph)

    # This loop tries all possible combinations roots.
    # 1. First, some nodes are selected from the main graph as root nodes.
    # 2. Then, match them to the root nodes in the pattern graph.
    # 3. If the root nodes are matched, then we try to match all downstream nodes.
    # 4. If all downstream nodes are matched, then we have found a match.
    # 5. Go back to step 1 and try another combination of root nodes.
    for selected_onnx_graph_indices in itertools.permutations(
        range(len(main_graph.node)), len(pattern_root_nodes)
    ):
        main_root_nodes = [main_graph.node[i] for i in selected_onnx_graph_indices]

        if any(node.name in skipped_main_node_names for node in main_root_nodes):
            continue

        if not all(
            node.op_type == another_node.op_type
            for node, another_node in zip(
                main_root_nodes,
                pattern_root_nodes,
            )
        ):
            continue

        # Initialize the queue of traversed nodes
        # with the root nodes.
        # 1. Downstream nodes of the roots nodes will be
        #    pushed into this queue for implementing BFS
        #    (breadth first search).
        # 2. nodes in queue should have been matched/bound.
        queue = list(
            (node, pattern_node)
            for node, pattern_node in zip(main_root_nodes, pattern_root_nodes)
        )

        # Record the bindings from pattern nodes to main nodes.
        pattern_to_main_node_bindings = {
            pattern_node.name: node.name
            for node, pattern_node in zip(main_root_nodes, pattern_root_nodes)
        }

        fail = False
        # {"main_graph_X", "pattern_graph_X"} means that "main_graph_X" in `main_graph`
        # is bound/matched to "pattern_graph_X" in `pattern_graph`.
        pattern_to_main_value_bindings: Dict[str, str] = {}

        def try_bind_values(bindings, node, pattern_node):
            # FIXME: examine type and shape bindings for graph inputs and outputs
            # We need something like try_bind_values(
            #   bindings,
            #   type_bindings,
            #   shape_bindings,
            #   node,
            #   pattern_node,
            #   main_graph,
            #   pattern_graph
            # )
            # If a variable in main_graph "x_main" is bound to pattern_graph "x",
            # then we examin if they have the same type in the corresponding value_info's.
            # Similarly, shapes should checked too.
            # FIXME: bind values in sub-graphs when node and pattern_node are If or Loop.

            for name, pattern_name in zip(node.input, pattern_node.input):
                if pattern_name in bindings and bindings[pattern_name] != name:
                    logger.debug(
                        "Binding fail because new pattern-to-main pair "
                        f"({pattern_name}, {name}) conflicts with existing binding "
                        f"({pattern_name}, {bindings[pattern_name]})."
                    )
                    return False
                elif pattern_name in bindings and bindings[pattern_name] == name:
                    pass
                else:
                    # Now, the value called `pattern_name` in `pattern_graph` is matched
                    # to the value called `name` in `main_graph`.
                    logger.debug(
                        f"Binding succeeds for pattern-to-main pair ({pattern_name}, {name})."
                    )
                    bindings[pattern_name] = name

            for name, pattern_name in zip(node.output, pattern_node.output):
                if pattern_name in bindings and bindings[pattern_name] != name:
                    logger.debug(
                        "Binding fails because new pattern-to-main pair "
                        f"({pattern_name}, {name}) conflicts with "
                        f"existing binding ({pattern_name}, {bindings[pattern_name]})."
                    )
                    return False
                elif pattern_name in bindings and bindings[pattern_name] == name:
                    pass
                else:
                    # Now, the value called `pattern_name` in `pattern_graph` is matched
                    # to the value called `name` in `main_graph`.
                    logger.debug(
                        f"Binding succeeds for pattern-to-main pair ({pattern_name}, {name})."
                    )
                    bindings[pattern_name] = name

            return True

        for node, pattern_node in zip(main_root_nodes, pattern_root_nodes):
            if not try_bind_values(pattern_to_main_value_bindings, node, pattern_node):
                fail = True
                break

        # binding node to pattern_node fails because it conflicts
        # with existing value name bindings.
        # Let's stop here and try another set of root nodes.
        if fail:
            continue

        while queue:
            main_node, pattern_nodes = queue.pop()

            logger.debug(
                "Current pattern_to_main_node_bindings: ", pattern_to_main_node_bindings
            )

            to_queue = []
            traversed_names = []

            if len(main_downstream_nodes[main_node.name]) != len(
                pattern_downstream_nodes[pattern_nodes.name]
            ):
                break

            # flag to enable early stop if binding fails
            # so that we can skip the rest of the nodes in the queue.
            fail = False
            for main_downstream_node, pattern_downstream_node in zip(
                main_downstream_nodes[main_node.name],
                pattern_downstream_nodes[pattern_nodes.name],
            ):
                logger.debug(
                    f"Try binding main graph node {main_downstream_node.name} "
                    f"to pattern node {pattern_downstream_node.name}"
                )
                if main_downstream_node.name in skipped_main_node_names:
                    logger.debug(
                        f"Binding fail because main graph node "
                        f"{main_downstream_node.name} should be skipped "
                        "(e.g., when a node is already fused into another sub-graph)."
                    )
                    fail = True
                    break
                if main_downstream_node.op_type != pattern_downstream_node.op_type:
                    logger.debug(
                        "Binding fail because different op_types "
                        f"{main_downstream_node.op_type} and "
                        f"{pattern_downstream_node.op_type}"
                    )
                    fail = True
                    break
                # TODO: check Constant node's value
                if (
                    pattern_downstream_node.name in pattern_to_main_node_bindings
                    and pattern_to_main_node_bindings[pattern_downstream_node.name]
                    != main_downstream_node.name
                ):
                    logger.debug(
                        "Binding fail because new pattern-to-main pair "
                        f"({pattern_downstream_node.name}, {main_downstream_node.name}) "
                        "conflicts with existing binding ("
                        f"{pattern_downstream_node.name}, "
                        f"{pattern_to_main_node_bindings[pattern_downstream_node.name]}"
                        ")"
                    )
                    fail = True
                    break
                if not try_bind_values(
                    pattern_to_main_value_bindings,
                    main_downstream_node,
                    pattern_downstream_node,
                ):
                    logger.debug(
                        "Binding fail because value bindings conflict for "
                        f"{main_downstream_node.name} and {pattern_downstream_node.name}"
                    )
                    fail = True
                    break
                logger.debug("Binding success")
                to_queue.append((main_downstream_node, pattern_downstream_node))
                traversed_names.append(pattern_downstream_node.name)

            if fail:
                break

            for node, pattern_node in to_queue:
                pattern_to_main_node_bindings[pattern_node.name] = node.name

            queue.extend(to_queue)

        if fail:
            logger.debug("Binding fail when binding intermediate nodes.")
            continue

        logger.debug(
            "Node matches (pattern node name -> node name in model graph to fuse) to fuse: "
        )
        for pattern_node, main_node in pattern_to_main_node_bindings.items():
            logger.debug(f"{pattern_node} -> {main_node}")

        return pattern_to_main_node_bindings, pattern_to_main_value_bindings
    return None, None


def search_for_patterns(main_graph, pattern_graph):
    skipped_main_node_names: Set[str] = set()
    matches = []

    (
        pattern_upstreams_nodes,
        pattern_downstream_nodes,
    ) = build_upstream_nodes_and_downstream_nodes(pattern_graph)
    (
        main_upstreams_nodes,
        main_downstream_nodes,
    ) = build_upstream_nodes_and_downstream_nodes(main_graph)

    while True:
        (
            pattern_to_main_node_bindings,
            pattern_to_main_value_bindings,
        ) = search_for_pattern(
            main_graph=main_graph,
            pattern_graph=pattern_graph,
            skipped_main_node_names=skipped_main_node_names,
            pattern_upstreams_nodes=pattern_upstreams_nodes,
            pattern_downstream_nodes=pattern_downstream_nodes,
            main_upstreams_nodes=main_upstreams_nodes,
            main_downstream_nodes=main_downstream_nodes,
        )
        if (
            pattern_to_main_node_bindings is not None
            and pattern_to_main_value_bindings is not None
        ):
            skipped_main_node_names.update(pattern_to_main_node_bindings.values())
            matches.append(
                (pattern_to_main_node_bindings, pattern_to_main_value_bindings)
            )
        else:
            break

    return matches


def fuse(
    graph,
    pattern,
    fusion_node,
    pattern_to_main_node_bindings,
    pattern_to_main_value_bindings,
):
    main_node_names = set(value for value in pattern_to_main_node_bindings.values())
    main_value_names = set(value for value in pattern_to_main_value_bindings.values())

    main_node_indices = []
    for i, node in enumerate(graph.node):
        if node.name in main_node_names:
            main_node_indices.append(i)

    graph.node.insert(main_node_indices[-1] + 1, fusion_node)

    deleted_output_names = set()
    for i in sorted(main_node_indices, reverse=True):
        deleted_output_names.update(graph.node[i].output)
        del graph.node[i]
    value_infos = [
        value_info
        for value_info in graph.value_info
        if value_info.name not in deleted_output_names
        or value_info.name in main_value_names
    ]
    del graph.value_info[:]

    graph.value_info.extend(value_infos)


def apply_fusion(
    model: "onnx.ModelProto",  # type: ignore[name-defined]
    pattern_graph: "onnx.GraphProto",  # type: ignore[name-defined]
    fusion_node_type: str,
    fusion_node_attributes: Dict[str, Any],
    fusion_node_domain: str,
    fusion_node_version: int,
):
    logger.debug(f"Applying fusion for {fusion_node_type}")
    matches = search_for_patterns(model.graph, pattern_graph)

    if len(matches) > 0 and all(
        opset.domain != fusion_node_domain for opset in model.opset_import
    ):
        logger.debug(
            f"Add opset with domain={fusion_node_domain} and version={fusion_node_version} to model"
        )
        opset = model.opset_import.add()
        opset.domain = fusion_node_domain
        opset.version = fusion_node_version

    for pattern_to_main_node_bindings, pattern_to_main_value_bindings in matches:
        if (
            pattern_to_main_node_bindings is None
            or pattern_to_main_value_bindings is None
        ):
            # This is not a match.
            continue
        fuse(
            model.graph,
            pattern_graph,
            # TODO: enable fusing sub-graph into another sub-graph.
            # Currently, only single-node fusion is supported.
            onnx.helper.make_node(
                op_type=fusion_node_type,
                inputs=[
                    pattern_to_main_value_bindings[value_info.name]
                    for value_info in pattern_graph.input
                ],
                outputs=[
                    pattern_to_main_value_bindings[value_info.name]
                    for value_info in pattern_graph.output
                ],
                name=pattern_to_main_node_bindings[pattern_graph.node[-1].name],
                domain=fusion_node_domain,
                **fusion_node_attributes,
            ),
            pattern_to_main_node_bindings,
            pattern_to_main_value_bindings,
        )


class FusionPattern:
    def __init__(
        self,
        pattern: "onnx.GraphProto",  # type: ignore[name-defined]
        # Fused operator type. e.g., "SoftmaxGrad" or "Gemm".
        fused_op_type: str,
        # Fused operator attributes. e.g., {"transA": 0, "transB": 1}.
        fused_op_kwargs: Dict[str, Any],
        # Operator domain. e.g., "com.microsoft" or "ai.onnx".
        fused_op_domain: str,
        # Operator version. e.g., 1 or 12.
        fused_op_version: int,
    ):
        self.pattern: "onnx.GraphProto" = pattern  # type: ignore[name-defined]
        self.fused_op_type: str = fused_op_type
        self.fused_op_kwargs: Dict[str, Any] = fused_op_kwargs
        self.fused_op_domain: str = fused_op_domain
        self.fused_op_version: int = fused_op_version


# Patterns called by `apply_all_fusions` are stored in `_FUSION_PATTERNS`.
_FUSION_PATTERNS: List[FusionPattern] = []


def push_pattern(pattern: FusionPattern):
    _FUSION_PATTERNS.append(pattern)


def pop_pattern():
    _FUSION_PATTERNS.pop()


def apply_all_fusions(onnx_model: "onnx.ModelProto"):  # type: ignore[name-defined]
    """Apply all fusions in _FUSION_PATTERNS to the given ONNX model.

    A new model is generated so the fusion is not in-place. Use
    pattern_push and pattern_pop to adjust _FUSION_PATTERNS if needed.
    """
    logger.debug("Applying all fusions")
    # Make this function immutable for `onnx_model`.

    fused_onnx_model = copy.deepcopy(onnx_model)
    # Apply fusion for SoftmaxBackward in-place (i.e., the input graph is changed)
    for pattern in _FUSION_PATTERNS:
        apply_fusion(
            fused_onnx_model,
            pattern.pattern,
            pattern.fused_op_type,
            pattern.fused_op_kwargs,
            pattern.fused_op_domain,
            pattern.fused_op_version,
        )

    return fused_onnx_model
