import copy
import itertools
import os
from typing import Dict, List, Set

import onnx


def build_uses_and_sources(onnx_graph):
    # Build a dictionary of uses and sources for each node in the model
    # {"x": [node0, node1], "y": [node2], ...}
    uses: Dict[str, List["onnx.NodeProto"]] = {}
    # {"x": node0, "y": node1, ...}
    # Model inputs and initializers do not have sources.
    sources: Dict[str, "onnx.NodeProto"] = {}
    for node in onnx_graph.node:
        assert node.op_type not in ("If", "Loop"), "If and Loop are not supported."
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


def build_upstream_nodes_and_downstream_nodes(onnx_graph):
    uses, sources = build_uses_and_sources(onnx_graph)
    # {node0: [node1, node2], node1: [node3, node4], ...}
    upstream_nodes: Dict[str, List["onnx.NodeProto"]] = {}
    downstream_nodes: Dict[str, List["onnx.NodeProto"]] = {}

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


def search_for_root(onnx_graph):
    _, sources = build_uses_and_sources(onnx_graph)
    # Find the root node of the graph
    # A root node is a node consumes the graph's inputs or initializers;
    # that is, some of the node's inputs don't have source node.

    root_nodes = []
    for node in onnx_graph.node:
        if all(input_name not in sources for input_name in node.input):
            root_nodes.append(node)

    return root_nodes


def print_message(*messages):
    if os.environ.get("DEBUG") == "1":
        print(*messages)


def search_for_pattern(
    main_graph: "onnx.GraphProto",
    pattern_graph: "onnx.GraphProto",
    skipped_main_node_names: Set[str],
    pattern_upstreams_nodes: Dict[str, List["onnx.NodeProto"]],
    pattern_downstream_nodes: Dict[str, List["onnx.NodeProto"]],
    main_upstreams_nodes: Dict[str, List["onnx.NodeProto"]],
    main_downstream_nodes: Dict[str, List["onnx.NodeProto"]],
):
    pattern_root_nodes = search_for_root(pattern_graph)

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

        queue = [
            (node, pattern_node)
            for node, pattern_node in zip(main_root_nodes, pattern_root_nodes)
        ]

        pattern_to_main_node_bindings = {
            pattern_node.name: node.name
            for node, pattern_node in zip(main_root_nodes, pattern_root_nodes)
        }

        fail = False
        pattern_to_main_value_bindings = {}

        def try_bind_values(bindings, node, pattern_node):
            # TODO: examine type and shape bindings for graph inputs and outputs
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

            for name, pattern_name in zip(node.input, pattern_node.input):
                if pattern_name in bindings and bindings[pattern_name] != name:
                    print_message(
                        f"Binding fail because new pattern-to-main pair ({pattern_name}, {name}) conflicts with existing binding ({pattern_name}, {bindings[pattern_name]})"
                    )
                    return False
                elif pattern_name in bindings and bindings[pattern_name] == name:
                    pass
                else:
                    bindings[pattern_name] = name

            for name, pattern_name in zip(node.output, pattern_node.output):
                if pattern_name in bindings and bindings[pattern_name] != name:
                    print_message(
                        f"Binding fail because new pattern-to-main pair ({pattern_name}, {name}) conflicts with existing binding ({pattern_name}, {bindings[pattern_name]})"
                    )
                    return False
                elif pattern_name in bindings and bindings[pattern_name] == name:
                    pass
                else:
                    bindings[pattern_name] = name

            return True

        for node, pattern_node in zip(main_root_nodes, pattern_root_nodes):
            if not try_bind_values(pattern_to_main_value_bindings, node, pattern_node):
                fail = True
                break

        if fail:
            continue

        while queue:
            main_node, pattern_nodes = queue.pop()

            print_message(
                "Current pattern_to_main_node_bindings: ", pattern_to_main_node_bindings
            )

            to_queue = []
            traversed_names = []

            if len(main_downstream_nodes[main_node.name]) != len(
                pattern_downstream_nodes[pattern_nodes.name]
            ):
                break

            fail = False
            for main_downstream_node, pattern_downstream_node in zip(
                main_downstream_nodes[main_node.name],
                pattern_downstream_nodes[pattern_nodes.name],
            ):
                print_message(
                    f"Try binding main graph node {main_downstream_node.name} to pattern node {pattern_downstream_node.name}"
                )
                if main_downstream_node.name in skipped_main_node_names:
                    print_message(
                        f"Binding fail because main graph node {main_downstream_node.name} should be skipped (e.g., when a node is already fused into another sub-graph)."
                    )
                    fail = True
                    break
                if main_downstream_node.op_type != pattern_downstream_node.op_type:
                    print_message(
                        f"Binding fail because different op_types {main_downstream_node.op_type} and {pattern_downstream_node.op_type}"
                    )
                    fail = True
                    break
                # TODO: check Constant node's value
                if (
                    pattern_downstream_node.name in pattern_to_main_node_bindings
                    and pattern_to_main_node_bindings[pattern_downstream_node.name]
                    != main_downstream_node.name
                ):
                    print_message(
                        f"Binding fail because new pattern-to-main pair ({pattern_downstream_node.name}, {main_downstream_node.name}) conflicts with existing binding ({pattern_downstream_node.name}, {pattern_to_main_node_bindings[pattern_downstream_node.name]})"
                    )
                    fail = True
                    break
                if not try_bind_values(
                    pattern_to_main_value_bindings,
                    main_downstream_node,
                    pattern_downstream_node,
                ):
                    print_message(
                        f"Binding fail because value bindings conflict for {main_downstream_node.name} and {pattern_downstream_node.name}"
                    )
                    fail = True
                    break
                print_message("Binding success")
                to_queue.append((main_downstream_node, pattern_downstream_node))
                traversed_names.append(pattern_downstream_node.name)

            if fail:
                break

            for node, pattern_node in to_queue:
                pattern_to_main_node_bindings[pattern_node.name] = node.name

            queue.extend(to_queue)

        if fail:
            continue

        print_message("=======================")
        print_message("One match found:")
        print_message(pattern_to_main_node_bindings)
        print_message(pattern_to_main_value_bindings)
        print_message("=======================")
        return pattern_to_main_node_bindings, pattern_to_main_value_bindings
    return None, None


def search_for_patterns(main_graph, pattern_graph):
    skipped_main_node_names = set()
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
    main_node_names = {value for value in pattern_to_main_node_bindings.values()}
    main_value_names = {value for value in pattern_to_main_value_bindings.values()}

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


def apply_fusion(model, pattern_graph, fusion_node_type, fusion_node_attributes):
    matches = search_for_patterns(model.graph, pattern_graph)
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
                domain="com.microsoft",
                **fusion_node_attributes,
            ),
            pattern_to_main_node_bindings,
            pattern_to_main_value_bindings,
        )


try:
    import onnxscript
    from onnxscript import FLOAT, opset18

    _REGISTER_PREDEFINED_FUSION = True
except ImportError:
    _REGISTER_PREDEFINED_FUSION = False


aten_opset = onnxscript.values.Opset(domain="pkg.onnxscript.torch_lib", version=1)


@onnxscript.script(default_opset=opset18)
def softmax_backward(dY: FLOAT, Y: FLOAT):
    dYY = aten_opset.aten_mul(dY, Y)
    sum = aten_opset._aten_sum_dim_onnx(dYY, -1)
    scaled = aten_opset.aten_mul(Y, sum)
    dX = aten_opset.aten_sub(dYY, scaled)
    return dX


softmax_backward_model = softmax_backward.to_model_proto(
    input_types=[FLOAT["S", "H"], FLOAT["S", "H"]], output_types=[FLOAT["S", "H"]]
)


def apply_all_fusions(onnx_model):
    # Make this function immutable for `onnx_model`.

    fused_onnx_model = copy.deepcopy(onnx_model)
    # Apply fusion for SoftmaxBackward in-place (i.e., the input graph is changed)
    apply_fusion(
        fused_onnx_model, softmax_backward_model.graph, "SoftmaxGrad", {"axis": -1}
    )

    opset = fused_onnx_model.opset_import.add()
    opset.domain = "com.microsoft"
    opset.version = 1

    return fused_onnx_model