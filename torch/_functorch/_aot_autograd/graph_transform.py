"""
Graph transformation utilities for descriptor-based FX graphs.

This module provides utility functions for manipulating FX graphs that have
AOTAutograd descriptors attached to their nodes (via node.meta["desc"]).
These utilities enable safe graph transformations while maintaining correct
metadata.

Example usage:
    from torch._functorch._aot_autograd.graph_transform import (
        get_meta, add_input, remove_output, partition_by_output
    )

    # Get descriptor-based metadata from a graph
    meta = get_meta(gm)

    # Add a new input with descriptor
    new_node = add_input(gm, PlainAOTInput(idx=99), example_value=torch.randn(3, 4))

    # Remove an output by descriptor
    remove_output(gm, GradAOTOutput(grad_of=BufferAOTInput(target="running_mean")))

    # Create a subgraph keeping only certain outputs
    new_gm = partition_by_output(gm, lambda d: isinstance(d, GradAOTOutput))
"""

from __future__ import annotations

import copy
from typing import Any, Optional, TYPE_CHECKING

import torch
import torch.fx as fx

from .descriptors import AOTInput, AOTOutput
from .schemas import DescriptorBasedMeta


if TYPE_CHECKING:
    from collections.abc import Callable

    from .schemas import ViewAndMutationMeta


def get_meta(gm: fx.GraphModule) -> DescriptorBasedMeta:
    """
    Extract descriptor-based metadata from a graph module.

    This scans the graph for placeholder nodes (inputs) and the output node,
    extracting descriptors from node.meta["desc"] to build a DescriptorBasedMeta.

    Args:
        gm: The FX GraphModule with descriptors attached to nodes

    Returns:
        DescriptorBasedMeta derived from the graph's descriptors
    """
    return DescriptorBasedMeta.from_graph(gm.graph)


def _make_placeholder_name(desc: AOTInput, graph: fx.Graph) -> str:
    """Generate a valid placeholder name for a descriptor."""
    # Use a sanitized version of the descriptor expression
    base_name = desc.expr().replace("[", "_").replace("]", "").replace(".", "_")
    base_name = base_name.replace("(", "_").replace(")", "").replace("'", "")
    base_name = base_name.replace(" ", "_").replace(",", "_").replace("=", "_")

    # Ensure it starts with a letter or underscore
    if base_name and not base_name[0].isalpha() and base_name[0] != "_":
        base_name = "_" + base_name

    # Ensure uniqueness
    existing_names = {n.name for n in graph.nodes}
    name = base_name
    counter = 0
    while name in existing_names:
        counter += 1
        name = f"{base_name}_{counter}"

    return name


def add_input(
    gm: fx.GraphModule,
    desc: AOTInput,
    example_value: Any,
    insert_before: Optional[fx.Node] = None,
) -> fx.Node:
    """
    Add a new input placeholder with the given descriptor.

    Args:
        gm: The FX GraphModule to modify
        desc: The AOTInput descriptor for the new input
        example_value: Example value for the input (used for node.meta["val"])
        insert_before: Optional node to insert before. If None, inserts at the
                       beginning of the graph (before the first node).

    Returns:
        The newly created placeholder node
    """
    graph = gm.graph

    # Find insertion point
    if insert_before is None:
        # Insert at the beginning, but after any existing placeholders
        insert_point = None
        for node in graph.nodes:
            if node.op != "placeholder":
                insert_point = node
                break
        if insert_point is None:
            # Graph has only placeholders or is empty, insert at end
            insert_point = list(graph.nodes)[-1] if list(graph.nodes) else None
    else:
        insert_point = insert_before

    # Generate a valid placeholder name
    placeholder_name = _make_placeholder_name(desc, graph)

    # Create the placeholder
    if insert_point is not None:
        with graph.inserting_before(insert_point):
            node = graph.placeholder(placeholder_name)
    else:
        node = graph.placeholder(placeholder_name)

    node.meta["desc"] = desc
    node.meta["val"] = example_value

    gm.recompile()
    return node


def remove_input(gm: fx.GraphModule, desc: AOTInput) -> None:
    """
    Remove an input placeholder by its descriptor.

    This removes the placeholder node with the matching descriptor and runs
    dead code elimination to clean up any downstream uses.

    Args:
        gm: The FX GraphModule to modify
        desc: The AOTInput descriptor of the input to remove

    Raises:
        ValueError: If no input with the given descriptor is found
    """
    graph = gm.graph

    # Find the placeholder with matching descriptor
    node_to_remove = None
    for node in graph.nodes:
        if node.op == "placeholder" and node.meta.get("desc") == desc:
            node_to_remove = node
            break

    if node_to_remove is None:
        raise ValueError(f"No input found with descriptor: {desc}")

    # Replace all uses with None before removing
    node_to_remove.replace_all_uses_with(None)  # type: ignore[arg-type]
    graph.erase_node(node_to_remove)
    graph.eliminate_dead_code()
    gm.recompile()


def remove_output(gm: fx.GraphModule, desc: AOTOutput) -> None:
    """
    Remove an output by its descriptor.

    This removes the output with the matching descriptor from the output node
    and runs dead code elimination to remove any unused computation.

    Args:
        gm: The FX GraphModule to modify
        desc: The AOTOutput descriptor of the output to remove

    Raises:
        ValueError: If no output with the given descriptor is found
    """
    graph = gm.graph

    # Find the output node
    output_node = None
    for node in graph.nodes:
        if node.op == "output":
            output_node = node
            break

    if output_node is None:
        raise ValueError("Graph has no output node")

    descs = output_node.meta.get("desc", [])
    if not isinstance(descs, (list, tuple)):
        descs = [descs]

    # Find the index of the descriptor to remove
    idx_to_remove = None
    for i, d in enumerate(descs):
        if d == desc:
            idx_to_remove = i
            break

    if idx_to_remove is None:
        raise ValueError(f"No output found with descriptor: {desc}")

    # Update output node args and descriptors
    output_args = list(output_node.args[0])
    del output_args[idx_to_remove]

    new_descs = list(descs)
    del new_descs[idx_to_remove]

    output_node.args = (output_args,)
    output_node.meta["desc"] = new_descs

    graph.eliminate_dead_code()
    gm.recompile()


def get_nodes_by_desc_type(
    gm: fx.GraphModule, desc_type: type
) -> list[fx.Node]:
    """
    Get all nodes (input or output) matching a descriptor type.

    Args:
        gm: The FX GraphModule to search
        desc_type: The descriptor type to match (e.g., ParamAOTInput, GradAOTOutput)

    Returns:
        List of nodes whose descriptors are instances of desc_type
    """
    result: list[fx.Node] = []

    for node in gm.graph.nodes:
        if node.op == "placeholder":
            desc = node.meta.get("desc")
            if isinstance(desc, desc_type):
                result.append(node)
        elif node.op == "output":
            output_args = node.args[0] if node.args else []
            descs = node.meta.get("desc", [])
            if not isinstance(descs, (list, tuple)):
                descs = [descs]
            for sub_node, sub_desc in zip(output_args, descs):
                if isinstance(sub_desc, desc_type) and isinstance(sub_node, fx.Node):
                    result.append(sub_node)

    return result


def partition_by_output(
    gm: fx.GraphModule,
    output_filter: Callable[[AOTOutput], bool],
) -> fx.GraphModule:
    """
    Create a new graph keeping only outputs matching the filter.

    Inputs are pruned via dead code elimination. Descriptors on nodes
    are preserved.

    Args:
        gm: The FX GraphModule to partition
        output_filter: Callable that returns True for outputs to keep

    Returns:
        A new GraphModule with only the filtered outputs
    """
    new_gm = copy.deepcopy(gm)
    graph = new_gm.graph

    # Find the output node
    output_node = None
    for node in graph.nodes:
        if node.op == "output":
            output_node = node
            break

    if output_node is None:
        return new_gm

    descs = output_node.meta.get("desc", [])
    if not isinstance(descs, (list, tuple)):
        descs = [descs]

    output_args = output_node.args[0] if output_node.args else []

    # Filter outputs
    kept = [
        (n, d)
        for n, d in zip(output_args, descs)
        if output_filter(d)
    ]

    output_node.args = ([n for n, d in kept],)
    output_node.meta["desc"] = [d for n, d in kept]

    graph.eliminate_dead_code()
    new_gm.recompile()
    return new_gm


def get_input_nodes(gm: fx.GraphModule) -> list[fx.Node]:
    """
    Get all placeholder (input) nodes from the graph in order.

    Args:
        gm: The FX GraphModule

    Returns:
        List of placeholder nodes in their graph order
    """
    return [node for node in gm.graph.nodes if node.op == "placeholder"]


def get_output_nodes(gm: fx.GraphModule) -> list[fx.Node]:
    """
    Get the nodes that are outputs of the graph.

    Args:
        gm: The FX GraphModule

    Returns:
        List of nodes that are returned by the output node
    """
    for node in gm.graph.nodes:
        if node.op == "output":
            args = node.args[0] if node.args else []
            return [n for n in args if isinstance(n, fx.Node)]
    return []


def get_input_descs(gm: fx.GraphModule) -> list[AOTInput]:
    """
    Get all input descriptors from the graph in order.

    Args:
        gm: The FX GraphModule

    Returns:
        List of AOTInput descriptors for all placeholders
    """
    return [
        node.meta.get("desc")
        for node in gm.graph.nodes
        if node.op == "placeholder" and node.meta.get("desc") is not None
    ]


def get_output_descs(gm: fx.GraphModule) -> list[AOTOutput]:
    """
    Get all output descriptors from the graph.

    Args:
        gm: The FX GraphModule

    Returns:
        List of AOTOutput descriptors for all outputs
    """
    for node in gm.graph.nodes:
        if node.op == "output":
            descs = node.meta.get("desc", [])
            if not isinstance(descs, (list, tuple)):
                return [descs] if descs is not None else []
            return list(descs)
    return []


def update_input_desc(
    gm: fx.GraphModule, old_desc: AOTInput, new_desc: AOTInput
) -> None:
    """
    Update the descriptor of an input node.

    Args:
        gm: The FX GraphModule to modify
        old_desc: The current descriptor to find
        new_desc: The new descriptor to set

    Raises:
        ValueError: If no input with the old descriptor is found
    """
    for node in gm.graph.nodes:
        if node.op == "placeholder" and node.meta.get("desc") == old_desc:
            node.meta["desc"] = new_desc
            return

    raise ValueError(f"No input found with descriptor: {old_desc}")


def update_output_desc(
    gm: fx.GraphModule, old_desc: AOTOutput, new_desc: AOTOutput
) -> None:
    """
    Update the descriptor of an output.

    Args:
        gm: The FX GraphModule to modify
        old_desc: The current descriptor to find
        new_desc: The new descriptor to set

    Raises:
        ValueError: If no output with the old descriptor is found
    """
    for node in gm.graph.nodes:
        if node.op == "output":
            descs = node.meta.get("desc", [])
            if not isinstance(descs, (list, tuple)):
                descs = [descs]

            new_descs = list(descs)
            for i, d in enumerate(new_descs):
                if d == old_desc:
                    new_descs[i] = new_desc
                    node.meta["desc"] = new_descs
                    return

    raise ValueError(f"No output found with descriptor: {old_desc}")


def attach_mutation_info_to_graph(
    gm: fx.GraphModule,
    meta: "ViewAndMutationMeta",
) -> None:
    """
    Attach mutation/alias info from ViewAndMutationMeta to graph nodes.

    This attaches InputMutationInfo to placeholder nodes (via node.meta["mutation_info"])
    and OutputAliasInfoDesc to the output node (via node.meta["alias_info"]).

    After calling this function, DescriptorBasedMeta.from_graph() will be able to
    recover the full mutation/alias information.

    Args:
        gm: The FX GraphModule with descriptors attached to nodes
        meta: The ViewAndMutationMeta to extract mutation/alias info from
    """
    from .schemas import InputMutationInfo, OutputAliasInfoDesc, ViewAndMutationMeta

    input_descs = get_input_descs(gm)
    output_descs = get_output_descs(gm)

    # Attach mutation_info to placeholder nodes
    input_idx = 0
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            desc = node.meta.get("desc")
            if desc is not None and input_idx < len(meta.input_info):
                info = meta.input_info[input_idx]
                node.meta["mutation_info"] = InputMutationInfo.from_input_alias_info(info)
                input_idx += 1

    # Attach alias_info to output node
    for node in gm.graph.nodes:
        if node.op == "output":
            descs = node.meta.get("desc", [])
            if not isinstance(descs, (list, tuple)):
                descs = [descs]

            alias_infos = []
            num_user_outputs = len(meta.output_info)
            for i, desc in enumerate(descs):
                if i < len(meta.output_info):
                    info = meta.output_info[i]
                    alias_info = OutputAliasInfoDesc.from_output_alias_info(
                        info, input_descs, output_descs, num_user_outputs
                    )
                    alias_infos.append(alias_info)
                else:
                    alias_infos.append(None)

            node.meta["alias_info"] = alias_infos
            break


def detach_mutation_info_from_graph(gm: fx.GraphModule) -> None:
    """
    Remove mutation/alias info from graph nodes.

    This removes node.meta["mutation_info"] from placeholders and
    node.meta["alias_info"] from the output node.

    Args:
        gm: The FX GraphModule to modify
    """
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            node.meta.pop("mutation_info", None)
        elif node.op == "output":
            node.meta.pop("alias_info", None)


def validate_descriptor_meta_equivalence(
    descriptor_meta: "DescriptorBasedMeta",
    view_mutation_meta: "ViewAndMutationMeta",
) -> None:
    """
    Validate that DescriptorBasedMeta derives equivalent runtime behavior
    to ViewAndMutationMeta.

    This function checks that all derived indices match between the two
    metadata systems. Use this to verify the descriptor-based system
    produces correct results.

    Args:
        descriptor_meta: The descriptor-based metadata
        view_mutation_meta: The legacy index-based metadata

    Raises:
        AssertionError: If any derived indices don't match
    """
    descriptor_meta.validate_equivalence(view_mutation_meta)


def create_validated_descriptor_meta(
    view_mutation_meta: "ViewAndMutationMeta",
    input_descs: list[AOTInput],
    output_descs: list[AOTOutput],
) -> "DescriptorBasedMeta":
    """
    Create a DescriptorBasedMeta from ViewAndMutationMeta and validate equivalence.

    This is a convenience function that:
    1. Creates a DescriptorBasedMeta from the provided metadata and descriptors
    2. Validates that it derives the same runtime behavior as the original
    3. Returns the validated DescriptorBasedMeta

    Args:
        view_mutation_meta: The legacy ViewAndMutationMeta
        input_descs: List of input descriptors (matching the graph's placeholders)
        output_descs: List of output descriptors (matching the graph's outputs)

    Returns:
        A validated DescriptorBasedMeta

    Raises:
        AssertionError: If validation fails
    """
    descriptor_meta = DescriptorBasedMeta.from_view_and_mutation_meta(
        view_mutation_meta, input_descs, output_descs
    )
    descriptor_meta.validate_equivalence(view_mutation_meta)
    return descriptor_meta
