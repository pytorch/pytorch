"""
Inductor pass to decompose functional custom ops to their out variants.

This pass runs during Inductor's post-grad optimization phase. It:
1. Identifies functional custom ops with registered out variants
2. Inserts buffer allocation nodes before the op
3. Replaces the functional op call with the out variant call

This enables:
- CUDAGraph compatibility (requires fixed buffer addresses)
- Future memory optimization through buffer reuse (Phase 2)

The pass is controlled by `torch._inductor.config.decompose_functional_to_out`.
"""

import logging
import operator
from typing import Any, Optional

import torch
import torch.fx as fx
from torch._library.functional_to_out import (
    FunctionalToOutMapping,
    get_out_variant,
    has_any_registered_mappings,
    TensorSpec,
)


log = logging.getLogger(__name__)


def decompose_functional_to_out(graph: fx.Graph) -> bool:
    """
    Decompose functional custom ops to their out variants.

    Args:
        graph: The FX graph to transform

    Returns:
        True if any transformations were made
    """
    if not has_any_registered_mappings():
        return False

    modified = False
    nodes_to_transform = []

    for node in graph.nodes:
        if node.op != "call_function":
            continue
        if get_out_variant(node.target) is not None:
            nodes_to_transform.append(node)

    for node in nodes_to_transform:
        mapping = get_out_variant(node.target)
        if mapping is None:
            continue

        success = _decompose_node(graph, node, mapping)
        modified |= success
        if success:
            log.debug("Decomposed %s -> %s", node.target, mapping.out_op)

    if modified:
        graph.lint()
        graph.eliminate_dead_code()

    return modified


def _decompose_node(
    graph: fx.Graph,
    node: fx.Node,
    mapping: FunctionalToOutMapping,
) -> bool:
    """Transform a single functional op node to its out variant."""
    output_specs = _get_output_specs_from_node(node, mapping)
    if output_specs is None:
        log.warning("Could not infer output specs for %s", node.target)
        return False

    with graph.inserting_before(node):
        output_buffers = []
        for i, spec in enumerate(output_specs):
            alloc_node = _create_allocation_node(graph, spec, node, i)
            output_buffers.append(alloc_node)

        out_args = tuple(output_buffers) + tuple(node.args)
        out_kwargs = dict(node.kwargs)

        out_op = mapping.out_op
        if hasattr(out_op, "default"):
            out_op = out_op.default

        out_call = graph.call_function(out_op, args=out_args, kwargs=out_kwargs)
        out_call.meta["val"] = None
        out_call.meta["decomposed_from"] = node.target

    _replace_output_uses(graph, node, output_buffers)
    graph.erase_node(node)

    return True


def _get_output_specs_from_node(
    node: fx.Node,
    mapping: FunctionalToOutMapping,
) -> Optional[list[TensorSpec]]:
    """Extract output tensor specifications from the node's metadata."""
    val = node.meta.get("val")
    if val is None:
        fake_args = tuple(_get_fake_val(a) for a in node.args)
        fake_kwargs = {k: _get_fake_val(v) for k, v in node.kwargs.items()}

        if any(isinstance(a, fx.Node) for a in fake_args):
            return None
        if any(isinstance(v, fx.Node) for v in fake_kwargs.values()):
            return None

        return mapping.get_output_specs(fake_args, fake_kwargs)

    if isinstance(val, torch.Tensor):
        vals = (val,)
    elif isinstance(val, (tuple, list)):
        vals = val
    else:
        return None

    specs = []
    for v in vals:
        if isinstance(v, torch.Tensor):
            specs.append(
                TensorSpec(
                    shape=tuple(v.shape),
                    dtype=v.dtype,
                    device=v.device,
                    requires_grad=v.requires_grad,
                )
            )

    return specs if specs else None


def _get_fake_val(node_or_val: Any) -> Any:
    """Get the fake tensor value from a node or return the value as-is."""
    if isinstance(node_or_val, fx.Node):
        return node_or_val.meta.get("val", node_or_val)
    return node_or_val


def _create_allocation_node(
    graph: fx.Graph,
    spec: TensorSpec,
    source_node: fx.Node,
    output_idx: int,
) -> fx.Node:
    """Create a tensor allocation node in the graph."""
    alloc_node = graph.call_function(
        torch.empty,
        args=(spec.shape,),
        kwargs={"dtype": spec.dtype, "device": spec.device},
    )

    from torch._guards import detect_fake_mode

    fake_mode = detect_fake_mode([source_node.meta.get("val")])

    if fake_mode is not None:
        with fake_mode:
            fake_tensor = torch.empty(
                spec.shape,
                dtype=spec.dtype,
                device=spec.device,
                requires_grad=spec.requires_grad,
            )
    else:
        fake_tensor = torch.empty(
            spec.shape,
            dtype=spec.dtype,
            device="meta",
            requires_grad=spec.requires_grad,
        )

    alloc_node.meta["val"] = fake_tensor
    alloc_node.meta["allocation_for"] = (source_node.target, output_idx)

    return alloc_node


def _replace_output_uses(
    graph: fx.Graph,
    original_node: fx.Node,
    output_buffers: list[fx.Node],
) -> None:
    """Replace uses of the original functional op's outputs with the new buffers."""
    if len(output_buffers) == 1:
        original_node.replace_all_uses_with(output_buffers[0])
        return

    users_to_replace = []
    for user in list(original_node.users.keys()):
        if user.op == "call_function" and user.target is operator.getitem:
            idx = user.args[1]
            if isinstance(idx, int) and 0 <= idx < len(output_buffers):
                users_to_replace.append((user, output_buffers[idx]))

    for getitem_node, replacement in users_to_replace:
        getitem_node.replace_all_uses_with(replacement)
        graph.erase_node(getitem_node)


def run_decompose_pass(gm: fx.GraphModule) -> fx.GraphModule:
    """
    Run the decompose pass on a GraphModule.

    Args:
        gm: The GraphModule to transform

    Returns:
        The transformed GraphModule
    """
    from torch._inductor import config

    if not getattr(config, "decompose_functional_to_out", False):
        return gm

    modified = decompose_functional_to_out(gm.graph)

    if modified:
        gm.recompile()
        log.info("Decomposed functional ops to out variants")

    return gm
