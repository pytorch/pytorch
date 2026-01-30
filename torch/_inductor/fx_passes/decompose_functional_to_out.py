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
    get_out_variant,
    has_out_variant,
    FunctionalToOutMapping,
    TensorSpec,
)


log = logging.getLogger(__name__)


def decompose_functional_to_out(graph: fx.Graph) -> bool:
    """
    Decompose functional custom ops to their out variants.

    This is the main entry point called from post_grad passes.

    Args:
        graph: The FX graph to transform

    Returns:
        True if any transformations were made
    """
    modified = False
    nodes_to_transform = []

    # First pass: identify nodes to transform
    for node in graph.nodes:
        if node.op != "call_function":
            continue

        # Check if this op has a registered out variant
        if has_out_variant(node.target):
            nodes_to_transform.append(node)

    # Second pass: transform nodes (iterate over copy to allow modification)
    for node in nodes_to_transform:
        mapping = get_out_variant(node.target)
        if mapping is None:
            continue

        try:
            success = _decompose_node(graph, node, mapping)
            modified |= success
            if success:
                log.debug(f"Decomposed {node.target} -> {mapping.out_op}")
        except Exception as e:
            log.warning(f"Failed to decompose {node.target}: {e}")
            continue

    if modified:
        # Clean up the graph
        graph.lint()
        # DCE is safe here because out variant ops are marked as mutable
        # (via mutates_args in their definition), so is_impure() returns True
        # and eliminate_dead_code() will preserve them.
        graph.eliminate_dead_code()

    return modified


def _decompose_node(
    graph: fx.Graph,
    node: fx.Node,
    mapping: FunctionalToOutMapping,
) -> bool:
    """
    Transform a single functional op node to its out variant.

    Transformation:
    ---------------
    Before:
        %result = call_function[target=functional_op](%input, %scale)
        %output = call_function[target=getitem](%result, 0)
        %out_scale = call_function[target=getitem](%result, 1)
        ... uses of %output and %out_scale ...

    After:
        %output_buf = call_function[target=torch.empty](shape, dtype=..., device=...)
        %scale_buf = call_function[target=torch.empty](shape, dtype=..., device=...)
        call_function[target=out_op](%output_buf, %scale_buf, %input, %scale)
        ... %output -> %output_buf, %out_scale -> %scale_buf ...

    Args:
        graph: The FX graph
        node: The functional op node to transform
        mapping: The functional→out mapping

    Returns:
        True if transformation succeeded
    """
    # Get output tensor specifications from fake tensors in node metadata
    output_specs = _get_output_specs_from_node(node, mapping)
    if output_specs is None:
        log.warning(f"Could not infer output specs for {node.target}")
        return False

    # Insert allocation nodes before the functional op
    with graph.inserting_before(node):
        output_buffers = []
        for i, spec in enumerate(output_specs):
            # Create the allocation call
            alloc_node = _create_allocation_node(graph, spec, node, i)
            output_buffers.append(alloc_node)

        # Build args for out variant: (out1, out2, ..., in1, in2, ..., **kwargs)
        out_args = tuple(output_buffers) + tuple(node.args)
        out_kwargs = dict(node.kwargs)

        # Insert the out variant call
        # Ensure we use the OpOverload (e.g., .default) not OpOverloadPacket
        # because OpOverload has proper schema for is_impure() check in DCE
        out_op = mapping.out_op
        if hasattr(out_op, "default"):
            out_op = out_op.default

        out_call = graph.call_function(
            out_op,
            args=out_args,
            kwargs=out_kwargs,
        )
        # Out ops return None
        out_call.meta["val"] = None
        out_call.meta["decomposed_from"] = node.target

    # Replace uses of the original node's outputs
    _replace_output_uses(graph, node, output_buffers)

    # Remove the original node
    graph.erase_node(node)

    return True


def _get_output_specs_from_node(
    node: fx.Node,
    mapping: FunctionalToOutMapping,
) -> Optional[list[TensorSpec]]:
    """
    Extract output tensor specifications from the node's metadata.

    The fake tensor values in node.meta["val"] contain shape/dtype/device info.
    """
    val = node.meta.get("val")
    if val is None:
        # Try to infer from the mapping using the input args
        try:
            # Get fake values from args
            fake_args = tuple(_get_fake_val(a) for a in node.args)
            fake_kwargs = {k: _get_fake_val(v) for k, v in node.kwargs.items()}
            return mapping.get_output_specs(fake_args, fake_kwargs)
        except Exception as e:
            log.debug(f"Could not infer specs from mapping: {e}")
            return None

    # Normalize to tuple
    if isinstance(val, torch.Tensor):
        vals = (val,)
    elif isinstance(val, (tuple, list)):
        vals = val
    else:
        return None

    # Extract specs from fake tensors
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
        else:
            # Non-tensor in output tuple
            log.debug(f"Non-tensor output element: {type(v)}")

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
    """
    Create a tensor allocation node in the graph.

    Args:
        graph: The FX graph
        spec: Tensor specification
        source_node: The original functional op node (for metadata)
        output_idx: Which output this allocation is for

    Returns:
        The allocation node
    """
    # Use torch.empty for allocation
    # Note: Inductor's empty lowering doesn't support requires_grad kwarg
    alloc_node = graph.call_function(
        torch.empty,
        args=(spec.shape,),
        kwargs={
            "dtype": spec.dtype,
            "device": spec.device,
        },
    )

    # Set metadata for the allocation
    # Create a fake tensor with the same spec for downstream passes
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch._guards import detect_fake_mode

    # Try to get existing fake mode
    fake_mode = detect_fake_mode([source_node.meta.get("val")])

    if fake_mode is not None:
        # Use existing fake mode
        with fake_mode:
            fake_tensor = torch.empty(
                spec.shape,
                dtype=spec.dtype,
                device=spec.device,
                requires_grad=spec.requires_grad,
            )
    else:
        # Create a basic fake tensor for metadata
        fake_tensor = torch.empty(
            spec.shape,
            dtype=spec.dtype,
            device="meta",  # Use meta device for shape-only tensor
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
    """
    Replace uses of the original functional op's outputs with the new buffers.

    Handles:
    1. Single output case: direct replacement
    2. Multiple outputs case: handle getitem nodes

    Args:
        graph: The FX graph
        original_node: The original functional op node
        output_buffers: List of allocation nodes for outputs
    """
    if len(output_buffers) == 1:
        # Single output - direct replacement
        original_node.replace_all_uses_with(output_buffers[0])
        return

    # Multiple outputs - need to handle getitem nodes
    users_to_replace = []

    for user in list(original_node.users.keys()):
        if user.op == "call_function" and user.target is operator.getitem:
            idx = user.args[1]
            if isinstance(idx, int) and 0 <= idx < len(output_buffers):
                users_to_replace.append((user, output_buffers[idx]))

    # Replace getitem users
    for getitem_node, replacement in users_to_replace:
        getitem_node.replace_all_uses_with(replacement)
        graph.erase_node(getitem_node)


# =============================================================================
# Integration with Inductor pipeline
# =============================================================================


def should_decompose_functional_to_out() -> bool:
    """Check if the decompose pass should run based on config."""
    from torch._inductor import config

    return getattr(config, "decompose_functional_to_out", True)


def run_decompose_pass(gm: fx.GraphModule) -> fx.GraphModule:
    """
    Run the decompose pass on a GraphModule.

    This is the entry point for integration with Inductor's pass pipeline.

    Args:
        gm: The GraphModule to transform

    Returns:
        The transformed GraphModule (modified in place)
    """
    if not should_decompose_functional_to_out():
        return gm

    modified = decompose_functional_to_out(gm.graph)

    if modified:
        gm.recompile()
        log.info("Decomposed functional ops to out variants")

    return gm
