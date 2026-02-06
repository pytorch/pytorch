"""
Inductor pass to replace functional custom ops to their out variants.
"""

import logging
import operator

import torch
import torch.fx as fx
from torch._ops import OpOverload


log = logging.getLogger(__name__)


def decompose_functional_to_out(graph: fx.Graph) -> bool:
    """
    Decompose functional custom ops to their out variants.
    """
    from torch._library._out_variant import get_out_arg_count, to_out_variant

    modified = False
    nodes_to_process = []

    # Collect nodes that have out variants
    for node in graph.nodes:
        if node.op != "call_function":
            continue

        if not isinstance(node.target, OpOverload):
            continue

        out_op = to_out_variant(node.target)
        if out_op is not None:
            nodes_to_process.append((node, out_op))

    # Transform each node
    for node, out_op in nodes_to_process:
        num_outputs = get_out_arg_count(out_op)
        if num_outputs == 0:
            continue

        success = _transform_node(graph, node, out_op, num_outputs)
        if success:
            modified = True
            log.debug("Decomposed %s -> %s", node.target, out_op)

    if modified:
        graph.lint()
        graph.eliminate_dead_code()

    return modified


def _transform_node(
    graph: fx.Graph,
    node: fx.Node,
    out_op,
    num_outputs: int,
) -> bool:
    """Transform a functional op node to its out variant."""
    # TODO(tianrengao): add reinplace logic
    # Get output tensor info from fake tensor metadata
    output_tensors = _get_output_tensors(node)
    if output_tensors is None or len(output_tensors) != num_outputs:
        log.warning("Could not get output tensors for %s", node.target)
        return False

    with graph.inserting_before(node):
        # Create allocation nodes for each output
        alloc_nodes = []
        for i, fake_tensor in enumerate(output_tensors):
            alloc_node = graph.call_function(
                torch.empty,
                args=(tuple(fake_tensor.shape),),
                kwargs={
                    "dtype": fake_tensor.dtype,
                    "device": fake_tensor.device,
                },
            )
            # Propagate fake tensor metadata
            alloc_node.meta["val"] = _create_fake_like(fake_tensor, node)
            alloc_nodes.append(alloc_node)

        # Create out variant call: out_op(out1, out2, ..., input1, input2, ...)
        out_args = tuple(alloc_nodes) + tuple(node.args)
        out_call = graph.call_function(out_op, args=out_args, kwargs=dict(node.kwargs))
        out_call.meta["val"] = None

    # Replace uses of original outputs with allocated buffers
    _replace_uses(graph, node, alloc_nodes)
    graph.erase_node(node)

    return True


def _get_output_tensors(node: fx.Node) -> list[torch.Tensor] | None:
    """Get fake tensors from node metadata."""
    val = node.meta.get("val")
    if val is None:
        return None

    if isinstance(val, torch.Tensor):
        return [val]
    elif isinstance(val, (tuple, list)):
        tensors = [v for v in val if isinstance(v, torch.Tensor)]
        return tensors if tensors else None

    return None


def _create_fake_like(fake_tensor: torch.Tensor, source_node: fx.Node) -> torch.Tensor:
    """Create a fake tensor with same properties."""
    from torch._guards import detect_fake_mode

    fake_mode = detect_fake_mode([source_node.meta.get("val")])

    if fake_mode is not None:
        with fake_mode:
            return torch.empty(
                tuple(fake_tensor.shape),
                dtype=fake_tensor.dtype,
                device=fake_tensor.device,
                requires_grad=fake_tensor.requires_grad,
            )
    else:
        return torch.empty(
            tuple(fake_tensor.shape),
            dtype=fake_tensor.dtype,
            device="meta",
            requires_grad=fake_tensor.requires_grad,
        )


def _replace_uses(
    graph: fx.Graph,
    original: fx.Node,
    replacements: list[fx.Node],
) -> None:
    """Replace uses of original node outputs with replacement nodes."""
    if len(replacements) == 1:
        original.replace_all_uses_with(replacements[0])
        return

    # Handle tuple outputs: replace getitem nodes
    for user in list(original.users.keys()):
        if user.op == "call_function" and user.target is operator.getitem:
            idx = user.args[1]
            if isinstance(idx, int) and 0 <= idx < len(replacements):
                user.replace_all_uses_with(replacements[idx])
                graph.erase_node(user)


def run_decompose_pass(gm: fx.GraphModule) -> fx.GraphModule:
    """
    Entry point for running the decompose pass.

    Controlled by torch._inductor.config.decompose_functional_to_out.
    """
    from torch._inductor import config

    if not getattr(config, "decompose_functional_to_out", False):
        return gm

    modified = decompose_functional_to_out(gm.graph)

    if modified:
        gm.recompile()
        log.info("Decomposed functional ops to out variants")

    return gm
