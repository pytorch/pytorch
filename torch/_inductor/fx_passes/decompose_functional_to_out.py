"""
Inductor pass to replace functional ops with their out variants.
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
    """Transform a functional op node to its out variant.

    Tries to reuse an input buffer for each output (reinplace) before falling
    back to allocating a fresh buffer via torch.empty. Reinplacing avoids the
    allocation entirely when the input is not used after this node.

    TODO(tianrengao): Phase 3 — replace _can_reuse_input with shared
    ReinplaceAnalyzer extracted from reinplace.py for full edge-case coverage.
    """
    from torch._library._out_variant import get_out_arg_names

    output_tensors = _get_output_tensors(node)
    if output_tensors is None or len(output_tensors) != num_outputs:
        log.warning("Could not get output tensors for %s", node.target)
        return False

    out_arg_names = get_out_arg_names(out_op)
    if len(out_arg_names) != num_outputs:
        log.warning(
            "Out arg name count mismatch for %s: expected %d, got %d",
            out_op,
            num_outputs,
            len(out_arg_names),
        )
        return False

    # Build node ordering once for reinplace safety checks
    node_order = {n: i for i, n in enumerate(graph.nodes)}
    # Track which inputs have already been reused to avoid double-mutation
    reused_inputs: set[fx.Node] = set()

    with graph.inserting_before(node):
        alloc_nodes: list[fx.Node] = []
        for fake_tensor in output_tensors:
            # Try to reuse an input buffer instead of allocating
            candidate = _find_matching_input(node, fake_tensor, reused_inputs)
            if candidate is not None and _can_reuse_input(
                node, candidate, node_order
            ):
                alloc_nodes.append(candidate)
                reused_inputs.add(candidate)
                log.debug(
                    "Reinplacing output with input %s for %s",
                    candidate,
                    node.target,
                )
            else:
                # Fallback: allocate fresh buffer
                alloc_node = graph.call_function(
                    torch.empty,
                    args=(tuple(fake_tensor.shape),),
                    kwargs={
                        "dtype": fake_tensor.dtype,
                        "device": fake_tensor.device,
                    },
                )
                alloc_node.meta["val"] = _create_fake_like(fake_tensor, node)
                alloc_nodes.append(alloc_node)

        # Create out variant call
        out_kwargs = dict(node.kwargs)
        out_kwargs.update(zip(out_arg_names, alloc_nodes))
        out_call = graph.call_function(out_op, args=node.args, kwargs=out_kwargs)
        out_call.meta["val"] = None

    _replace_uses(graph, node, alloc_nodes)
    graph.erase_node(node)

    return True


def _find_matching_input(
    node: fx.Node,
    fake_tensor: torch.Tensor,
    reused_inputs: set[fx.Node],
) -> fx.Node | None:
    """Find an input arg whose buffer matches the output's shape/dtype/device/strides.

    Each input can only be reused once (tracked by reused_inputs) to prevent
    double-mutation where the same buffer is used for multiple out args.
    """
    for arg in node.args:
        if not isinstance(arg, fx.Node):
            continue
        if arg in reused_inputs:
            continue
        arg_val = arg.meta.get("val")
        if arg_val is None or not isinstance(arg_val, torch.Tensor):
            continue
        if (
            arg_val.shape == fake_tensor.shape
            and arg_val.dtype == fake_tensor.dtype
            and arg_val.device == fake_tensor.device
            and arg_val.stride() == fake_tensor.stride()
        ):
            return arg
    return None


def _can_reuse_input(
    node: fx.Node,
    candidate: fx.Node,
    node_order: dict[fx.Node, int],
) -> bool:
    """Check if candidate input buffer can be safely overwritten as an out buffer.

    This is safe when:
    1. candidate is not a graph input (placeholder/get_attr) — overwriting those
       would require copy_ epilogue analysis (handled in Phase 3).
    2. candidate has trackable storage.
    3. candidate has no uses after the current node — otherwise overwriting it
       would corrupt data that downstream nodes need.

    This is a simplified version of reinplace.py's can_inplace(). It does not
    handle view chains or overlapping storage. Phase 3 will replace this with
    the shared ReinplaceAnalyzer for full coverage.
    """
    from torch._inductor.fx_utils import get_node_storage

    # Don't overwrite graph inputs
    if candidate.op in ("placeholder", "get_attr"):
        return False

    if get_node_storage(candidate) is None:
        return False

    # candidate must have no uses after node
    node_pos = node_order[node]
    for user in candidate.users:
        if user is node:
            continue
        if node_order.get(user, 0) > node_pos:
            return False

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
