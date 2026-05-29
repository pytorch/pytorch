# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
FX graph passes for torch.comms.

This module provides compiler passes that can be applied to FX graphs
to optimize torchcomms operations.

NOTE: when inductor is enabled, these passes are applied automatically,
so they should be skipped to avoid redundant work.

- reinplacement_pass: converts functional ops back to in-place ops where possible
- strip_with_effects_pass: removes with_effects HOP wrappers from torch.comms ops

NOTE: when used in conjunction, the reinplacement pass should be applied first.
"""

import logging
import operator
from inspect import signature

import torch
from torch._higher_order_ops.effects import with_effects
from torch._inductor.fx_passes.reinplace import reinplace_inplaceable_ops
from torch._inductor.fx_utils import FakeTensorUpdater


__all__ = ["reinplacement_pass", "strip_with_effects_pass"]

logger = logging.getLogger(__name__)


def reinplacement_pass(
    gm: torch.fx.GraphModule, example_inputs=None
) -> torch.fx.GraphModule:
    """Convert functional ops back to in-place ops where safe to do so.

    Detects the FakeMode from node metadata and uses Inductor's
    ``reinplace_inplaceable_ops`` to replace functional ops with their
    in-place variants when the mutation is provably safe. If no FakeMode
    is found on the graph, the pass is skipped with a warning.

    Args:
        gm: The GraphModule to transform.
        example_inputs: Unused; accepted for compatibility with the
            standard pass interface.

    Returns:
        The same GraphModule, mutated in-place with the graph recompiled.

    Example::

        # Before: functional add
        #   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%x, 1), ...)
        #   return (%add,)

        # After: in-place add (when safe)
        #   %add_ : [num_users=1] = call_function[target=torch.ops.aten.add_.Tensor](args = (%x, 1), ...)
        #   return (%add_,)
    """
    logger.debug("starting reinplacement pass")
    logger.debug("graph before reinplacement pass: %s", gm.graph)

    from torch._guards import detect_fake_mode
    from torch._inductor.virtualized import V

    # TODO: remove this check after https://github.com/pytorch/pytorch/pull/159523 is in
    # a PyTorch release.
    if "gm" in signature(FakeTensorUpdater).parameters:
        # pyrefly: ignore[bad-argument-type]
        fake_tensor_updater = FakeTensorUpdater(gm)
    else:
        fake_tensor_updater = FakeTensorUpdater(gm.graph)

    fake_mode = detect_fake_mode(
        [node.meta.get("val") for node in gm.graph.nodes if "val" in node.meta]
    )

    if fake_mode is not None:
        with V.set_fake_mode(fake_mode):  # type: ignore[arg-type]
            reinplace_inplaceable_ops(fake_tensor_updater, gm.graph)
            fake_tensor_updater.incremental_update()
    else:
        logger.warning("No fake mode detected, skipping reinplacement pass")

    gm.recompile()
    logger.debug("finished reinplacement pass")
    logger.debug("graph after reinplacement pass: %s", gm.graph)
    return gm


def _replace_in_output_args(output_node, old_node, new_node):
    """Replace old_node with new_node in output args, handling nested structures."""

    def replace_in_structure(obj):
        if obj is old_node:
            return new_node
        elif isinstance(obj, (list, tuple)):
            result = [replace_in_structure(item) for item in obj]
            return type(obj)(result)
        elif isinstance(obj, dict):
            return {k: replace_in_structure(v) for k, v in obj.items()}
        else:
            return obj

    output_node.args = replace_in_structure(output_node.args)


def strip_with_effects_pass(
    gm: torch.fx.GraphModule, example_inputs=None
) -> torch.fx.GraphModule:
    """Remove ``with_effects`` higher-order-op wrappers from torch.comms ops.

    ``torch.compile`` wraps ops that have side effects in
    ``with_effects(token, op, *args)`` to sequence them via token
    threading. This pass strips those wrappers for torchcomms ops,
    replacing them with direct ``call_function`` nodes so the graph can
    be executed outside of Inductor without the token overhead.

    Non-torchcomms ``with_effects`` calls are left untouched.

    Args:
        gm: The GraphModule to transform.
        example_inputs: Unused; accepted for compatibility with the
            standard pass interface.

    Returns:
        The same GraphModule, mutated in-place with the graph recompiled.

    Example::

        # Before: torchcomms op wrapped in with_effects
        #   %token0 = call_function[target=aten._make_dep_token.default]()
        #   %with_effects = call_function[target=with_effects](%token0, torch.comms.allreduce, %x)
        #   %getitem_0 = call_function[target=operator.getitem](%with_effects, 0)  # token
        #   %getitem_1 = call_function[target=operator.getitem](%with_effects, 1)  # result
        #   return (%getitem_0, %getitem_1)

        # After: direct call, wrappers removed
        #   %allreduce = call_function[target=torch.comms.allreduce](%x)
        #   return (%allreduce, %allreduce)
    """
    graph = gm.graph

    logger.debug("starting strip_with_effects pass for torchcomms ops")
    logger.debug("graph before strip_with_effects pass: %s", graph)

    nodes_to_erase = []
    replacements_made = 0

    # iterate in reverse order so downstream with_effects nodes (which consume tokens
    # from upstream ones) are processed first, allowing less strenuous token chain cleanup
    for node in reversed(list(graph.nodes)):
        if node.op != "call_function" or node.target is not with_effects:
            continue

        # with_effects(token, op, *args)
        wrapped_op = node.args[1]

        if not str(wrapped_op).startswith("torchcomms."):
            continue

        logger.debug("Found with_effects wrapping torchcomms op: %s", wrapped_op)

        actual_args = node.args[2:]
        actual_kwargs = node.kwargs

        with graph.inserting_before(node):
            direct_call = graph.call_function(wrapped_op, actual_args, actual_kwargs)

        for user in list(node.users):
            if user.op != "call_function" or user.target != operator.getitem:
                continue

            getitem_index = user.args[1]
            if getitem_index > 0:
                direct_call.meta = user.meta.copy()
                user.replace_all_uses_with(direct_call)
            else:
                for token_user in list(user.users):
                    if token_user.op == "output":
                        _replace_in_output_args(token_user, user, direct_call)
            nodes_to_erase.append(user)

        nodes_to_erase.append(node)
        replacements_made += 1

    remaining = list(nodes_to_erase)
    while remaining:
        made_progress = False
        still_remaining = []
        for node in remaining:
            if len(node.users) == 0:
                graph.erase_node(node)
                made_progress = True
            else:
                still_remaining.append(node)
        remaining = still_remaining
        if not made_progress:
            for node in remaining:
                logger.warning(
                    "Could not erase node %s, still has users: %s",
                    node,
                    list(node.users),
                )
            break

    gm.recompile()

    logger.debug(
        "finished strip_with_effects pass: removed %s with_effects wrappers",
        replacements_made,
    )
    logger.debug("graph after strip_with_effects pass: %s", gm.graph)

    return gm
