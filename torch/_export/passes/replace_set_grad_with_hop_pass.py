# mypy: allow-untyped-defs
from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torch
from torch._higher_order_ops.wrap import (
    tag_activation_checkpoint,
    wrap_with_set_grad_enabled,
)
from torch.fx.passes.split_module import split_module

from ..utils import node_inline_, nodes_filter, nodes_first, nodes_map
from .replace_with_hop_pass_util import (
    _replace_with_hop_helper,
    _replace_with_hop_pass_helper,
    _sequential_split_and_maybe_inline_subgraphs_helper,
)


if TYPE_CHECKING:
    from torch.export.graph_signature import ExportGraphSignature


_REPEATED_STACK_FRAME_RE = re.compile(
    r"\s*\[Previous line repeated (\d+) more times?\]"
)


def _is_set_grad_enabled_node(node: torch.fx.Node) -> torch.fx.Node | bool:
    return (
        node
        and node.op == "call_function"
        and node.target is torch._C._set_grad_enabled
    )


def _is_set_grad_enabled_sub_mod(
    node: torch.fx.Node, omit_if_same_with_ambient: bool = False
) -> bool | torch.Tensor:
    if node.op == "call_module":
        if not isinstance(node.target, str):
            raise AssertionError(f"expected str target, got {type(node.target)}")
        subgm = getattr(node.graph.owning_module, node.target)
        first_non_ph = nodes_first(
            subgm.graph.nodes, lambda node: node.op != "placeholder"
        )
        if (
            first_non_ph
            and first_non_ph.op == "call_function"
            and first_non_ph.target is torch._C._set_grad_enabled
        ):
            return (
                first_non_ph.args[0] != torch.is_grad_enabled()  # type: ignore[bad-return]
                if omit_if_same_with_ambient
                else True
            )
    return False


def _get_reentrant_checkpoint_stack_trace_key(
    stack_trace: object, frame_marker: str
) -> str | None:
    if not isinstance(stack_trace, str):
        return None
    stack_trace = stack_trace.replace("\\", "/")
    if (
        "torch/utils/checkpoint.py" not in stack_trace
        or frame_marker not in stack_trace
    ):
        return None

    lines = []
    for line in stack_trace.splitlines():
        repeat_match = _REPEATED_STACK_FRAME_RE.fullmatch(line)
        if repeat_match is None:
            lines.append(line)
            continue

        frame_start = next(
            (
                i
                for i in range(len(lines) - 1, -1, -1)
                if lines[i].lstrip().startswith("File ")
            ),
            None,
        )
        if frame_start is None:
            return None
        repeated_frame = lines[frame_start:]
        for _ in range(int(repeat_match.group(1))):
            lines.extend(repeated_frame)

    for i in range(len(lines) - 2, -1, -1):
        line = lines[i]
        if (
            "torch/utils/checkpoint.py" in line
            and "in forward" in line
            and frame_marker in lines[i + 1]
        ):
            # Key by the complete caller stack. Its depth distinguishes nested
            # checkpoints, while repeated calls from the same source line are
            # still separated by checkpoint boundary nodes.
            return "\n".join(lines[:i]) if i > 0 else line
    return None


def _get_reentrant_checkpoint_boundary_key(node: torch.fx.Node) -> str | None:
    if not _is_set_grad_enabled_node(node):
        return None
    return _get_reentrant_checkpoint_stack_trace_key(
        node.meta.get("stack_trace"), "with torch.no_grad():"
    )


def _get_reentrant_checkpoint_body_key(node: torch.fx.Node) -> str | None:
    return _get_reentrant_checkpoint_stack_trace_key(
        node.meta.get("stack_trace"), "outputs = run_function(*args)"
    )


def _split_set_grad_and_reentrant_checkpoints(
    gm: torch.fx.GraphModule,
) -> torch.fx.GraphModule:
    split_map = {}
    split_id = 0
    active_checkpoint_key = None
    split_after_checkpoint = False

    # Keep each reentrant checkpoint's outer no_grad enter/body/exit in one
    # submodule so it can be replaced by a single AC HOP. Nested checkpoint
    # boundaries are handled recursively after the outer AC HOP is installed.
    for node in gm.graph.nodes:
        started_new_split = False
        if split_after_checkpoint:
            split_id += 1
            split_after_checkpoint = False
            started_new_split = True

        boundary_key = _get_reentrant_checkpoint_boundary_key(node)
        if active_checkpoint_key is not None:
            split_map[node] = split_id
            if boundary_key == active_checkpoint_key:
                active_checkpoint_key = None
                split_after_checkpoint = True
            continue

        if boundary_key is not None:
            if not started_new_split:
                split_id += 1
            active_checkpoint_key = boundary_key
        elif _is_set_grad_enabled_node(node):
            split_id += 1
        split_map[node] = split_id

    if active_checkpoint_key is not None:
        raise AssertionError("expected a matching reentrant checkpoint boundary")

    new_gm = split_module(
        gm,
        gm,
        lambda node: split_map[node],
        keep_original_order=True,
        keep_original_node_name=True,
    )
    new_gm.graph._codegen = gm.graph._codegen
    new_gm.recompile()
    return new_gm


def _get_reentrant_checkpoint_sub_mod_key(node: torch.fx.Node) -> str | None:
    if not _is_set_grad_enabled_sub_mod(node):
        return None
    if not isinstance(node.target, str):
        raise AssertionError(f"expected str target, got {type(node.target)}")
    subgm = getattr(node.graph.owning_module, node.target)
    first_non_ph = nodes_first(
        subgm.graph.nodes, lambda sub_node: sub_node.op != "placeholder"
    )
    if first_non_ph is None:
        return None
    boundary_key = _get_reentrant_checkpoint_boundary_key(first_non_ph)
    if boundary_key is None:
        return None
    boundary_nodes = [
        sub_node
        for sub_node in subgm.graph.nodes
        if _get_reentrant_checkpoint_boundary_key(sub_node) == boundary_key
    ]
    if len(boundary_nodes) != 2:
        return None
    for sub_node in subgm.graph.nodes:
        if _get_reentrant_checkpoint_body_key(sub_node) == boundary_key:
            return boundary_key
    return None


def _replace_with_hop(
    node: torch.fx.Node,
    *,
    wrap_hoo=wrap_with_set_grad_enabled,
    include_enter_block_args: bool = True,
) -> None:
    if node.op != "call_module":
        raise AssertionError(f"expected call_module op, got {node.op}")
    graph: torch.fx.Graph = node.graph
    if graph.owning_module is None:
        raise AssertionError("graph.owning_module must not be None")
    gm: torch.fx.GraphModule = graph.owning_module
    if not isinstance(node.target, str):
        raise AssertionError(f"expected str target, got {type(node.target)}")
    sub_gm = getattr(gm, node.target)
    sub_graph = sub_gm.graph
    set_grad_nodes = nodes_filter(sub_graph.nodes, _is_set_grad_enabled_node)
    if len(set_grad_nodes) > 0:
        if len(set_grad_nodes) != 1:
            raise AssertionError(
                f"expected exactly 1 set_grad node, got {len(set_grad_nodes)}"
            )
        set_grad_node = set_grad_nodes[0]
        _replace_with_hop_helper(
            node,
            set_grad_node,
            wrap_hoo,
            include_enter_block_args=include_enter_block_args,
        )
        sub_graph.erase_node(set_grad_node)


def _replace_reentrant_checkpoint_with_hop(
    node: torch.fx.Node, checkpoint_key: str
) -> None:
    if node.op != "call_module":
        raise AssertionError(f"expected call_module op, got {node.op}")
    graph: torch.fx.Graph = node.graph
    if graph.owning_module is None:
        raise AssertionError("graph.owning_module must not be None")
    gm: torch.fx.GraphModule = graph.owning_module
    if not isinstance(node.target, str):
        raise AssertionError(f"expected str target, got {type(node.target)}")
    sub_gm = getattr(gm, node.target)
    sub_graph = sub_gm.graph
    boundary_nodes = [
        sub_node
        for sub_node in sub_graph.nodes
        if _get_reentrant_checkpoint_boundary_key(sub_node) == checkpoint_key
    ]
    if len(boundary_nodes) != 2:
        raise AssertionError(
            f"expected exactly 2 checkpoint boundary nodes, got {len(boundary_nodes)}"
        )
    user_set_grad_nodes = [
        sub_node
        for sub_node in sub_graph.nodes
        if _is_set_grad_enabled_node(sub_node)
        and _get_reentrant_checkpoint_boundary_key(sub_node) is None
    ]
    if user_set_grad_nodes:
        # Context exits restore the checkpoint's outer no_grad state in the
        # trace. That stale False value cannot be replayed in the AC HOP's
        # grad-enabled recomputation without structured enter/exit provenance.
        raise NotImplementedError(
            "non-strict export of a reentrant checkpoint does not support "
            "grad-mode changes inside the checkpointed function"
        )
    _replace_with_hop_helper(
        node,
        boundary_nodes[0],
        tag_activation_checkpoint,
        include_enter_block_args=False,
    )
    for boundary_node in boundary_nodes:
        sub_graph.erase_node(boundary_node)


def _remove_set_grad_and_inline(node: torch.fx.Node) -> None:
    if node.op != "call_module":
        raise AssertionError(f"expected call_module op, got {node.op}")
    graph: torch.fx.Graph = node.graph
    if graph.owning_module is None:
        raise AssertionError("graph.owning_module must not be None")
    gm: torch.fx.GraphModule = graph.owning_module
    if not isinstance(node.target, str):
        raise AssertionError(f"expected str target, got {type(node.target)}")
    sub_gm = getattr(gm, node.target)
    sub_graph = sub_gm.graph
    nodes_map(
        sub_graph.nodes,
        lambda n: sub_graph.erase_node(n) if _is_set_grad_enabled_node(n) else n,
    )
    node_inline_(node)


def _sequential_split_and_maybe_inline_subgraphs(
    gm: torch.fx.GraphModule, graph_signature: ExportGraphSignature | None
) -> tuple[torch.fx.GraphModule, ExportGraphSignature | None]:
    """
    Helper function for replace_set_grad_with_hop_pass().
    Split the graph module into multiple subgraphs based on the set_grad_enabled nodes.
    For each subgraph, decides whether to construct a HOO subgraph, or inline the calls
    back into the parent graph module.
    """
    need_replacing = any(_is_set_grad_enabled_node(node) for node in gm.graph.nodes)
    if not need_replacing:
        return gm, graph_signature

    # Splitting returns a new graph module that could have different output args
    # names. We need to fix the graph signature.
    new_gm = _split_set_grad_and_reentrant_checkpoints(gm)

    def _maybe_inline_or_replace_with_hop(node: torch.fx.Node):
        if _is_set_grad_enabled_sub_mod(node, omit_if_same_with_ambient=True):
            checkpoint_key = _get_reentrant_checkpoint_sub_mod_key(node)
            if checkpoint_key is not None:
                _replace_reentrant_checkpoint_with_hop(node, checkpoint_key)
            else:
                _replace_with_hop(node)
        else:
            _remove_set_grad_and_inline(node)

    return _sequential_split_and_maybe_inline_subgraphs_helper(
        new_gm, graph_signature, _maybe_inline_or_replace_with_hop
    )


def replace_set_grad_with_hop_pass(
    gm: torch.fx.GraphModule, graph_signature: ExportGraphSignature | None
) -> tuple[torch.fx.GraphModule, ExportGraphSignature | None]:
    """
    Split gm into sub-graph-modules using `sequential_split_and_maybe_inline_subgraphs`, and
    then recursively call itself on each of the submodules.
    """
    return _replace_with_hop_pass_helper(
        gm,
        graph_signature,
        _sequential_split_and_maybe_inline_subgraphs,
    )
