# mypy: allow-untyped-defs
"""
Optimization pass to remove redundant precision emulation casts.

When emulate_precision_casts is enabled, we insert precision emulation patterns:
    to_dtype(x, lowp, use_compute_types=False)  -> actual cast to bf16/fp16
    to_dtype(x, lowp)  -> with use_compute_types=True (default), casts to fp32

This creates chains like bf16(literal)->bf16(fp32)->bf16(literal)->bf16(fp32)
which can crash Triton's PassManager.

This pass identifies and removes redundant patterns by detecting consecutive
precision emulation pairs that can be collapsed.
"""

from typing import Any

import torch
from torch.utils._ordered_set import OrderedSet

from .loop_body import LoopBody


LOWP_FP_DTYPES = (torch.bfloat16, torch.float16)


def _is_to_dtype_node(node: Any) -> bool:
    """Check if a node is a to_dtype call."""
    return (
        hasattr(node, "op")
        and hasattr(node, "target")
        and node.op == "call_method"
        and node.target == "to_dtype"
    )


def _get_dtype_arg(node: Any) -> Any:
    """Get the dtype argument from a to_dtype node."""
    if len(node.args) >= 3:
        return node.args[2]
    return None


def _get_input_node(node: Any) -> Any:
    """Get the input node from a to_dtype node."""
    if len(node.args) >= 2:
        return node.args[1]
    return None


def _get_use_compute_types(node: Any) -> bool:
    """Get the use_compute_types flag from a to_dtype node."""
    return node.kwargs.get("use_compute_types", True)


def _produces_fp32(node: Any) -> bool:
    """
    Check if a node produces float32 output.

    This includes:
    - to_dtype nodes with use_compute_types=True (computes as fp32)
    - Arithmetic operations which compute in fp32
    - Nodes that are not to_dtype lowp casts

    Does NOT include:
    - Placeholder nodes (may represent bf16 input data)
    - to_dtype nodes with use_compute_types=False (produce bf16/fp16)
    """
    if not hasattr(node, "op"):
        return True  # Non-node types (constants, etc.) treated as fp32

    # Placeholder nodes represent input data - could be bf16
    if node.op == "placeholder":
        return False

    if _is_to_dtype_node(node):
        # to_dtype with uct=True produces fp32 (compute type)
        # to_dtype with uct=False and lowp dtype produces bf16/fp16
        if _is_lowp_dtype(node) and not _get_use_compute_types(node):
            return False  # produces bf16/fp16
        return True  # produces fp32

    # Arithmetic operations (mul, add, truediv, sqrt, etc.) produce fp32
    # get_attr nodes (like "ops") are not actual values
    if node.op == "get_attr":
        return False

    # Call method nodes (mul, add, sqrt, etc.) produce fp32 in compute
    if node.op == "call_method":
        return True

    # Default: conservatively assume fp32 in compute context
    return True


def _is_lowp_dtype(node: Any) -> bool:
    """Check if this to_dtype targets a low-precision dtype."""
    if not _is_to_dtype_node(node):
        return False
    dtype = _get_dtype_arg(node)
    return dtype in LOWP_FP_DTYPES


def remove_redundant_precision_casts(loop_body: LoopBody, debug: bool = False) -> None:
    """
    Removes redundant precision cast chains that can crash Triton's PassManager.

    Patterns removed:
    1. compute(literal(compute(x))) - True->False->True becomes just inner_input
    2. compute(compute(x)) - Two consecutive compute casts (both fp32)
    3. literal(compute(literal(x))) - False->True->False, the middle is redundant
    4. literal(literal(x)) - Two consecutive literal casts
    """
    all_graphs = [loop_body.root_block.graph]
    all_graphs.extend(block.graph for block in loop_body.subblocks.values())

    for graph in all_graphs:
        if debug:
            print("=" * 60)
            print("BEFORE OPTIMIZATION")
            print("=" * 60)
            for n in graph.nodes:
                if _is_to_dtype_node(n):
                    dtype = _get_dtype_arg(n)
                    uct = _get_use_compute_types(n)
                    inp = _get_input_node(n)
                    inp_name = inp.name if inp and hasattr(inp, "name") else str(inp)
                    user_names = [u.name for u in n.users]
                    print(
                        f"  {n.name}: to_dtype({inp_name}, {dtype}, uct={uct}), users={user_names}"
                    )

        total_patterns = 0

        # Pass 1: Remove compute(literal(fp32_producer)) patterns
        # Pattern: True -> False -> fp32_value, collapse to fp32_value
        # This handles: fp32 <- bf16 <- fp32, which is redundant
        #
        # Key insight: if inner_input produces fp32, then:
        # inner_input(fp32) -> literal_bf16 -> compute_fp32 is pointless
        # We're casting fp32 to bf16 and back to fp32 - just use original fp32
        processed_nodes: set = OrderedSet()
        modified = True
        while modified:
            modified = False
            for node in list(graph.nodes):
                if node in processed_nodes or len(node.users) == 0:
                    continue
                if not _is_lowp_dtype(node):
                    continue
                if not _get_use_compute_types(node):
                    continue  # node must be compute (True) - produces fp32

                inner_node = _get_input_node(node)
                if not _is_to_dtype_node(inner_node):
                    continue
                if not _is_lowp_dtype(inner_node):
                    continue
                if _get_use_compute_types(inner_node):
                    continue  # inner must be literal (False) - produces bf16

                inner_input = _get_input_node(inner_node)

                # Check if inner_input produces fp32
                # If so, the chain fp32 -> bf16 -> fp32 is redundant
                if _produces_fp32(inner_input):
                    # Pattern: compute(literal(fp32_value)) -> replace with inner_input
                    if debug:
                        inp_name = (
                            inner_input.name
                            if hasattr(inner_input, "name")
                            else str(inner_input)
                        )
                        print(f"Pass1: {node.name} <- {inner_node.name} <- {inp_name}")
                    total_patterns += 1
                    node.replace_all_uses_with(inner_input)
                    processed_nodes.add(node)

                    if len(node.users) == 0:
                        graph.erase_node(node)
                    if len(inner_node.users) == 0:
                        graph.erase_node(inner_node)

                    modified = True
                    break

        # Pass 2: Remove False -> True -> False patterns
        # literal(compute(literal(x))) - the compute in the middle is redundant
        processed_nodes.clear()
        modified = True
        while modified:
            modified = False
            for node in list(graph.nodes):
                if node in processed_nodes or len(node.users) == 0:
                    continue
                if not _is_lowp_dtype(node):
                    continue
                if _get_use_compute_types(node):
                    continue  # node is literal (False)

                inner_node = _get_input_node(node)
                if not _is_to_dtype_node(inner_node):
                    continue
                if not _is_lowp_dtype(inner_node):
                    continue
                if not _get_use_compute_types(inner_node):
                    continue  # inner is compute (True)

                inner_input = _get_input_node(inner_node)
                if not _is_to_dtype_node(inner_input):
                    continue
                if not _is_lowp_dtype(inner_input):
                    continue
                if _get_use_compute_types(inner_input):
                    continue  # inner_input is literal (False)

                # Pattern: literal(compute(literal(x))) -> replace with inner_input
                if debug:
                    print(
                        f"Pass2: {node.name} <- {inner_node.name} <- {inner_input.name}"
                    )
                total_patterns += 1
                node.replace_all_uses_with(inner_input)
                processed_nodes.add(node)

                if len(node.users) == 0:
                    graph.erase_node(node)
                if len(inner_node.users) == 0:
                    graph.erase_node(inner_node)

                modified = True
                break

        # Pass 3: Remove consecutive compute type casts
        # compute(compute(x)) - both resolve to fp32, outer is redundant
        processed_nodes.clear()
        modified = True
        while modified:
            modified = False
            for node in list(graph.nodes):
                if node in processed_nodes or len(node.users) == 0:
                    continue
                if not _is_to_dtype_node(node):
                    continue
                if not _get_use_compute_types(node):
                    continue

                inner_node = _get_input_node(node)
                if not _is_to_dtype_node(inner_node):
                    continue
                if not _get_use_compute_types(inner_node):
                    continue

                # Both are compute type casts (resolve to fp32)
                if debug:
                    print(f"Pass3: {node.name} <- {inner_node.name}")
                total_patterns += 1
                node.replace_all_uses_with(inner_node)
                processed_nodes.add(node)

                if len(node.users) == 0:
                    graph.erase_node(node)

                modified = True
                break

        # Pass 4: Remove consecutive literal type casts
        # literal(literal(x)) - both are bf16, outer is redundant
        processed_nodes.clear()
        modified = True
        while modified:
            modified = False
            for node in list(graph.nodes):
                if node in processed_nodes or len(node.users) == 0:
                    continue
                if not _is_lowp_dtype(node):
                    continue
                if _get_use_compute_types(node):
                    continue  # node is literal

                inner_node = _get_input_node(node)
                if not _is_to_dtype_node(inner_node):
                    continue
                if not _is_lowp_dtype(inner_node):
                    continue
                if _get_use_compute_types(inner_node):
                    continue  # inner is also literal

                # Both are literal bf16 casts, outer is redundant
                if debug:
                    print(f"Pass4: {node.name} <- {inner_node.name}")
                total_patterns += 1
                node.replace_all_uses_with(inner_node)
                processed_nodes.add(node)

                if len(node.users) == 0:
                    graph.erase_node(node)

                modified = True
                break

        if debug:
            print(f"Total patterns fixed: {total_patterns}")
            print("=" * 60)
            print("AFTER OPTIMIZATION")
            print("=" * 60)
            for n in graph.nodes:
                if _is_to_dtype_node(n):
                    dtype = _get_dtype_arg(n)
                    uct = _get_use_compute_types(n)
                    inp = _get_input_node(n)
                    inp_name = inp.name if inp and hasattr(inp, "name") else str(inp)
                    print(
                        f"  {n.name}: to_dtype({inp_name}, {dtype}, uct={uct}), users={len(n.users)}"
                    )
