"""Detection of q-by-kv contractions inside a score_mod subgraph.

A score_mod may contract a query-indexed vector against a key-indexed vector, as
MLA's decoupled RoPE term does:

    score + dot(query_rope[b, h, q_idx], key_rope[b, h, kv_idx])

Lowered naively this is a reduction per score element, which the pointwise
subgraph lowering cannot represent at all; the only way to compile it today is to
raise unroll_reductions_threshold above the contracted extent so it expands into
that many pointwise multiply-adds per element of the score tile.

Such a contraction factorizes over the score tile: the [BLOCK_M, BLOCK_N, R]
intermediate the unrolled form materializes is just
    A_tile[BLOCK_M, R] @ B_tile[R, BLOCK_N]
so the template can compute it with one extra tl.dot and never form the rank-3
intermediate. This module recognizes the pattern; the lowering hands the operands
to the template as tiles.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch


aten = torch.ops.aten
prims = torch.ops.prims

# score, b, h, q_idx, kv_idx -- everything after these is a captured tensor.
NUM_SCORE_MOD_FIXED_ARGS = 5

_Q_IDX_ARG = 3
_KV_IDX_ARG = 4

# Name of the placeholder that replaces the contraction; the template supplies a
# [BLOCK_M, BLOCK_N] tile for it, exactly like the "score" placeholder.
CONTRACTION_INPUT_NAME = "qk_extra"

# tl.dot needs a contracted extent of at least 16.
MIN_CONTRACT_EXTENT_ROUNDED = 16


@dataclass(frozen=True)
class ScoreModContraction:
    """One recognized dot(A[b, h, q_idx], B[b, h, kv_idx]) term.

    ``sum_node`` is the node whose value the template supplies instead, as a
    [BLOCK_M, BLOCK_N] tile. ``q_arg`` / ``kv_arg`` are placeholder positions in
    the subgraph argument list, so the lowering can pick the matching operands
    out of the captured tensors it already has.
    """

    sum_node: torch.fx.Node
    q_arg: int
    kv_arg: int
    contract_extent: int
    out_dtype: torch.dtype

    @property
    def contract_extent_rounded(self) -> int:
        return max(
            MIN_CONTRACT_EXTENT_ROUNDED,
            1 << (self.contract_extent - 1).bit_length(),
        )


def _strip_converts(node: Any) -> tuple[Any, torch.dtype | None]:
    """Peel convert_element_type wrappers, returning the widest dtype seen."""
    dtype: torch.dtype | None = None
    while (
        isinstance(node, torch.fx.Node)
        and node.op == "call_function"
        and node.target is prims.convert_element_type.default
    ):
        if dtype is None and isinstance(node.args[1], torch.dtype):
            dtype = node.args[1]
        node = node.args[0]
    return node, dtype


def _placeholder_position(node: Any, placeholders: list[torch.fx.Node]) -> int | None:
    if not isinstance(node, torch.fx.Node) or node.op != "placeholder":
        return None
    return placeholders.index(node)


def _match_indexed_capture(
    node: Any,
    placeholders: list[torch.fx.Node],
    seq_arg: int,
) -> tuple[int, int] | None:
    """Match ``capture[b, h, <seq_arg>]`` -> (capture arg position, extent).

    Requires the batch and head indices to be the score_mod's own b/h arguments.
    A transformed head index (GQA's ``h // n``, say) is deliberately rejected:
    the template loads whole tiles, so a head index that is not the raw argument
    would need its own broadcast handling.
    """
    if not isinstance(node, torch.fx.Node) or node.op != "call_function":
        return None
    if node.target is not aten.index.Tensor:
        return None
    if len(node.args) != 2:
        return None

    base, indices = node.args
    if not isinstance(indices, (list, tuple)) or len(indices) != 3:
        return None

    base_pos = _placeholder_position(base, placeholders)
    if base_pos is None or base_pos < NUM_SCORE_MOD_FIXED_ARGS:
        return None

    if [_placeholder_position(i, placeholders) for i in indices] != [1, 2, seq_arg]:
        return None

    val = placeholders[base_pos].meta.get("val")
    if val is None or val.dim() != 4:
        return None
    extent = val.shape[-1]
    # A symbolic trailing extent would make the tile shape dynamic; bail.
    if not isinstance(extent, int):
        return None

    return base_pos, extent


def detect_score_mod_contraction(
    graph_module: torch.fx.GraphModule,
) -> ScoreModContraction | None:
    """Recognize a single q-by-kv contraction in a score_mod subgraph.

    Returns None whenever the graph is not exactly the supported shape. Bailing
    is always safe: the caller keeps today's unroll-or-fail behaviour.
    """
    graph = graph_module.graph
    placeholders = [n for n in graph.nodes if n.op == "placeholder"]
    if len(placeholders) <= NUM_SCORE_MOD_FIXED_ARGS:
        return None

    found: list[ScoreModContraction] = []
    for node in graph.nodes:
        if node.op != "call_function":
            continue
        # torch.dot over the trailing dim arrives as mul + full sum.
        keepdim = len(node.args) > 2 and node.args[2]
        if node.target is aten.sum.default:
            mul, accum_dtype = _strip_converts(node.args[0])
        elif (
            node.target is aten.sum.dim_IntList
            and node.args[1] in ([-1], [0])
            and not keepdim
        ):
            mul, accum_dtype = _strip_converts(node.args[0])
        else:
            continue

        if (
            not isinstance(mul, torch.fx.Node)
            or mul.op != "call_function"
            or mul.target is not aten.mul.Tensor
        ):
            continue

        lhs, lhs_dtype = _strip_converts(mul.args[0])
        rhs, rhs_dtype = _strip_converts(mul.args[1])

        q_match = _match_indexed_capture(lhs, placeholders, _Q_IDX_ARG)
        kv_match = _match_indexed_capture(rhs, placeholders, _KV_IDX_ARG)
        if q_match is None or kv_match is None:
            # Operands may appear in either order.
            q_match = _match_indexed_capture(rhs, placeholders, _Q_IDX_ARG)
            kv_match = _match_indexed_capture(lhs, placeholders, _KV_IDX_ARG)
        if q_match is None or kv_match is None:
            continue

        (q_arg, q_extent), (kv_arg, kv_extent) = q_match, kv_match
        if q_extent != kv_extent or q_arg == kv_arg:
            continue

        out_val = node.meta.get("val")
        found.append(
            ScoreModContraction(
                sum_node=node,
                q_arg=q_arg,
                kv_arg=kv_arg,
                contract_extent=q_extent,
                out_dtype=(
                    out_val.dtype
                    if out_val is not None
                    else (accum_dtype or lhs_dtype or rhs_dtype or torch.float32)
                ),
            )
        )

    # More than one contraction would need several extra tl.dots and a matching
    # number of template operands; not handled yet.
    if len(found) != 1:
        return None
    return found[0]


def rewrite_score_mod_for_contraction(
    graph_module: torch.fx.GraphModule,
) -> tuple[torch.fx.GraphModule, ScoreModContraction] | None:
    """Replace a recognized contraction with a trailing placeholder.

    The returned module is a copy: the caller's graph is shared with the
    backward lowering and with the other flex backends, none of which know about
    the fused input. The new placeholder goes last, so it maps onto one extra
    argument appended after the captured tensors.
    """
    if detect_score_mod_contraction(graph_module) is None:
        return None

    # Detected again on the copy so the returned node belongs to the graph the
    # caller gets; detection is much cheaper than copying every score_mod graph.
    fused = torch.fx.GraphModule(graph_module, copy.deepcopy(graph_module.graph))
    contraction = detect_score_mod_contraction(fused)
    if contraction is None:
        raise AssertionError("contraction detection is not stable under graph copy")

    graph = fused.graph
    placeholders = [n for n in graph.nodes if n.op == "placeholder"]
    with graph.inserting_after(placeholders[-1]):
        tile = graph.placeholder(CONTRACTION_INPUT_NAME)
    tile.meta.update(contraction.sum_node.meta)

    contraction.sum_node.replace_all_uses_with(tile)
    graph.erase_node(contraction.sum_node)
    # Drops the now-dead index/mul nodes. Placeholders are impure to fx, so the
    # operands stay and the argument positions the caller was given hold.
    graph.eliminate_dead_code()
    fused.recompile()
    return fused, contraction
