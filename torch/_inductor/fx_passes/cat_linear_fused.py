"""Pre-grad FX pass that rewrites ``F.linear(cat([...], dim=-1), W, b)``
into a reduce-sum of per-piece ``F.linear`` calls on contiguous weight
slices, eliminating the ``cat`` materialisation in both forward and
backward.

Off by default; opt in by setting
``torch._inductor.config.pre_grad_custom_pass = cat_linear_fused_pre_grad_pass``
(or composing it with other custom passes).
"""

from __future__ import annotations

import logging
import operator

import torch
import torch.fx as fx
from torch._dynamo.utils import counters
from torch.utils._ordered_set import OrderedSet


log = logging.getLogger(__name__)


MAX_TOTAL_CAT_WIDTH = 384
MIN_PIECE_WIDTH = 8
MAX_PARTS = 3

# `mul`-parented parts are gated patterns that another fusion (block_cat_fused)
# is responsible for; skip them here to avoid overlap.
_REJECT_PARENT_TARGETS = OrderedSet([torch.mul, operator.mul])

# Dynamo sometimes inlines F.linear into torch._C._nn.linear on hot paths.
_LINEAR_TARGETS = [
    t
    for t in (
        torch.nn.functional.linear,
        getattr(torch._C._nn, "linear", None),
    )
    if t is not None
]


def _val_of(node):
    if not hasattr(node, "meta"):
        return None
    # Use explicit `is None` instead of `or` here: when meta["val"] is a
    # Tensor (which can happen if the matcher is invoked on a post-grad
    # graph or a test graph that populates meta["val"] directly), the
    # `or` short-circuit calls `Tensor.__bool__`, which raises for any
    # multi-element tensor.
    v = node.meta.get("val")
    if v is None:
        v = node.meta.get("example_value")
    return v


def _shape_of(node):
    val = _val_of(node)
    if val is None:
        return None
    try:
        return tuple(int(s) for s in val.shape)
    except Exception:
        return None


def _is_cat(node):
    if node.op != "call_function":
        return False
    return node.target in (
        torch.cat,
        getattr(torch, "concatenate", None),
        getattr(torch, "concat", None),
    )


def _is_linear(node):
    return node.op == "call_function" and node.target in _LINEAR_TARGETS


def _try_match_cat_linear(ln):
    """Match `linear(cat([parts...], dim=-1), W, bias)`. Returns
    (parts, weight, bias_or_None, K_offsets) or None.
    """
    if len(ln.args) < 2:
        return None
    cat_in = ln.args[0]
    weight = ln.args[1]
    bias = ln.args[2] if len(ln.args) >= 3 else ln.kwargs.get("bias")

    if not isinstance(cat_in, fx.Node) or not _is_cat(cat_in):
        return None
    if not cat_in.args:
        return None
    parts_arg = cat_in.args[0]
    if not isinstance(parts_arg, (list, tuple)):
        return None
    parts = [p for p in parts_arg if isinstance(p, fx.Node)]
    if not (2 <= len(parts) <= MAX_PARTS):
        return None
    if len(parts) != len(parts_arg):
        return None

    cat_dim = cat_in.args[1] if len(cat_in.args) >= 2 else cat_in.kwargs.get("dim", 0)
    if isinstance(cat_dim, str):
        try:
            cat_dim = int(cat_dim)
        except ValueError:
            return None
    cat_shape = _shape_of(cat_in)
    if cat_shape is None or len(cat_shape) < 2:
        return None
    rank = len(cat_shape)
    if not isinstance(cat_dim, int):
        return None
    cat_dim_norm = cat_dim + rank if cat_dim < 0 else cat_dim
    if cat_dim_norm != rank - 1:
        return None

    for p in parts:
        if p.op == "call_function" and p.target in _REJECT_PARENT_TARGETS:
            return None

    K_total = int(cat_shape[-1])
    if K_total > MAX_TOTAL_CAT_WIDTH:
        return None

    K_sum = 0
    K_offsets = [0]
    for p in parts:
        sh = _shape_of(p)
        if sh is None or len(sh) != rank:
            return None
        K_i = int(sh[-1])
        if K_i < MIN_PIECE_WIDTH:
            return None
        K_sum += K_i
        K_offsets.append(K_sum)
    if K_sum != K_total:
        return None

    w_shape = _shape_of(weight)
    if w_shape is None or len(w_shape) != 2 or int(w_shape[-1]) != K_total:
        return None

    return parts, weight, bias, K_offsets


def _slice_then_contiguous(graph, weight, start, stop, weight_val):
    """Emit aten.slice + aten.clone(memory_format=contiguous) on the weight
    so the per-piece F.linear sees a dense buffer (hipBLASLt/cuBLASLt fast
    path prefers contiguous inputs).
    """
    slice_node = graph.call_function(
        torch.ops.aten.slice.Tensor, args=(weight, 1, start, stop)
    )
    if weight_val is not None:
        try:
            slice_node.meta["val"] = torch.ops.aten.slice.Tensor(
                weight_val, 1, start, stop
            )
        except Exception:
            pass
    clone = graph.call_function(
        torch.ops.aten.clone.default,
        args=(slice_node,),
        kwargs={"memory_format": torch.contiguous_format},
    )
    if slice_node.meta.get("val") is not None:
        try:
            clone.meta["val"] = slice_node.meta["val"].contiguous()
        except Exception:
            pass
    return clone


def fuse_cat_linear_in_graph(graph: fx.Graph) -> int:
    """Run the matcher across `graph`, returning the number of rewrites."""
    n = 0
    linear_target = torch.nn.functional.linear
    add_target = operator.add

    candidates = [node for node in list(graph.nodes) if _is_linear(node)]
    for ln in candidates:
        match = _try_match_cat_linear(ln)
        if match is None:
            continue
        parts, weight, bias, K_off = match
        weight_val = _val_of(weight)
        ln_val = _val_of(ln)

        with graph.inserting_before(ln):
            partial_outs = []
            for i in range(len(parts)):
                w_slc = _slice_then_contiguous(
                    graph, weight, K_off[i], K_off[i + 1], weight_val
                )
                # Bias only on the first F.linear; mathematically equivalent
                # and saves a standalone tensor-add at the end.
                out_i = graph.call_function(
                    linear_target,
                    args=(parts[i], w_slc, bias if i == 0 else None),
                )
                if ln_val is not None:
                    out_i.meta["val"] = ln_val
                partial_outs.append(out_i)
            acc = partial_outs[0]
            for o in partial_outs[1:]:
                acc = graph.call_function(add_target, args=(acc, o))
                if ln_val is not None:
                    acc.meta["val"] = ln_val

        ln.replace_all_uses_with(acc)
        graph.erase_node(ln)
        # The cat is now dead; DCE will clean it up.

        counters["inductor"]["cat_linear_fused"] += 1
        log.debug(
            "cat_linear_fused: rewrote linear(cat n=%d, K_total=%d, has_bias=%s)",
            len(parts),
            K_off[-1],
            bias is not None,
        )
        n += 1
    return n


def cat_linear_fused_pre_grad_pass(graph: fx.Graph):
    n = fuse_cat_linear_in_graph(graph)
    if n > 0:
        log.info("cat_linear_fused: %d replacement(s)", n)
    return graph


__all__ = [
    "fuse_cat_linear_in_graph",
    "cat_linear_fused_pre_grad_pass",
]
