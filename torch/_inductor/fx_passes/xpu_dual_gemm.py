# mypy: allow-untyped-defs
"""
XPU Dual GEMM fusion pass for Inductor.

Recognizes the pattern:
    out = silu(x @ w1.T) * (x @ w3.T)

And replaces it with a single fused kernel call:
    out = torch.ops.xpu_ops.dual_gemm_silu_mul(x, w1, w3)

This saves one full read of x from HBM (the two GEMM operations share the A
matrix), and merges three separate kernel launches into one.

The pass operates on the post-grad aten IR, before Inductor lowering.  It
matches the silu decomposition pattern that Inductor emits for bfloat16/fp16:
    x / (exp(-x) + 1)   (with prims.convert_element_type casts around it)
"""

import logging

import torch
import torch.fx as fx
from torch._dynamo.utils import counters

log = logging.getLogger(__name__)

aten = torch.ops.aten
prims = torch.ops.prims


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_single_user(node: fx.Node):
    """Return the unique user of node, or None if it has 0 or 2+ users."""
    users = list(node.users)
    return users[0] if len(users) == 1 else None


def _match_permute_t(node: fx.Node):
    """
    Return the input tensor if node is aten.permute(tensor, [1, 0]), else None.
    This matches the weight.T pattern for 2-D weights.
    """
    if node.target is not aten.permute.default:
        return None
    perm = node.args[1] if len(node.args) > 1 else node.kwargs.get("dims")
    if perm != [1, 0]:
        return None
    return node.args[0]


def _match_silu_of_mm(mm_node: fx.Node):
    """
    Try to match the inductor silu decomposition pattern starting from an
    aten.mm node:

        mm_node   = aten.mm(x, w_T)
        cvt_fp32  = prims.convert_element_type(mm_node, float32)   [optional]
        neg       = aten.neg(cvt_fp32 | mm_node)
        exp_neg   = aten.exp(neg)
        add1      = aten.add(exp_neg, 1)
        div_silu  = aten.div(cvt_fp32 | mm_node, add1)
        cvt_back  = prims.convert_element_type(div_silu, orig_dtype) [optional]

    Returns the final silu output node, or None if the pattern does not match.
    The mm_node is allowed to have 2 users (one for cvt_fp32 path and one
    direct path), but the final div / cvt_back must have exactly 1 user (the
    mul).
    """
    # Step 0: optional upcast to float32 for numerics
    silu_input = mm_node
    cvt_up = None
    mm_users = list(mm_node.users)
    for u in mm_users:
        if (
            u.target is prims.convert_element_type.default
            and u.args[1] == torch.float32
        ):
            cvt_up = u
            silu_input = u
            break

    if silu_input is None:
        return None

    # Step 1: neg
    silu_in_users = list(silu_input.users)
    neg_node = None
    for u in silu_in_users:
        if u.target is aten.neg.default and u.args[0] is silu_input:
            neg_node = u
            break
    if neg_node is None:
        return None
    if _get_single_user(neg_node) is None:
        return None

    # Step 2: exp
    exp_node = _get_single_user(neg_node)
    if exp_node is None or exp_node.target is not aten.exp.default:
        return None

    # Step 3: add(exp, 1)
    add_node = _get_single_user(exp_node)
    if add_node is None or add_node.target is not aten.add.Tensor:
        return None
    add_args = add_node.args
    if not (
        (add_args[0] is exp_node and add_args[1] == 1)
        or (add_args[1] is exp_node and add_args[0] == 1)
    ):
        return None

    # Step 4: div(silu_input, add)
    add_user = _get_single_user(add_node)
    if add_user is None or add_user.target is not aten.div.Tensor:
        return None
    div_args = add_user.args
    if div_args[0] is not silu_input or div_args[1] is not add_node:
        return None
    div_node = add_user

    # Step 5: optional downcast back to original dtype
    final_node = div_node
    div_users = list(div_node.users)
    if len(div_users) == 1:
        u = div_users[0]
        if u.target is prims.convert_element_type.default and u.args[1] in (
            torch.bfloat16,
            torch.float16,
        ):
            final_node = u

    return final_node


def _is_xpu_bf16_or_fp16(node: fx.Node):
    """Return True if node produces a bf16 or fp16 tensor on XPU."""
    val = node.meta.get("val")
    if val is None:
        return False
    return val.device.type == "xpu" and val.dtype in (
        torch.bfloat16,
        torch.float16,
    )


# ---------------------------------------------------------------------------
# Main pass
# ---------------------------------------------------------------------------

def xpu_dual_gemm_pass(graph: fx.Graph) -> None:
    """
    Scan the graph for the pattern:
        mm1  = aten.mm(x, permute(w1, [1,0]))
        silu1 = silu_decomposed(mm1)
        mm3  = aten.mm(x, permute(w3, [1,0]))
        out  = aten.mul(silu1, mm3)

    and replace it with:
        out  = torch.ops.xpu_ops.dual_gemm_silu_mul(x, w1, w3)

    Only applied when x, w1, w3 are bfloat16/fp16 on XPU and the kernel is
    available (compiled with USE_SYCLTLA).

    Note: mm3 may have additional users (e.g. saved tensors for the backward
    pass in training mode).  In that case the fused node replaces only the
    ``mul`` output; the separate ``mm3`` node is left in the graph so that
    those users remain valid.  DCE will clean up other dead nodes.
    """
    import torch.ops  # ensure ops are loaded

    # Check availability
    if not hasattr(torch.ops, "xpu_ops"):
        return
    if not hasattr(torch.ops.xpu_ops, "dual_gemm_silu_mul"):
        return

    fused_op = torch.ops.xpu_ops.dual_gemm_silu_mul.default

    # Collect all aten.mul nodes – these are the roots of our pattern search
    mul_nodes = list(graph.find_nodes(op="call_function", target=aten.mul.Tensor))

    for mul_node in mul_nodes:
        if mul_node._erased:
            continue
        # mul(silu_out, mm3)
        silu_out, rhs = mul_node.args[0], mul_node.args[1]

        # Check that rhs is an aten.mm node
        if not (
            isinstance(rhs, fx.Node)
            and rhs.target is aten.mm.default
        ):
            # try commuted: mul(mm3, silu_out)
            silu_out, rhs = rhs, silu_out
            if not (
                isinstance(rhs, fx.Node)
                and rhs.target is aten.mm.default
            ):
                continue

        mm3_node = rhs

        # mm3 = aten.mm(x, permute(w3, [1,0]))
        if len(mm3_node.args) < 2:
            continue
        x_from_mm3, w3_t_node = mm3_node.args[0], mm3_node.args[1]
        if not isinstance(w3_t_node, fx.Node):
            continue
        w3 = _match_permute_t(w3_t_node)
        if w3 is None:
            continue

        # silu_out traces back to an aten.mm node through the silu decomposition
        # We need to find which mm node silu was applied to.
        # Walk up through optional cvt_back -> div -> ... -> mm1

        # Peel off optional downcast
        silu_top = silu_out
        if (
            isinstance(silu_top, fx.Node)
            and silu_top.target is prims.convert_element_type.default
        ):
            silu_top = silu_top.args[0]  # div node

        # Now silu_top should be the div node (or the silu_out directly if no cvt)
        if not (isinstance(silu_top, fx.Node) and silu_top.target is aten.div.Tensor):
            continue
        div_node = silu_top
        # div(silu_input, add)
        silu_input = div_node.args[0]
        add_node_cand = div_node.args[1]
        if not (isinstance(add_node_cand, fx.Node) and add_node_cand.target is aten.add.Tensor):
            continue

        # Peel off optional upcast
        mm1_node = silu_input
        if (
            isinstance(mm1_node, fx.Node)
            and mm1_node.target is prims.convert_element_type.default
        ):
            mm1_node = mm1_node.args[0]  # actual mm node

        if not (isinstance(mm1_node, fx.Node) and mm1_node.target is aten.mm.default):
            continue

        # mm1 = aten.mm(x, permute(w1, [1,0]))
        if len(mm1_node.args) < 2:
            continue
        x_from_mm1, w1_t_node = mm1_node.args[0], mm1_node.args[1]
        if not isinstance(w1_t_node, fx.Node):
            continue
        w1 = _match_permute_t(w1_t_node)
        if w1 is None:
            continue

        # x must be the same node in both mm calls
        if x_from_mm1 is not x_from_mm3:
            continue

        x = x_from_mm1

        # Dtype/device checks
        if not _is_xpu_bf16_or_fp16(mm1_node):
            continue
        if not _is_xpu_bf16_or_fp16(mm3_node):
            continue
        # Also check that w1 and w3 are the same dtype (they should be constants)
        w1_val = w1.meta.get("val")
        w3_val = w3.meta.get("val")
        if w1_val is None or w3_val is None:
            continue
        if w1_val.dtype != w3_val.dtype:
            continue
        if mm1_node.meta["val"].dtype != mm3_node.meta["val"].dtype:
            continue

        # Verify the full silu chain is intact (guard against partial matches)
        expected_silu_out = _match_silu_of_mm(mm1_node)
        if expected_silu_out is not silu_out:
            continue

        # All checks passed – perform the replacement
        log.debug(
            "xpu_dual_gemm_pass: fusing mm(%s, %s.T) + silu + mm(%s, %s.T)",
            x.name, w1.name, x.name, w3.name,
        )

        with graph.inserting_before(mul_node):
            fused_node = graph.call_function(fused_op, args=(x, w1, w3))
            fused_node.meta["val"] = mul_node.meta.get("val")

        mul_node.replace_all_uses_with(fused_node)

        # Clean up the now-dead nodes (dead code elimination will handle the rest).
        # Note: mm3_node is intentionally NOT erased here – it may still have
        # other users (e.g. saved tensors for the backward pass), and DCE will
        # remove it only when it becomes truly dead.
        graph.erase_node(mul_node)

        counters["inductor"]["xpu_dual_gemm_silu_mul"] += 1

    graph.eliminate_dead_code()
