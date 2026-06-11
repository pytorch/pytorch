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

Shape guard:
    The fused DualGemm kernel uses a fixed 128×128×64 tile on Intel XPU
    (sycl-tla).  This is efficient when M is large relative to the tile size,
    but wastes compute when M is small (LLaMA inference with batch=1..128).
    For small M, oneDNN (eager) is significantly faster — benchmarks show the
    fused kernel at only 0.17x–0.45x of eager speed on LLaMA shapes
    (K=4096, N=14336) for M=1..1024.

    The fusion is therefore gated by:
      - TORCH_XPU_DUAL_GEMM_MIN_M: minimum M dimension (default: 512)
      - N/M ratio: when N >> M the tile efficiency drops further
      - Minimum K requirement (≥64) for XMX efficiency

    For shapes below threshold, the pass leaves the pattern unmatched so
    Inductor falls back to oneDNN (which is faster for these cases).
"""

import logging
import os

import torch
import torch.fx as fx
from torch._dynamo.utils import counters

log = logging.getLogger(__name__)

aten = torch.ops.aten
prims = torch.ops.prims

# Minimum M dimension to trigger fusion (tunable via env var).
# Default 512: benchmarks show fused kernel only starts approaching oneDNN
# at very large M for LLaMA-style shapes.  Set to 1 to force fusion always.
_MIN_M_FOR_FUSION = int(os.environ.get("TORCH_XPU_DUAL_GEMM_MIN_M", "512"))


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


def _should_fuse(x_val, w1_val):
    """
    Shape guard: decide whether the fused kernel will outperform eager.

    The fused DualGemm kernel uses a fixed 128×128×64 tile on Intel XPU.
    Benchmarks on PVC (Max 1550) show that for LLaMA-style shapes
    (K=4096, N=14336) the fused kernel is ~3-5x SLOWER than eager (oneDNN)
    for all tested M values (1..1024).

    Root cause: the 128×128 M-tile wastes compute when M < 128 (padding),
    and for N=14336 the grid has 112 N-tiles per M-tile row.  The overhead
    of launching that many tiles with padding vastly exceeds the HBM savings
    from the shared-A optimization.

    Heuristic:
      1. M must be >= _MIN_M_FOR_FUSION (default 512)
      2. K must be >= 64 (XMX efficiency minimum)
      3. The "tile utilization" (M / 128) * (N / 128) should indicate enough
         useful work per tile.  When M is small relative to the tile, the
         padding cost dominates.
      4. Compute/memory ratio: the fused kernel saves one x read (M*K*2 bytes)
         but adds tile padding overhead of (128 - M%128) * N * 2 GEMMs.
         Fusion is only worthwhile when the savings exceed the waste.

    Returns True if fusion should be applied.
    """
    if x_val is None or w1_val is None:
        return False  # can't determine shape at compile time — be conservative

    # For symbolic shapes, we can't evaluate at compile time
    M = x_val.shape[0]
    K = x_val.shape[1]
    N = w1_val.shape[0]  # w1 is [N, K]

    # Check if shapes are concrete (not symbolic)
    try:
        M_val = int(M)
        N_val = int(N)
        K_val = int(K)
    except (TypeError, ValueError):
        # Symbolic shapes — be conservative, skip fusion
        log.debug("xpu_dual_gemm_pass: skipping fusion for symbolic shapes")
        return False

    # K must be reasonable for XMX efficiency (at least 64 for the K-tile)
    if K_val < 64:
        log.debug(
            "xpu_dual_gemm_pass: skipping fusion for small K=%d", K_val
        )
        return False

    # Minimum M guard (configurable via TORCH_XPU_DUAL_GEMM_MIN_M)
    if M_val < _MIN_M_FOR_FUSION:
        log.debug(
            "xpu_dual_gemm_pass: skipping fusion for M=%d < MIN_M=%d",
            M_val,
            _MIN_M_FOR_FUSION,
        )
        return False

    # Tile utilization check: the kernel pads M to next multiple of 128.
    # If the effective utilization is too low, the wasted FLOPs dominate.
    tile_m = 128
    m_padded = ((M_val + tile_m - 1) // tile_m) * tile_m
    utilization = M_val / m_padded
    if utilization < 0.5:
        log.debug(
            "xpu_dual_gemm_pass: skipping fusion for M=%d, util=%.1f%% "
            "(padded to %d)",
            M_val, utilization * 100, m_padded,
        )
        return False

    # Cost model: fusion saves one read of x (M*K*elem_bytes) but the
    # fused kernel has higher per-tile overhead.  For very tall-thin shapes
    # (large N, small M) the grid is huge and the overhead dominates.
    # Empirically, fused kernel is only competitive when M*N >= threshold.
    # Based on benchmarks on PVC Max 1550:
    #   - 1024x1024x1024: fused 1.51x faster → M*N = 1M ✓
    #   - 2048x2048x2048: fused 0.72x slower → M*N = 4M ✗
    #   - All LLaMA shapes (N=14336): fused always slower
    # The key insight: fused only wins when M ≈ N (square-ish) AND M*N ≥ 1M.
    # For tall-thin shapes (N >> M), oneDNN always wins.
    total_output_elems = M_val * N_val
    # Require at least ~1M output elements for the grid to be meaningful
    min_output_elems = int(os.environ.get("TORCH_XPU_DUAL_GEMM_MIN_OUTPUT", "1048576"))  # 1M
    if total_output_elems < min_output_elems:
        log.debug(
            "xpu_dual_gemm_pass: skipping fusion for M=%d N=%d, "
            "output_elems=%d < %d",
            M_val, N_val, total_output_elems, min_output_elems,
        )
        return False

    # Aspect ratio check: the fused kernel's 128×128 tile is most efficient
    # when M and N are comparable.  For tall-thin shapes (N >> M), the grid
    # has many N-tiles per M-tile row, and each one does padded computation.
    # Benchmarks show fused kernel loses to oneDNN for all N=14336 shapes.
    # Also, for very large square shapes (2048+), oneDNN auto-tuning wins.
    #
    # Empirical sweet spot on PVC Max 1550:
    #   - 1024×1024×1024: fused 1.51x faster
    #   - 2048×2048×2048: fused 0.72x slower
    #   - All N=14336 shapes: fused always slower
    #
    # Conservative max N/M ratio of 4.0 and output cap of 2M elements.
    aspect_ratio = max(N_val / max(M_val, 1), M_val / max(N_val, 1))
    max_aspect = float(os.environ.get("TORCH_XPU_DUAL_GEMM_MAX_ASPECT", "4.0"))
    if aspect_ratio > max_aspect:
        log.debug(
            "xpu_dual_gemm_pass: skipping fusion for M=%d N=%d, "
            "aspect_ratio=%.1f > %.1f",
            M_val, N_val, aspect_ratio, max_aspect,
        )
        return False

    # Cap on total output: beyond ~2M elements, oneDNN's parallelism wins.
    max_output_elems = int(os.environ.get("TORCH_XPU_DUAL_GEMM_MAX_OUTPUT", "2097152"))  # 2M
    if total_output_elems > max_output_elems:
        log.debug(
            "xpu_dual_gemm_pass: skipping fusion for M=%d N=%d, "
            "output_elems=%d > max %d",
            M_val, N_val, total_output_elems, max_output_elems,
        )
        return False

    log.debug(
        "xpu_dual_gemm_pass: fusion enabled for M=%d K=%d N=%d "
        "(util=%.0f%%, output_elems=%d, aspect=%.1f)",
        M_val, K_val, N_val, utilization * 100, total_output_elems, aspect_ratio,
    )
    return True


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

    Shape guard: The pass checks M, K, N dimensions to ensure the fused kernel
    is beneficial. The fused kernel has tile variants optimized for different M
    ranges. Set TORCH_XPU_DUAL_GEMM_MIN_M env var to control minimum M.

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

        # Shape guard: check if fusion is beneficial for the given dimensions
        x_val = x.meta.get("val")
        if not _should_fuse(x_val, w1_val):
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
