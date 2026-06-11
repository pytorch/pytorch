import torch
import triton
import triton.language as tl

@triton.jit
def triton_fused_rmsnorm_residual_gating_0(
    in_ptr0,        # Input x
    in_ptr1,        # Residual
    in_ptr2,        # RMSNorm Weights
    out_ptr0,       # Final Gated Output
    stride_xnumel,  # Stride between rows (Hidden Dim)
    rnumel,         # Total Hidden Dim (e.g., 8192)
    eps,            # RMSNorm epsilon
    RBLOCK: tl.constexpr,
):
    """
    Fused kernel that eliminates HBM roundtrips.

    Inductor's 2-kernel approach:
    - Kernel 0: Load x+res, compute variance, WRITE variance to HBM, discard x+res
    - Kernel 1: LOAD x+res again, load variance from HBM, compute output

    Our fused approach:
    - Load x+res for variance computation
    - Compute variance (kept in registers, NOT written to HBM)
    - Load x+res for final computation, using variance from registers
    - No intermediate HBM writes!

    The speedup comes from:
    1. No HBM write/read for variance
    2. Shared work for variance computation
    """
    # row_idx: which row we're processing
    xindex = tl.program_id(0)

    # row_start: starting offset for this row
    row_start = xindex * stride_xnumel

    # half_rnumel: output dimension
    half_rnumel = rnumel // 2

    # --- STEP 1: Compute variance (requires full row) ---
    # This replaces Inductor's triton_red_fused_add_mean_pow_0

    # rindex: indices for reduction over rnumel
    rindex = tl.arange(0, RBLOCK)
    rmask = rindex < rnumel

    # Load inputs for variance computation
    tmp0 = tl.load(in_ptr0 + row_start + rindex, mask=rmask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + row_start + rindex, mask=rmask, other=0.0).to(tl.float32)

    # added_for_var: x + residual
    tmp2 = tmp0 + tmp1

    # Square for variance computation
    tmp3 = tmp2 * tmp2

    # Compute variance (reduction) - kept in registers, NOT written to HBM!
    tmp4 = tl.sum(tmp3, axis=0) / rnumel

    # Add eps and rsqrt (Inductor-style explicit tensor constant)
    tmp5 = tl.full([1], eps, tl.float32)
    tmp6 = tmp4 + tmp5
    tmp7 = tl.math.rsqrt(tmp6)

    # --- STEP 2: Compute output (replaces Inductor's Kernel 1) ---
    # For each output position i (0 to half_rnumel-1):
    #   gate = normed(x[i] + res[i])
    #   up = normed(x[i+half] + res[i+half])
    #   output[i] = silu(gate) * up

    # out_index: indices for output (only half the elements)
    # Use RBLOCK // 2 to avoid wasted threads
    out_index = tl.arange(0, RBLOCK // 2)

    # Load gate part (first half: 0 to half_rnumel-1)
    tmp8 = tl.load(in_ptr0 + row_start + out_index).to(tl.float32)
    tmp9 = tl.load(in_ptr1 + row_start + out_index).to(tl.float32)
    tmp10 = tl.load(in_ptr2 + out_index).to(tl.float32)

    # gate: (x_gate + res_gate) * rsqrt_var * w_gate
    tmp11 = (tmp8 + tmp9) * tmp7 * tmp10

    # Load up part (second half: half_rnumel to rnumel-1)
    up_index = out_index + half_rnumel
    tmp12 = tl.load(in_ptr0 + row_start + up_index).to(tl.float32)
    tmp13 = tl.load(in_ptr1 + row_start + up_index).to(tl.float32)
    tmp14 = tl.load(in_ptr2 + up_index).to(tl.float32)

    # up: (x_up + res_up) * rsqrt_var * w_up
    tmp15 = (tmp12 + tmp13) * tmp7 * tmp14

    # Apply SiLU gating: silu(gate) * up where silu(x) = x * sigmoid(x)
    tmp16 = 1.0 / (1.0 + tl.exp(-tmp11))
    tmp17 = tmp11 * tmp16 * tmp15

    # Store output
    output_offset = xindex * half_rnumel
    tl.store(out_ptr0 + output_offset + out_index, tmp17)


def launch_fused_block(arg0_1, arg1_1, arg2_1):
    """
    Launch the fused kernel.

    Args:
        arg0_1: Input tensor x (M, N)
        arg1_1: Residual tensor (M, N)
        arg2_1: Weight tensor (N,)

    Returns:
        Output tensor (M, N//2)
    """
    # M: number of rows, N: hidden dim
    xnumel, rnumel = arg0_1.shape

    # Allocate output buffer
    buf0 = torch.empty((xnumel, rnumel // 2), device=arg0_1.device, dtype=arg0_1.dtype)

    # RBLOCK must be >= rnumel to handle full row for variance
    RBLOCK = triton.next_power_of_2(rnumel)

    # grid: one block per row
    grid = (xnumel,)

    # Launch kernel
    triton_fused_rmsnorm_residual_gating_0[grid](
        arg0_1, arg1_1, arg2_1, buf0,
        arg0_1.stride(0), rnumel, 1e-6,
        RBLOCK=RBLOCK,
    )
    return buf0
