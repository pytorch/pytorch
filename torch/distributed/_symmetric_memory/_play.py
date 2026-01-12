from typing import Callable

import torch
import triton
import triton.language as tl

# ==============================================================================
# Part 1: The Reusable Skeleton Component (Framework Level)
# ==============================================================================

# Type alias for prologue function: (ptr, offsets, mask, meta_ptr) -> tensor
PrologueFn = Callable[
    [tl.tensor, tl.tensor, tl.tensor, tl.tensor],
    tl.tensor,
]


@triton.jit
def elementwise_negation_skeleton(
    # Pointers
    in_ptr,
    out_ptr,
    # Grid Geometry
    n_elements,
    # Optional metadata pointer for user functions
    meta_ptr,
    # Compile-Time Constants
    BLOCK_SIZE: tl.constexpr,
    PROLOGUE_FN: "tl.constexpr[PrologueFn]",
    EPILOGUE_FN: tl.constexpr,
):
    """
    A generic skeleton for 1D element-wise negation.
    It handles grid calculation and delegates load/store logic.
    """
    # 1. Grid & Offset Calculation
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # 2. Prologue: Ingest Data
    # The skeleton passes the pointers and the geometric context (offsets, mask)
    # to the user-provided prologue.
    x = PROLOGUE_FN(in_ptr, offsets, mask, meta_ptr)

    # 3. Main Algorithm: Negation
    # This is the fixed behavior of this specific skeleton.
    result = -x

    # 4. Epilogue: Persist Data
    # The skeleton passes the result and context to the user-provided epilogue.
    EPILOGUE_FN(out_ptr, offsets, mask, result, meta_ptr)


# ==============================================================================
# Part 2: User-Defined Components (Application Level)
# ==============================================================================

# --- Prologue Examples ---


@triton.jit
def standard_load(ptr, offsets, mask, meta_ptr):
    """Simply loads contiguous data."""
    return tl.load(ptr + offsets, mask=mask)


@triton.jit
def cast_load(ptr, offsets, mask, meta_ptr):
    """Loads and casts to float32 (useful if input is FP16 but math needs FP32)."""
    val = tl.load(ptr + offsets, mask=mask)
    return val.to(tl.float32)


# --- Epilogue Examples ---


@triton.jit
def standard_store(ptr, offsets, mask, val, meta_ptr):
    """Simply stores contiguous data."""
    tl.store(ptr + offsets, val, mask=mask)


@triton.jit
def relu_store(ptr, offsets, mask, val, meta_ptr):
    """Fused ReLU: Stores max(0, val)."""
    zero = tl.zeros(val.shape, dtype=val.dtype)
    activated = tl.maximum(zero, val)
    tl.store(ptr + offsets, activated, mask=mask)


@triton.jit
def scaled_store(ptr, offsets, mask, val, meta_ptr):
    """Fused Scaling: Stores val * scale. Requires 'scale' loaded from meta_ptr."""
    scale = tl.load(meta_ptr)  # Retrieve dynamic scalar from meta pointer
    tl.store(ptr + offsets, val * scale, mask=mask)


# ==============================================================================
# Part 3: Host Launcher (Glue Code)
# ==============================================================================


def run_custom_negation(x: torch.Tensor, prologue, epilogue, meta_ptr=None):
    """
    Dispatches the skeleton kernel with the chosen prologue/epilogue.
    """
    # Allocate output
    out = torch.empty_like(x)

    # Grid setup
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    # Launch
    elementwise_negation_skeleton[grid](
        in_ptr=x,
        out_ptr=out,
        n_elements=n_elements,
        meta_ptr=meta_ptr,
        BLOCK_SIZE=BLOCK_SIZE,
        PROLOGUE_FN=prologue,
        EPILOGUE_FN=epilogue,
    )

    return out


# ==============================================================================
# Part 4: Verification
# ==============================================================================


def verify():
    print("Verifying customizable kernel...")

    # 1. Setup Data
    device = "cuda"
    x = torch.randn(4096, device=device)

    # 2. Test Case A: Standard Negation
    # Expected: y = -x
    y_custom = run_custom_negation(x, standard_load, standard_store)
    y_ref = -x
    assert torch.allclose(y_custom, y_ref), "Standard negation failed!"
    print("  [Pass] Standard Negation")

    # 3. Test Case B: Negation + ReLU
    # Expected: y = relu(-x)
    y_custom_relu = run_custom_negation(x, standard_load, relu_store)
    y_ref_relu = torch.relu(-x)
    assert torch.allclose(y_custom_relu, y_ref_relu), "Fused ReLU failed!"
    print("  [Pass] Fused ReLU")

    # 4. Test Case C: Negation + Scaling
    # Expected: y = -x * 2.0
    scale = torch.tensor(2.0, device=device)
    y_custom_scale = run_custom_negation(x, standard_load, scaled_store, meta_ptr=scale)
    y_ref_scale = -x * scale
    assert torch.allclose(y_custom_scale, y_ref_scale), "Fused Scaling failed!"
    print("  [Pass] Fused Scaling")


if __name__ == "__main__":
    verify()
