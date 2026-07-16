"""Shared strict-numerics reduction config — the SINGLE source of truth for the tile
(R0_BLOCK) and split factor, imported by BOTH eager (torch/_strict_reduction.py) and
Inductor (triton_heuristics / choices). Because both sides read the same functions, they
pick identical tiles/splits by construction, so a strict reduction is bitwise-identical
between eager and torch.compile. Pure Python (no cutlass / no torch import cost)."""

import functools
import math


STRICT_MAX_RBLOCK = 1024  # Blackwell MAX_R0_BLOCK
_C_MAX = 1 << 22
_SUBROW_TARGET = 8192
_SMEM_BUDGET = 192 * 1024


def _prev_pow2(n: int) -> int:
    return 1 << (n.bit_length() - 1) if n >= 1 else 1


def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length() if n > 1 else 1


def strict_rblock(n: int) -> int:
    """R0_BLOCK both sides use. For N <= 1024 -> next power of two (a PERSISTENT pad+tree,
    matching Inductor's persistent RBLOCK=next_pow2(N)); above -> 1024 (a LOOPED stride-fold
    tile). Deterministic + dtype-independent, so eager and Inductor tile identically and the
    reduction order is bitwise-identical for every N (pow2 and non-pow2 alike)."""
    n = int(n)
    if n <= 0:
        return 1
    if n <= STRICT_MAX_RBLOCK:
        return _next_pow2(n)
    return STRICT_MAX_RBLOCK


def _split_C(N, vec, smem_budget_elems, target=_SUBROW_TARGET):
    step = max(vec, 1)
    lo = max(step, 256)
    hi = min(smem_budget_elems, N)
    hi -= hi % step
    if hi < lo:
        return None
    tgt = min(max(target - (target % step), lo), hi)
    for d in range(0, hi - lo + step, step):
        for s in (tgt + d, tgt - d):
            if lo <= s <= hi and N % s == 0 and N // s <= _C_MAX:
                return N // s
    return None


@functools.cache
def strict_split_factor(N, num_outputs, num_sm, dtype_bytes=4):
    """Split factor C both sides use (contiguous chunks): 1 = no split. C > 1 splits the
    reduction axis into C chunks of N/C (two-stage). Deterministic given the inputs."""
    if int(num_outputs) >= 2 * int(num_sm):
        return 1
    vec = math.gcd(int(N), 128 // (dtype_bytes * 8))
    c = _split_C(int(N), vec, _SMEM_BUDGET // dtype_bytes)
    return c if c else 1
