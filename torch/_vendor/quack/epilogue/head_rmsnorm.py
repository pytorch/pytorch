# Copyright (c) 2026, Tri Dao.
"""Per-head rstd statistics for composed GEMM epilogues."""

from __future__ import annotations

import cutlass.cute as cute
from cutlass import Float32, const_expr

from torch._vendor.quack.epilogue.ops import GroupedColStatsBase


class HeadRstd(GroupedColStatsBase):
    """Per-(row, head) rstd value port — pure statistics, no weight.

    The prepass fn returns the SQUARED values under this op's name, the base
    add-combine fold sums them per (row, head) (deterministic, no float
    atomics), and ``stat_value`` finalizes rsqrt(mean + eps). Apply it in the
    epilogue fn: ``acc * rstd``, or ``acc * rstd * w`` with the norm weight
    as an independent RowVecLoad (pass it (N,)-shaped, i.e. the head weight
    repeated per head). Host arg: head_dim as a plain int, or any 1-D
    (head_dim,) tensor — only its LENGTH is used (it fixes the group
    width)."""

    def __init__(self, name, eps=1e-6):
        super().__init__(name)
        self.eps = eps

    def config_key(self):
        return (self.eps,)

    @cute.jit
    def stat_value(self, total, group_cols):
        inv_d = const_expr(1.0 / group_cols)
        return cute.math.rsqrt(total * inv_d + Float32(self.eps), fastmath=True)
