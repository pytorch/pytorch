# Shared launch-plan memoization for the CuteDSL native ops.
#
# Every op family does the same thing: derive a "plan" (the compiled kernel + all
# shape-invariant host-side decisions -- chosen path, vector width, output dtypes,
# alignment, ...) once per distinct operand signature, then on later calls reuse it
# and only do the per-call work (wrap the live tensors, launch). The plan is a pure
# function of the cache KEY (dtypes, shapes/strides, device, op params), which is
# exactly what is baked into the compiled kernel.
#
# Without this, the eager host cost is dominated by recomputing the plan every call
# (shape/promotion derivation, path selection, kernel lookup) -- far more than the
# irreducible wrap + launch. Caching collapses a repeated-shape workload to wrap +
# launch.
#
# A plan of None is a valid memoized result meaning "declined -- the caller should
# fall back" (e.g. a geometry this kernel cannot serve). It is cached like any other
# so the (sometimes non-trivial) decline decision is not recomputed; callers use the
# returned (hit, plan) pair to tell a cached-None from a miss.

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Callable


def cached_plan(cache: dict, key, build: Callable):
    """Return ``build()``'s plan for ``key``, memoized in ``cache`` (including a
    None plan, which means "declined / fall back"). ``build`` is called at most once
    per key. Returns the plan (possibly None)."""
    plan = cache.get(key, _MISS)
    if plan is _MISS:
        plan = build()
        cache[key] = plan
    return plan


_MISS = object()  # sentinel so a cached None (declined) is distinct from a miss
