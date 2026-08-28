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
# so the (sometimes non-trivial) decline decision is not recomputed. Callers get the
# plan itself back, None included; the miss-vs-cached-None distinction is internal
# (the _MISS sentinel) and deliberately not part of the signature.

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Callable


def cached_plan(cache: dict, key, build: Callable, *, op: str | None = None):
    """Return ``build()``'s plan for ``key``, memoized in ``cache`` (including a
    None plan, which means "declined / fall back"). ``build`` is called at most once
    per key. Returns the plan (possibly None).

    ``op`` (e.g. ``"aten::add"``): when set, the ``build()`` call -- which fires only
    on a cache MISS, i.e. exactly when a real CuTeDSL compile happens -- is wrapped with
    the TLParse instrumentation (torch._native.instrumentation), so each native-op
    compile emits one structured log line + tlparse artifact (timing, cache key). This
    is the memoization chokepoint every CuteDSL native op shares, so instrumenting here
    covers the whole family. instrumentation is imported lazily (it pulls in the tlparse
    logging path) and the wrap only happens on a miss, so cache hits keep the hot path
    free. None -> no instrumentation (the plain memo)."""
    plan = cache.get(key, _MISS)
    if plan is _MISS:
        if op is not None:
            from torch._native.instrumentation import instrument_cutedsl_compile

            plan = instrument_cutedsl_compile(op, key_fn=lambda: str(key))(build)()
        else:
            plan = build()
        cache[key] = plan
    return plan


_MISS = object()  # sentinel so a cached None (declined) is distinct from a miss
