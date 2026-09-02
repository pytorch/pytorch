# Shared launch-plan memoization for the CuteDSL native ops. A "plan" is the compiled kernel plus
# every shape-invariant host decision (path, vector width, output dtypes, alignment), derived once per
# operand signature so a repeat call does only the wrap and launch -- without it the eager host cost
# is dominated by re-deriving the plan, not by the launch.
#
# A plan of None is a valid memoized result meaning "declined, fall back", cached so a non-trivial
# decline is not recomputed. The miss-vs-cached-None distinction stays internal (_MISS).

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
