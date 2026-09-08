# Shared launch-plan memoization. A plan is the compiled kernel plus every shape-invariant host
# decision, derived once per operand signature -- without it the eager host cost is dominated
# by re-deriving the plan rather than by the launch. A plan of None is a valid memoized result
# meaning "declined", cached so a non-trivial decline is not recomputed.

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Callable


def cached_plan(cache: dict, key, build: Callable, *, op: str | None = None):
    """Return ``build()``'s plan for ``key``, memoized in ``cache``; None means declined.

    ``op`` wraps the build -- which fires only on a MISS, i.e. exactly when a real compile
    happens -- in the TLParse instrumentation. This is the chokepoint every CuteDSL native op
    shares, so instrumenting here covers the family; None gives the plain memo.
    """
    plan = cache.get(key, _MISS)
    if plan is _MISS:
        if op is not None:
            from torch._native.instrumentation import instrument_cutedsl_compile

            # compiled=True, not inferred: `build` is a plain closure with no cache_info and runs only on
            # the MISS arm, so the miss-delta inference would call every real compile a cache hit.
            plan = instrument_cutedsl_compile(
                op, key_fn=lambda: str(key), compiled=True
            )(build)()
        else:
            plan = build()
        cache[key] = plan
    return plan


_MISS = object()  # sentinel so a cached None (declined) is distinct from a miss
