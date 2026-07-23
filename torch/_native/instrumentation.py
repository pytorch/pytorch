"""Structured compile/cache instrumentation for ``torch._native`` ops.

Native DSL ops compile device kernels lazily on first call. Those compiles
are the dominant first-call latency, and a silent cache miss is a common
"why is this slow again?" question. This module surfaces both, with no
runtime cost when neither ``TORCH_LOGS`` nor structured tracing is enabled.

Two sinks, both fed by a single :class:`CompileEvent`:

* The ``native_dsl`` logger (``TORCH_LOGS=+native_dsl``): a one-line
  human-readable summary per compile -- outcome, wall time, and running
  hit/miss totals.
* ``trace_structured`` artifacts (tlparse): a JSON record per compile, for
  production jobs where only the structured trace is retrievable.

Two DSLs compile through different machinery, so there are two entry points.
Both reduce to the same shared core (:func:`_make_wrapper`): snapshot the
cache, time the call, snapshot again, and flag ``compiled`` when the miss
counter advanced. They differ only in how a snapshot is sampled:

* :func:`instrument_cutedsl_compile` -- for CuTeDSL, stacked *above* the
  vendored ``quack`` ``@jit_cache`` decorator. It reads the cache wrapper's
  ``cache_info()`` and times the wrapped ``cute.compile`` call::

      @instrument_cutedsl_compile("aten::topk")
      @jit_cache
      def _compile_topk_radix(N, K, deterministic): ...

  A ``cache_info().misses`` delta means the cache ran a real
  ``cute.compile``, so the measured wall time *is* the compile time;
  otherwise the key was served from the in-memory or on-disk ``.o`` cache.

* :func:`instrument_triton_kernel` -- for Triton ``@triton.jit`` kernels,
  which compile *and* launch in one ``kernel[grid](...)`` call and keep
  their own per-kernel cache (``JITFunction.device_caches``). Stacked above
  ``@triton.jit``, it wraps the kernel in a transparent proxy that watches
  *that one kernel's* variant count; a launch that grows it means a fresh
  compile. Scoping to the single kernel gives clean per-kernel attribution --
  two kernels in one module never collide. Because compile and launch are
  fused, ``wall_ms`` on a miss is compile + host-launch latency (compile
  dominates); on a hit it is just host-launch latency.

Both DSLs only expose miss-side signal directly (CuTeDSL's vendored cache
reports aggregate counters; Triton's cache only grows), so finer reasons
(disk-hit vs lock-timeout, Triton's on-disk cache) are not distinguished
here -- the boolean ``compiled`` flag plus wall time covers the common case.

Neither entry point touches the underlying DSL/vendored code, and neither
hijacks Triton's process-global ``knobs.runtime`` hooks (those would also
capture Inductor's unrelated Triton compiles).
"""

from __future__ import annotations

import functools
import logging
import time
from dataclasses import asdict, dataclass
from typing import Any, TYPE_CHECKING, TypeVar


if TYPE_CHECKING:
    from collections.abc import Callable


__all__ = [
    "CompileEvent",
    "InstrumentedTritonKernel",
    "instrument_cutedsl_compile",
    "instrument_triton_kernel",
    "instrumented_cutedsl_cache",
    "instrumented_triton_cache",
]

log = logging.getLogger(__name__)

# tlparse artifact name. The "artifact" envelope (see trace_structured_artifact)
# is the well-supported transport; the name lets tlparse group these events.
_ARTIFACT_NAME = "native_dsl_compile"

R = TypeVar("R")


@dataclass(frozen=True)
class CompileEvent:
    """One compile-function invocation, as recorded to logs and tlparse.

    ``compiled`` is the ground truth (did the cache run a real compile);
    ``outcome`` is its human-readable form. ``hits`` / ``misses`` are the
    cache's running totals *after* this call, useful for spotting churn
    (e.g. misses climbing across calls of the same shape => keys not
    stable, or the persistent cache is disabled).
    """

    op: str
    dsl: str
    outcome: str
    compiled: bool
    wall_ms: float
    key: str
    hits: int
    misses: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _listening() -> bool:
    """True if either sink would record an event.

    Lets the wrapper skip all instrumentation work (cache sampling, timing,
    key formatting, event construction) on the hot path when nothing is
    listening -- important because the CuTeDSL wrapper sits on the compile
    *cache* and so runs on every op call, cache hits included.

    Mirrors the gates the sinks apply internally: the logger on its effective
    level, and structured tracing on ``trace_log.handlers`` (the documented
    "is tracing enabled" idiom, also used by ``trace_structured`` itself).
    """
    from torch._logging._internal import trace_log

    return log.isEnabledFor(logging.INFO) or bool(trace_log.handlers)


def _emit(event: CompileEvent) -> None:
    """Fan the event out to the native_dsl logger and tlparse.

    Both sinks self-gate (logging on level, trace_structured on
    ``trace_log.handlers``), so this is cheap when nothing is listening.
    """
    log.info(
        "%s [%s] %s in %.1fms (key=%s, cache hits=%d misses=%d)",
        event.op,
        event.dsl,
        event.outcome,
        event.wall_ms,
        event.key,
        event.hits,
        event.misses,
    )

    # Local import keeps `import torch._native` from pulling torch._logging's
    # heavier transitive imports at registration time.
    from torch._logging._internal import trace_structured

    # Same "artifact" envelope as Dynamo's trace_structured_artifact(); we call
    # trace_structured directly only to pass expect_trace_id=False, since that
    # helper would capture an expensive stack on every eager event (no live
    # CompileContext). trace_structured still reads the live trace id, so a
    # native op compiling inside torch.compile is auto-tagged with the ambient
    # frame ids and nests under that compile in tlparse.
    trace_structured(
        "artifact",
        metadata_fn=lambda: {"name": _ARTIFACT_NAME, "encoding": "json"},
        expect_trace_id=False,
        payload_fn=lambda: _json_payload(event),
    )


def _json_payload(event: CompileEvent) -> str:
    import json

    return json.dumps(event.as_dict(), sort_keys=True)


def _format_key(args: tuple, kwargs: dict, key_fn: Callable | None) -> str:
    if key_fn is not None:
        try:
            return key_fn(*args, **kwargs)
        except Exception:
            pass
    parts = [repr(a) for a in args]
    parts += [f"{k}={v!r}" for k, v in sorted(kwargs.items())]
    return "(" + ", ".join(parts) + ")"


def _make_wrapper(
    fn: Callable[..., R],
    op: str,
    dsl: str,
    key_fn: Callable[..., str] | None,
    sample: Callable[[], tuple[int | None, int | None]],
) -> Callable[..., R]:
    """Shared instrumentation core for both DSL entry points.

    ``sample()`` returns a ``(hits, misses)`` snapshot of the relevant cache;
    a ``misses`` increase across the call means a real compile fired. Timing,
    error handling, classification, and emission are identical across DSLs --
    only ``sample`` (and the reported ``dsl``) differ.
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> R:
        # Fast path: do nothing extra unless a sink is listening. This runs on
        # every call to the wrapped fn (for CuTeDSL, every op invocation), so
        # the no-listener cost must stay at one predicate check.
        if not _listening():
            return fn(*args, **kwargs)

        _, misses_before = sample()
        start = time.perf_counter()
        try:
            result = fn(*args, **kwargs)
            outcome_is_error = False
        except Exception:
            outcome_is_error = True
            raise
        finally:
            wall_ms = (time.perf_counter() - start) * 1e3
            hits_after, misses_after = sample()
            compiled = (
                not outcome_is_error
                and misses_before is not None
                and misses_after is not None
                and misses_after > misses_before
            )
            if outcome_is_error:
                outcome = "error"
            else:
                outcome = "compiled" if compiled else "cache_hit"
            _emit(
                CompileEvent(
                    op=op,
                    dsl=dsl,
                    outcome=outcome,
                    compiled=compiled,
                    wall_ms=wall_ms,
                    key=_format_key(args, kwargs, key_fn),
                    hits=hits_after or 0,
                    misses=misses_after or 0,
                )
            )
        return result

    return wrapper


def _cache_info_sampler(fn: Any) -> Callable[[], tuple[int | None, int | None]]:
    """Sampler reading ``(hits, misses)`` from a ``jit_cache`` wrapper.

    Defensive: the wrapped callable may not be ``jit_cache``-decorated (e.g.
    a plain function in tests). Then we report ``(None, None)`` and the core
    still times the call, just without compile/hit classification.
    """

    def sample() -> tuple[int | None, int | None]:
        info = getattr(fn, "cache_info", None)
        if info is None:
            return None, None
        try:
            ci = info()
            return ci.hits, ci.misses
        except Exception:
            return None, None

    return sample


def instrument_cutedsl_compile(
    op: str,
    *,
    key_fn: Callable[..., str] | None = None,
) -> Callable[[Callable[..., R]], Callable[..., R]]:
    """Instrument a CuTeDSL (``@jit_cache``-decorated) compile function.

    Args:
        op: Operator symbol being compiled for, e.g. ``"aten::topk"``.
        key_fn: Optional callable with the wrapped function's signature
            returning a short string describing the compile key for logs.
            Defaults to a repr of the args/kwargs.

    Returns a decorator. The decorated function behaves identically to the
    original (same return value, same caching); it only adds a log line and
    a tlparse artifact per call. Errors raised by the wrapped compile are
    timed, reported with ``outcome="error"``, and re-raised unchanged.
    """

    def decorator(fn: Callable[..., R]) -> Callable[..., R]:
        wrapper = _make_wrapper(fn, op, "cutedsl", key_fn, _cache_info_sampler(fn))
        # Forward jit_cache's bespoke attributes (functools.wraps doesn't copy
        # them) so the instrumented function stays a drop-in for callers that
        # introspect the cache.
        for attr in ("cache", "cache_clear", "cache_info"):
            if hasattr(fn, attr):
                setattr(wrapper, attr, getattr(fn, attr))
        return wrapper

    return decorator


def _triton_cache_size(kernel: Any) -> int | None:
    """Compiled-variant count across one JITFunction's per-device caches.

    Triton stores one entry per specialized variant in
    ``JITFunction.device_caches[device]``, a ``(kernel_cache, ...)`` tuple
    whose first element is the dict of compiled kernels. The count grows by
    one each time a new (signature, constexpr, options) variant compiles, so
    a delta across a launch tells us whether *this* kernel compiled. Returns
    None if the object doesn't expose ``device_caches`` (a future Triton that
    renames this, or a non-kernel in tests).
    """
    caches = getattr(kernel, "device_caches", None)
    if caches is None:
        return None
    try:
        return sum(len(per_device[0]) for per_device in caches.values())
    except Exception:
        return None


class InstrumentedTritonKernel:
    """Transparent proxy over a single ``@triton.jit`` kernel.

    Watches only this kernel's ``device_caches``, so a launch that grows the
    variant count is unambiguously *this* kernel compiling -- two kernels in
    one module never collide. All attribute access except the ``kernel[grid]``
    launch path is delegated to the wrapped kernel, so the proxy is a drop-in
    for the original ``JITFunction`` for attribute/method use (``.warmup``,
    ``.cache_key``).

    Composition with ``torch.library.wrap_triton``: that API does a strict
    ``isinstance(JITFunction)`` check and would reject this proxy, but it
    targets a different path (export/tracing, where the compile does not run
    through our eager launch). When a native op needs ``wrap_triton``, pass the
    raw kernel via :attr:`jit_kernel`: ``wrap_triton(my_kernel.jit_kernel)``.
    """

    def __init__(
        self,
        kernel: Any,
        op: str,
        key_fn: Callable[..., str] | None,
    ) -> None:
        self._kernel = kernel
        self._op = op
        self._key_fn = key_fn

    @property
    def jit_kernel(self) -> Any:
        """The wrapped raw ``JITFunction`` (e.g. for ``wrap_triton``)."""
        return self._kernel

    def _sample(self) -> tuple[int | None, int | None]:
        # Triton has no hit counter; this kernel's variant count stands in for
        # `misses`, so a delta across the launch means a fresh compile.
        return None, _triton_cache_size(self._kernel)

    def __getitem__(self, grid: Any) -> Callable[..., Any]:
        # kernel[grid](*args) is the launch path -- wrap the bound launcher in
        # the shared core so the compile that may fire inside it is recorded.
        launcher = self._kernel[grid]
        return _make_wrapper(launcher, self._op, "triton", self._key_fn, self._sample)

    def __getattr__(self, name: str) -> Any:
        # Delegate everything else (warmup, cache_key, wrap_triton hooks, ...)
        # to the real kernel. Only triggered for names not set on the proxy.
        return getattr(self._kernel, name)


def instrument_triton_kernel(
    op: str,
    *,
    key_fn: Callable[..., str] | None = None,
) -> Callable[[Any], InstrumentedTritonKernel]:
    """Instrument a single ``@triton.jit`` kernel, stacked above the jit::

        @instrument_triton_kernel("aten::bmm")
        @triton.jit
        def _bmm_kernel(...): ...

    Unlike CuTeDSL, a Triton kernel compiles lazily *inside* its
    ``kernel[grid](...)`` launch and caches variants on the kernel object. So
    rather than wrap a separate compile fn, wrap the kernel itself: the proxy
    watches that one kernel's variant count and records a compile when the
    launch grows it. Because compile and launch are fused, ``wall_ms`` on a
    miss is compile + host-launch latency (compile dominates); on a hit it is
    just host-launch latency.

    Args:
        op: Operator symbol being compiled for, e.g. ``"aten::bmm"``.
        key_fn: Optional callable with the kernel's launch-arg signature
            returning a short string for logs (the constexprs it sees, e.g.
            ``BLOCK``, are the actual compile key). Defaults to a repr.
    """

    def decorator(kernel: Any) -> InstrumentedTritonKernel:
        return InstrumentedTritonKernel(kernel, op, key_fn)

    return decorator


# ---------------------------------------------------------------------------
# Combined decorators: the recommended one-decorator surface for native ops.
# They apply the DSL's own cache/jit *and* the instrumentation, so the op
# author writes a single decorator on a bare function and can't get the
# stacking order wrong. The lower-level instrument_* helpers above remain for
# custom composition.
# ---------------------------------------------------------------------------


def instrumented_cutedsl_cache(
    op: str,
    *,
    key_fn: Callable[..., str] | None = None,
) -> Callable[[Callable[..., R]], Callable[..., R]]:
    """Cache + instrument a CuTeDSL compile function in one decorator::

        @instrumented_cutedsl_cache("aten::topk")
        def _compile_topk_radix(N, K, deterministic):
            return cute.compile(...)

    Equivalent to ``instrument_cutedsl_compile(op) `` stacked above the
    vendored ``@jit_cache``. ``jit_cache`` is imported lazily so plain
    ``import torch._native`` doesn't pull in the CuTeDSL runtime.
    """
    # Lazy import: quack.cache pulls in cutlass, absent on CPU-only builds.
    from torch._vendor.quack.cache import jit_cache

    def decorator(fn: Callable[..., R]) -> Callable[..., R]:
        return instrument_cutedsl_compile(op, key_fn=key_fn)(jit_cache(fn))

    return decorator


def instrumented_triton_cache(
    op: str,
    *,
    key_fn: Callable[..., str] | None = None,
) -> Callable[[Callable[..., R]], InstrumentedTritonKernel]:
    """JIT + instrument a Triton kernel in one decorator::

        @instrumented_triton_cache("aten::bmm")
        def _bmm_kernel(...):  # bare kernel body, no @triton.jit
            ...

    Equivalent to ``instrument_triton_kernel(op)`` stacked above
    ``@triton.jit``. ``triton`` is imported lazily so plain
    ``import torch._native`` doesn't pull in the Triton runtime.
    """
    # Lazy import: keep Triton out of `import torch._native`.
    import triton

    def decorator(fn: Callable[..., R]) -> InstrumentedTritonKernel:
        return instrument_triton_kernel(op, key_fn=key_fn)(triton.jit(fn))

    return decorator
