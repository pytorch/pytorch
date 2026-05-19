from __future__ import annotations

import bisect
import hashlib
import json
import re
import threading
import weakref
from collections.abc import Hashable, Iterable
from typing import cast, Protocol, TYPE_CHECKING

import torch

from torch._inductor.runtime.runtime_utils import triton_config_to_hashable
from torch._inductor.runtime.triton_heuristics import CachingAutotuner

from ._resolver import submit_event
from .config import (
    max_timings_per_launcher,
    launcher_timing_aggregation,
    LauncherTimingAggregation,
)


if TYPE_CHECKING:
    from collections.abc import Callable


class RawLauncher(Protocol):
    """Structural shape of an entry in ``CachingAutotuner.launchers``.

    The ``CachingAutotuner`` does not export a typed launcher class;
    the raw launchers are anonymous callables annotated by
    ``triton_heuristics`` with ``config``, ``cache_hash`` and friends
    after construction. This Protocol pins down the attributes the
    incremental autotuner actually relies on. ``RawLauncherView``
    structurally satisfies it.
    """

    config: object
    cache_hash: str | None
    # ``found_by_coordesc`` is set by ``_finalize`` on the chosen raw
    # (or its view) so the autotuner's post-dispatch coordesc gate
    # skips on subsequent dispatches.
    found_by_coordesc: bool

    def __call__(self, *args: object, **kwargs: object) -> object: ...


class RawLauncherView:
    """Per-autotuner view of a (possibly shared) raw launcher.

    When two ``CachingAutotuner`` instances autotune the same
    kernel/config they share the same compiled binary (the
    ``RawLauncher``) — but each autotuner has its own ``cache_hash``
    (per-binary identifier from its own compile, since the same source
    can hash differently across binaries even when the kernel content
    is equivalent). The view wraps the real raw, overrides
    ``cache_hash`` with the per-autotuner value, and delegates
    everything else to the underlying raw.

    ``cache_hash`` may be ``None`` when stage-1 sharing skipped this
    autotuner's compile entirely (the view was created directly from
    a cached pool entry, with no fresh compile to derive a per-autotuner
    cache_hash from). The plugin's ``_finalize`` detects ``None`` and
    compiles on-demand to resolve it before ``save_cache_hook``.

    Holds a strong reference to the real raw so it stays alive for as
    long as any view (i.e., any sharing autotuner's ``launchers`` list)
    references it.
    """

    def __init__(
        self,
        real_raw: RawLauncher,
        cache_hash: str | None,
    ) -> None:
        self._real_raw = real_raw
        self.cache_hash = cache_hash
        # ``found_by_coordesc`` is per-autotuner: two sharing autotuners
        # may discover the same launcher via different paths (initial set
        # vs coordesc neighbor). Default False; ``_finalize`` flips it to
        # True for autotuners whose coordesc applied.
        self.found_by_coordesc: bool = False

    def __getattr__(self, name: str) -> object:
        # Any attribute not explicitly overridden falls through to the
        # real raw — config, n_regs, n_spills, shared, store_cubin,
        # _is_static, etc.
        return getattr(self._real_raw, name)

    def __call__(self, *args: object, **kwargs: object) -> object:
        return self._real_raw(*args, **kwargs)

    def __reduce__(self) -> tuple[object, ...]:
        # Without an explicit ``__reduce__``, pickle would route
        # ``__getstate__`` / ``__reduce_ex__`` introspection through
        # our ``__getattr__`` to ``_real_raw``, silently dropping the
        # per-autotuner ``cache_hash`` and ``found_by_coordesc``
        # overrides. Re-create the view explicitly. Note this still
        # requires ``_real_raw`` to be picklable; that's the same
        # constraint as pickling the underlying raw directly, which
        # ``CachingAutotuner.prepare_for_pickle`` already deals with.
        return (
            self.__class__,
            (self._real_raw, self.cache_hash),
            {"found_by_coordesc": self.found_by_coordesc},
        )

    def __setstate__(self, state: dict) -> None:
        self.found_by_coordesc = state.get("found_by_coordesc", False)


class OnTimingUpdateSubscription:
    """Handle returned by ``Launcher.add_on_timing_update_fn``.

    Owns the lifecycle of a single registered timing-update callback.
    The callback fires under ``_in_flight``, so at most one invocation
    of this callback is ever active. ``cancel()`` takes the same lock
    to set ``_active=False``: if a fire is in progress when ``cancel``
    is called, ``cancel`` blocks until it completes; once ``cancel``
    returns, the callback is guaranteed not to fire again. ``cancel``
    also physically removes the subscription from the parent
    Launcher's registry so future ``add_timing`` calls don't even
    iterate over it. Idempotent.

    Also held via :class:`weakref.WeakMethod`, so an owning instance
    that is garbage-collected without explicit ``cancel()`` still
    causes the subscription to fall dormant — ``is_alive()`` returns
    False and the next ``add_timing`` prunes it from the registry.
    """

    def __init__(
        self,
        fn: Callable[[Launcher, float, float], None],
        launcher: Launcher,
    ) -> None:
        self._weak_method: weakref.WeakMethod[
            Callable[[Launcher, float, float], None]
        ] = weakref.WeakMethod(fn)
        self._active: bool = True
        self._in_flight: threading.Lock = threading.Lock()
        # Weak back-ref so cancel() can remove this sub from the
        # Launcher's registry without keeping the Launcher alive.
        self._launcher_ref: weakref.ref[Launcher] = weakref.ref(launcher)

    def is_alive(self) -> bool:
        return self._active and self._weak_method() is not None

    def fire(self, launcher: Launcher, old_timing: float, new_timing: float) -> None:
        with self._in_flight:
            if not self._active:
                return
            cb = self._weak_method()
            if cb is None:
                return
            cb(launcher, old_timing, new_timing)

    def cancel(self) -> None:
        """Block until any in-flight invocation completes, prevent
        future invocations, AND remove this subscription from the
        parent Launcher's registry. Idempotent.
        """
        with self._in_flight:
            self._active = False
        launcher = self._launcher_ref()
        if launcher is not None:
            launcher.remove_subscription(self)


class Launcher:
    """A timing-collecting wrapper around a single raw kernel launcher.

    Holds a single weak reference to its current raw launcher (the
    front-of-queue model is gone — see the per-kernel pool docs for
    why). Strong ownership of the raw lives in
    ``CachingAutotuner.launchers`` (or in a ``RawLauncherView`` held by
    a sharing autotuner's ``launchers``). When the raw dies, the
    Launcher is "dormant": its accumulated timing samples remain valid
    but it can't dispatch until a fresh raw is attached via
    :func:`get_or_create_launcher`.

    Sharing model:
      Two ``IncrementalAutotuneState`` instances autotuning the same
      kernel/config share a single Launcher via the per-kernel pool.
      The pool stores Launchers; per-state strong refs to the raw live
      in ``autotuner.launchers`` (possibly via a per-autotuner
      ``RawLauncherView`` that overrides ``cache_hash``).

    Threading: ``__call__`` is the only method guaranteed to run on
    the main (dispatch) thread; every other public method on Launcher
    is fair game to be called from any thread (the resolver daemon
    fires ``add_timing`` and arbitrary owners may register/remove
    callbacks or query ``timing``). All shared state — ``_raw_ref``,
    ``_warm``, ``_timings``, ``_num_timings_in_flight``,
    ``_on_timing_update_subs`` — is protected by ``_lock``. ``__call__``
    takes the lock briefly to snapshot the current raw and flip
    ``_warm``, but releases it before the actual kernel dispatch so
    concurrent callers aren't blocked behind a long-running launch.
    """

    def __init__(self, raw_launcher: RawLauncher) -> None:
        self._raw_ref: weakref.ref[RawLauncher] | None = weakref.ref(raw_launcher)
        self._timings: list[float] = []
        self._num_timings_in_flight: int = 0
        self._warm: bool = False
        self._lock: threading.Lock = threading.Lock()
        self._on_timing_update_subs: list[OnTimingUpdateSubscription] = []

    def remove_subscription(self, sub: OnTimingUpdateSubscription) -> None:
        """Drop ``sub`` from this Launcher's registry. Idempotent —
        already-pruned subscriptions are a no-op.
        """
        with self._lock:
            try:
                self._on_timing_update_subs.remove(sub)
            except ValueError:
                pass

    def add_on_timing_update_fn(
        self, fn: Callable[[Launcher, float, float], None]
    ) -> OnTimingUpdateSubscription:
        """Register a bound-method callback fired after every ``add_timing``
        and return a ``OnTimingUpdateSubscription`` handle.

        The callback receives ``(self, old_timing, new_timing)``: the
        aggregated timing immediately before and after the new sample
        landed. Subscribers can use the diff to detect slow-downs (e.g.,
        a tracker that rescans only when ``new_timing > old_timing``).

        ``fn`` is held via :class:`weakref.WeakMethod`, so it is pruned
        automatically once its owning instance is garbage-collected.
        For deterministic cleanup (e.g., on ``_discard`` / convergence),
        call ``sub.cancel()`` on the returned subscription: cancel waits
        for any in-flight invocation to finish before returning, so the
        callback is provably retired when ``cancel`` returns.

        ``fn`` must be a bound method (``WeakMethod`` requires it).
        """
        sub = OnTimingUpdateSubscription(fn, self)
        with self._lock:
            self._on_timing_update_subs.append(sub)
        return sub

    def set_raw_launcher(self, raw_launcher: RawLauncher) -> None:
        """Attach a fresh raw to a previously-dormant Launcher.

        Used by :func:`get_or_create_launcher` when the pool's existing
        Launcher has lost its raw (autotuner that owned it filtered or
        was GC'd) and a new autotuner shows up with a fresh raw.
        Resets ``_warm`` because the new raw hasn't been dispatched
        through this Launcher yet.
        """
        with self._lock:
            self._raw_ref = weakref.ref(raw_launcher)
            self._warm = False

    def get_raw_launcher(self) -> RawLauncher | None:
        """Return the live raw launcher, or ``None`` if it's been collected."""
        with self._lock:
            if self._raw_ref is None:
                return None
            return self._raw_ref()

    def __call__(
        self,
        *args: object,
        time_if_warm: bool = False,
        **kwargs: object,
    ) -> object:
        """Dispatch the kernel.

        When ``time_if_warm`` is True and the raw is currently warm
        (has already been dispatched at least once via this Launcher),
        perform a CUDA-event-timed dispatch; otherwise dispatch
        untimed. ``_warm`` flips to True after the dispatch so the next
        ``time_if_warm=True`` call can actually time.

        The "is the raw warm?" decision is made inside ``self._lock``
        and after the strong reference to the raw is captured — that
        way the decision is consistent with the raw we actually
        dispatch on, even if other threads are racing on the launcher.
        Letting the Launcher gate timing internally (rather than
        exposing a warm-flag for callers to inspect and act on
        non-atomically) closes the read-then-dispatch window.
        """
        raw_launcher = self.get_raw_launcher()
        assert raw_launcher is not None, (
            "Launcher's raw launcher has been garbage-collected"
        )
        with self._lock:
            timed = time_if_warm and self._warm
            self._warm = True
        if not timed:
            return raw_launcher(*args, **kwargs)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        # Two failure modes for the raw launcher under timed dispatch:
        #   A) Launch failure (e.g., invalid configuration argument):
        #      raises synchronously here, before we increment
        #      _num_timings_in_flight, so the counter stays consistent
        #      and the caller observes the error normally.
        #   B) Execution failure (e.g., illegal memory access): is
        #      asynchronous and only surfaces on the next device
        #      sync (typically far away, on a different code path). We
        #      don't reconcile the counter for this case — such failures
        #      are assumed fatal to the entire program, so a small
        #      bookkeeping desync on a Launcher that is about to be
        #      discarded is moot.
        result = raw_launcher(*args, **kwargs)
        end_event.record()
        with self._lock:
            self._num_timings_in_flight += 1
        submit_event(self.add_timing, start_event, end_event)
        return result

    def add_timing(self, elapsed_ms: float) -> None:
        """Record a timing sample, decrement the in-flight counter, and fire
        every live on-timing-update subscription with ``(self, old_timing,
        new_timing)``. Cancelled subscriptions and subscriptions whose
        owning instance has been garbage-collected are pruned in place.
        """
        with self._lock:
            old_timing = self._aggregate_timings()
            bisect.insort(self._timings, elapsed_ms)
            self._num_timings_in_flight -= 1
            new_timing = self._aggregate_timings()
            live = self._get_live_subs()
        # Fire outside the launcher lock — each subscription's ``fire``
        # takes its own ``_in_flight`` lock around the actual callback,
        # which guarantees that ``cancel()`` returns only after this
        # invocation completes (and prevents any future invocations).
        for sub in live:
            sub.fire(self, old_timing, new_timing)

    def _get_live_subs(self) -> list[OnTimingUpdateSubscription]:
        """Return a snapshot of live on-timing-update subscriptions,
        dropping cancelled or owner-collected entries from the registry
        as a side effect.

        Caller must hold ``self._lock`` — this both reads and rebuilds
        ``self._on_timing_update_subs``. The returned list is a
        separate copy so concurrent ``add_on_timing_update_fn`` calls
        (which append to the registry under the lock) don't appear
        partway through the snapshot.
        """
        live: list[OnTimingUpdateSubscription] = [
            sub for sub in self._on_timing_update_subs if sub.is_alive()
        ]
        self._on_timing_update_subs = list(live)
        return live

    @property
    def num_available_timings(self) -> int:
        with self._lock:
            return len(self._timings)

    @property
    def num_in_flight_timings(self) -> int:
        with self._lock:
            return self._num_timings_in_flight

    @property
    def num_total_timings(self) -> int:
        with self._lock:
            return len(self._timings) + self._num_timings_in_flight

    def _aggregate_timings(self) -> float:
        """Return the aggregated timing per the configured strategy.

        Caller must hold ``self._lock``. Returns ``inf`` when no samples
        have landed yet (so a fresh launcher compares as worst).
        """
        if not self._timings:
            return float("inf")
        match launcher_timing_aggregation:
            case LauncherTimingAggregation.MEAN:
                return sum(self._timings) / len(self._timings)
            case LauncherTimingAggregation.MEDIAN:
                n = len(self._timings)
                mid = n // 2
                if n % 2 == 1:
                    return self._timings[mid]
                return (self._timings[mid - 1] + self._timings[mid]) / 2
            case _:
                # Defensive: LauncherTimingAggregation is exhaustively matched
                # above so this branch is currently unreachable. Keep it
                # so that adding a new LauncherTimingAggregation member without
                # a matching case fails loudly instead of silently
                # returning the wrong aggregation.
                raise ValueError(f"Unknown timing aggregation: {launcher_timing_aggregation!r}")

    @property
    def timing(self) -> float:
        """Representative timing aggregated per the configured strategy."""
        with self._lock:
            return self._aggregate_timings()


# --------------------------------------------------------------------------
# Per-kernel Launcher dedup. Two ``IncrementalAutotuneState`` instances that
# compile the same kernel/config pair share a single Launcher (and its
# accumulated timings) via the registry below.
# --------------------------------------------------------------------------


# Per-kernel pool of Launcher instances, keyed by a content hash of the
# CachingAutotuner (see :func:`_caching_autotuner_instance_key`) and then
# by the Triton config (a ``Hashable`` produced by
# ``triton_config_to_hashable``). Two CachingAutotuner instances whose
# kernels hash to the same key share the same inner dict, so two
# autotuners compiling the same kernel reuse Launchers (and their
# accumulated timings) per matching config.
#
# This dict is the only persistent strong reference to Launcher objects
# across the lifetime of the program. That's intentional and cheap:
# Launcher only weakrefs the underlying raw launcher, so the device
# memory pinned by raw launchers is not held here. Lifetime of those raw
# launchers is owned by ``CachingAutotuner.launchers`` (directly, or via
# a ``RawLauncherView`` for sharing autotuners).
_caching_autotuner_launcher_registry: dict[str, dict[Hashable, Launcher]] = {}

# Single lock guarding all reads/writes of every registry in this module.
# Today the only registry is ``_caching_autotuner_launcher_registry``; if
# we add more in the future they should share this lock so callers don't
# have to reason about a per-registry locking matrix.
_registry_lock: threading.Lock = threading.Lock()


def _caching_autotuner_instance_key(autotuner: CachingAutotuner) -> str:
    """Return a content hash that identifies ``autotuner``'s kernel.

    Two CachingAutotuner instances with matching keys are considered to
    have equivalent underlying kernels — so a Launcher built for one can
    safely be reused for the other (when their Triton configs also
    match).

    Raises ``ValueError`` when ``autotuner`` is missing one of the
    attributes we hash over. We want such cases to fail loudly so they
    can be investigated and fixed rather than silently dropping the
    autotuner out of the shared pool.
    """
    fn = autotuner.fn
    # Attributes hashed below. Several of these (``size_hints``,
    # ``heuristic_type``, ``custom_kernel``, ``mutated_arg_names``,
    # ``reset_to_zero_arg_names``, ``optimize_mem``) are NOT in
    # ``inductor_meta`` / ``triton_meta`` at runtime — ``cached_autotune``
    # pops them out before constructing the autotuner. They still
    # affect autotune correctness (compile-result selection, benchmark
    # arg cloning / zeroing, register-spill filtering), so we include
    # them in the key explicitly.
    missing = [
        attr
        for obj, attr in (
            (fn, "src"),
            (fn, "__name__"),
            (autotuner, "inductor_meta"),
            (autotuner, "triton_meta"),
            (autotuner, "size_hints"),
            (autotuner, "heuristic_type"),
            (autotuner, "custom_kernel"),
            (autotuner, "mutated_arg_names"),
            (autotuner, "reset_to_zero_arg_names"),
            (autotuner, "optimize_mem"),
        )
        if not hasattr(obj, attr)
    ]
    if missing:
        raise ValueError(
            f"CachingAutotuner instance is missing required attributes: {missing}"
        )

    # Inductor mangles each generated kernel's name with an incrementing
    # suffix (``triton_foo_0``, ``triton_foo_1``, ...) so symbol names
    # don't collide. Two structurally-identical kernels would therefore
    # hash differently if we kept the names. Normalize every occurrence
    # of the kernel name in ``fn.src`` (and in the inductor metadata,
    # which embeds the same name) to a fixed sentinel so the resulting
    # hash depends only on kernel content. Whole-word matching avoids
    # clobbering substrings of unrelated identifiers that happen to
    # contain the kernel name.
    name: str = fn.__name__
    name_re = re.compile(rf"\b{re.escape(name)}\b")
    normalized_src = name_re.sub("triton_", fn.src)
    normalized_inductor = name_re.sub(
        "triton_",
        json.dumps(autotuner.inductor_meta, sort_keys=True, default=repr),
    )
    # ``triton_meta`` does not embed the kernel name, so it goes in
    # as-is.
    raw_triton = json.dumps(autotuner.triton_meta, sort_keys=True, default=repr)
    # Extra attributes that aren't reflected in inductor_meta /
    # triton_meta but affect autotune correctness.
    extra_attrs = json.dumps(
        {
            "size_hints": autotuner.size_hints,
            "heuristic_type": repr(autotuner.heuristic_type),
            "custom_kernel": autotuner.custom_kernel,
            "mutated_arg_names": list(autotuner.mutated_arg_names),
            "reset_to_zero_arg_names": list(autotuner.reset_to_zero_arg_names),
            "optimize_mem": autotuner.optimize_mem,
        },
        sort_keys=True,
        default=repr,
    )

    # Length-prefix each chunk so two distinct (src, inductor, triton, extra)
    # tuples whose concatenated bytes would otherwise align cannot collide.
    hasher = hashlib.sha256()
    for chunk in (normalized_src, normalized_inductor, raw_triton, extra_attrs):
        encoded = chunk.encode("utf-8")
        hasher.update(len(encoded).to_bytes(8, "big"))
        hasher.update(encoded)
    return hasher.hexdigest()


def _get_autotuner_specific_registry(
    autotuner: object,
) -> dict[Hashable, Launcher]:
    """Return the ``config -> Launcher`` pool for ``autotuner``.

    ``autotuner`` is typed as ``object`` so additional autotuner types
    can be plugged in later without touching the call sites; today only
    ``CachingAutotuner`` is recognized. Any other type raises
    ``ValueError`` (we'd rather hear about an unsupported caller than
    silently bypass the pool). ``CachingAutotuner`` instances that are
    missing the attributes needed for content-keying (see
    :func:`_caching_autotuner_instance_key`) also raise ``ValueError``.
    """
    if isinstance(autotuner, CachingAutotuner):
        key = _caching_autotuner_instance_key(autotuner)
        return _caching_autotuner_launcher_registry.setdefault(key, {})
    raise ValueError(
        f"Unsupported autotuner type {type(autotuner).__name__}; "
        "only CachingAutotuner is recognized today."
    )


def _get_or_create_launchers(
    autotuner: object,
    raw_launchers: Iterable[RawLauncher],
) -> list[tuple[Launcher, RawLauncher | RawLauncherView]]:
    """Unlocked worker shared by :func:`get_or_create_launcher` and
    :func:`get_or_create_launchers`.

    For each input raw launcher, returns a tuple
    ``(Launcher, raw_or_view)``. The second element is what should be
    placed in ``autotuner.launchers``:

    * A new ``RawLauncher`` (the one passed in) if this autotuner is
      the sole / first owner of the Launcher (pool miss, or pool hit
      with a dead raw).
    * A ``RawLauncherView`` wrapping another autotuner's raw if a live
      raw was already attached to this Launcher in the pool (sharing).

    Caller is responsible for holding ``_registry_lock``.
    """
    if not isinstance(autotuner, CachingAutotuner):
        raise ValueError(
            f"Unsupported autotuner type {type(autotuner).__name__}; "
            "only CachingAutotuner is recognized today."
        )
    pool = _get_autotuner_specific_registry(autotuner)
    result: list[tuple[Launcher, RawLauncher | RawLauncherView]] = []
    for raw_launcher in raw_launchers:
        new_raw = cast(RawLauncher, raw_launcher)
        cfg_key = triton_config_to_hashable(new_raw.config)
        existing = pool.get(cfg_key)
        if existing is None:
            # Pool miss — this autotuner is the first to register a
            # Launcher for this config. Use the new raw as-is.
            launcher = Launcher(raw_launcher=new_raw)
            pool[cfg_key] = launcher
            chosen: RawLauncher | RawLauncherView = new_raw
        else:
            launcher = existing
            live_raw = launcher.get_raw_launcher()
            if live_raw is not None:
                # Sharing path. Reuse the existing live raw via a
                # per-autotuner view so this autotuner's ``cache_hash``
                # is preserved for cache persistence.
                chosen = RawLauncherView(
                    cast(RawLauncher, live_raw),
                    new_raw.cache_hash,
                )
            else:
                # Pool hit but the prior raw is gone. Attach the new
                # raw and use it.
                launcher.set_raw_launcher(new_raw)
                chosen = new_raw
        result.append((launcher, chosen))
    return result


def get_or_create_launcher(
    autotuner: object,
    raw_launcher: RawLauncher,
) -> tuple[Launcher, RawLauncher | RawLauncherView]:
    """Return ``(Launcher, raw_to_register)`` for ``raw_launcher``.

    The Launcher is fetched from (or registered into) the shared
    per-kernel pool so two autotuners with the same kernel and
    overlapping configs share Launchers and accumulated timings. The
    second element of the returned tuple is the raw (or view) that the
    caller should place in ``autotuner.launchers`` — see
    :func:`_get_or_create_launchers` for the case dispatch.

    Raises ``ValueError`` when ``autotuner`` isn't a recognized type or
    when its kernel can't be content-keyed (see
    :func:`_get_autotuner_specific_registry`).
    """
    with _registry_lock:
        return _get_or_create_launchers(autotuner, [raw_launcher])[0]


def get_or_create_launchers(
    autotuner: object,
    raw_launchers: Iterable[RawLauncher],
) -> list[tuple[Launcher, RawLauncher | RawLauncherView]]:
    """Batch form of :func:`get_or_create_launcher`.

    Acquires ``_registry_lock`` once for the whole batch instead of once
    per launcher. Same contract as :func:`get_or_create_launcher` for
    each ``(autotuner, raw_launcher)`` pair.
    """
    with _registry_lock:
        return _get_or_create_launchers(autotuner, raw_launchers)


def lookup_launcher(
    autotuner: object, config: object
) -> Launcher | None:
    """Return the existing pool ``Launcher`` for ``(autotuner-kernel,
    config)``, or ``None`` if none is registered.

    Read-only — does not mutate the pool, does not create a Launcher,
    does not attach a raw. Used by the plugin's post-convergence
    kick-off paths to skip submitting compiles for configs whose
    Launchers are already in the pool from prior autotuners (stage 1
    cache-hit).

    Returns ``None`` for unrecognized autotuner types or for configs
    that ``triton_config_to_hashable`` can't hash (best-effort lookup —
    a non-hashable config just means "no cache hit, please compile").
    """
    if not isinstance(autotuner, CachingAutotuner):
        return None
    try:
        cfg_key = triton_config_to_hashable(config)
    except (AttributeError, TypeError):
        return None
    with _registry_lock:
        pool = _get_autotuner_specific_registry(autotuner)
        return pool.get(cfg_key)
