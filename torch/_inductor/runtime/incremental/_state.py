from __future__ import annotations

import contextlib
import threading
from collections import deque
from typing import TYPE_CHECKING

from torch.utils._ordered_set import OrderedSet

from .config import (
    filter_threshold_decay_exp,
    force_timing_if_lt_n_timings,
    initial_filter_threshold,
    max_timings_per_launcher,
    min_timings_before_filter,
    timed_sampling_rate,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from torch._inductor.runtime.triton_heuristics import CachingAutotuner

    from ._launcher import (
        Launcher,
        OnTimingUpdateSubscription,
        RawLauncher,
        RawLauncherView,
    )


# Precomputed threshold scale factors for sample counts 1..max_timings_per_launcher.
# Index i corresponds to sample_count == i+1.  Avoids repeated pow() calls.
_THRESHOLD_MULTIPLIERS: tuple[float, ...] = tuple(
    1.0 - ((n - 1) / (max_timings_per_launcher - 1)) ** filter_threshold_decay_exp
    for n in range(1, max_timings_per_launcher + 1)
)


class IncrementalAutotuneState:
    """Drives incremental autotuning across a set of ``Launcher`` candidates.

    Call the instance to dispatch one kernel invocation; over many calls
    the state collects timings, discards underperformers, and converges
    on a single best launcher.

    Threading model:
      * The dispatch thread (single, the user's thread that calls into
        the autotuner) runs ``__call__`` and the queue-walk inside
        ``_next_launcher``.
      * Worker threads run ``add_launcher`` /
        ``note_compilation_failed`` from ``AsyncCompile``'s pool or the
        streaming-compile reader. ``register_pending_compilations`` is
        called from the dispatch thread (plugin's ``_kick_off_*``).
      * The global resolver daemon (single thread, see
        ``_resolver.py``) fires per-launcher timing-update callbacks
        sequentially.

      ``self._lock`` guards every mutation of ``_queue``,
      ``_certified_launcher``, ``_provisional_subs``,
      ``_pending_compilations``, and ``_convergence_callback_fired``.

      Lock ordering: ``state.lock → launcher.lock``. The reverse path
      ``autotuner.lock → state.lock`` exists in
      ``CachingAutotuner.precompile`` (it calls ``plugin.pre_compile``,
      which calls ``register_pending_compilations``), so anything that
      takes ``autotuner.lock`` while holding ``state.lock`` would
      deadlock — namely user callbacks ``on_discard_fn`` and
      ``on_convergence_fn``. Both are deferred out of the locked
      region: ``_discard`` / ``_maybe_promote`` accept a
      ``pending_discards`` list to append to, and the ``_mutation``
      context manager fires them after releasing the lock; the
      convergence callback is fired in ``__call__`` after the locked
      check-and-pop.

      ``_provisional_launcher`` is intentionally lock-free: it's only
      a heuristic threshold reference, mutated by
      ``_maybe_update_provisional`` (which fires from the resolver
      thread without the state lock). Stale reads / racy writes
      self-correct on the next timing event.
    """

    def __init__(
        self,
        pre_launch_fn: Callable | None = None,
        post_launch_fn: Callable[[], None] | None = None,
        on_discard_fn: Callable[
            [Launcher, "RawLauncher | RawLauncherView"], None
        ]
        | None = None,
        on_convergence_fn: Callable[[IncrementalAutotuneState], None] | None = None,
    ) -> None:
        # Queue holds launchers that either still need timing runs or have
        # launched enough but are waiting for in-flight timings to land.
        # ``_certified_launcher`` is set once a launcher has reached its
        # full sample budget — its timing is the stable answer used at
        # convergence and as a threshold-check baseline.
        self._queue: deque[Launcher] = deque()
        self._certified_launcher: Launcher | None = None
        # Lowest-timing launcher across the queue and ``_certified_launcher``,
        # i.e. the best candidate so far whose timing hasn't yet been
        # certified by reaching the sample budget. Maintained incrementally
        # via the per-launcher timing-update callback; rescanned only when
        # its current holder is discarded.
        self._provisional_launcher: Launcher | None = None

        self._pre_launch_fn = pre_launch_fn
        self._post_launch_fn = post_launch_fn
        self._on_discard_fn = on_discard_fn
        self._on_convergence_fn = on_convergence_fn

        self._dispatch_count: int = 0

        # Per-launcher subscription handles, so we can deterministically
        # cancel callbacks on discard / convergence rather than relying
        # on GC alone. Keyed by launcher identity (Launcher uses default
        # object identity hashing).
        self._provisional_subs: dict[
            Launcher, OnTimingUpdateSubscription
        ] = {}

        # Per-launcher autotuner-side raw (or view). Looked up at
        # discard time so ``on_discard_fn`` can remove the right entry
        # from ``autotuner.launchers`` without scanning. Keyed by
        # Launcher identity.
        self._per_launcher_raw: dict[
            Launcher, "RawLauncher | RawLauncherView"
        ] = {}

        # Streaming-compile coordination. Producers (worker threads
        # compiling kernels) call ``register_pending_compilations(n)``
        # to declare that ``n`` more launchers are on the way, then
        # call ``add_launcher`` (success) or ``note_compilation_failed``
        # (failure) to satisfy each pending entry. Convergence is gated
        # on ``_pending_compilations == 0`` so the state can't pre-converge
        # while compiles are still in flight. ``_compile_signal`` lets
        # ``_next_launcher`` block efficiently for the first launcher.
        self._lock: threading.Lock = threading.Lock()
        self._pending_compilations: int = 0
        self._compile_signal: threading.Event = threading.Event()
        # ``on_convergence_fn`` is fired once per converged generation. The
        # flag is re-armed whenever new work arrives (add_launcher or
        # register_pending_compilations) so a multi-generation flow
        # (initial → rblock → coordesc → done) re-fires the callback at
        # each convergence point.
        self._convergence_callback_fired: bool = False

    # -- Lock helpers ------------------------------------------------------

    @contextlib.contextmanager
    def _mutation(
        self,
    ) -> Iterator[list[tuple[Launcher, "RawLauncher | RawLauncherView"]]]:
        """Acquire ``self._lock`` for a state-mutation block; on exit,
        release the lock and fire ``on_discard_fn`` for any launchers
        accumulated in the yielded list.

        The yielded list is the deferral channel: ``_discard`` /
        ``_maybe_promote`` append ``(launcher, raw)`` tuples, and the
        callbacks are fired outside the lock so they don't AB/BA against
        ``autotuner.lock`` (the reverse path exists in
        ``CachingAutotuner.precompile`` ⇒ ``plugin.pre_compile`` ⇒
        ``register_pending_compilations``). Doing the firing in
        ``__exit__`` (rather than releasing mid-mutation) keeps the
        lock continuously held across the full state mutation, avoiding
        any window where another thread can swap state out from under
        us.

        ``raw`` is the autotuner-side raw (or view) that was registered
        via ``add_launcher`` — passed through to ``on_discard_fn`` so
        it can remove the right entry from ``autotuner.launchers``
        without re-deriving it.
        """
        pending: list[tuple[Launcher, "RawLauncher | RawLauncherView"]] = []
        with self._lock:
            yield pending
        on_discard_fn = self._on_discard_fn
        if on_discard_fn is not None:
            for launcher, raw in pending:
                on_discard_fn(launcher, raw)

    # -- Producer API (worker threads) -------------------------------------

    def register_pending_compilations(self, n: int) -> None:
        """Declare that ``n`` more launchers are about to arrive.

        Called by the producer (e.g., the plugin's async-compile
        scheduler) before submitting compiles. Each pending entry must
        eventually be satisfied by either ``add_launcher`` (success) or
        ``note_compilation_failed`` (failure). Until all pending entries
        are satisfied, ``converged`` returns False so we don't terminate
        prematurely.
        """
        assert n > 0, f"register_pending_compilations requires n > 0, got {n}"
        with self._lock:
            self._pending_compilations += n
            self._convergence_callback_fired = False

    def note_compilation_failed(self) -> None:
        """Record that a previously-registered compilation failed.

        Decrements the pending counter without adding a launcher. Only
        signals the first-launcher waiter when the counter hits zero,
        since that's the transition the waiter cares about (it surfaces
        ``NoTritonConfigsError``); intermediate decrements leave the
        wait condition unchanged.
        """
        with self._lock:
            assert self._pending_compilations > 0, (
                "note_compilation_failed called with no pending compilations"
            )
            self._pending_compilations -= 1
            should_signal = self._pending_compilations == 0
        if should_signal:
            self._compile_signal.set()

    def add_launcher(
        self,
        launcher: Launcher,
        raw: "RawLauncher | RawLauncherView",
    ) -> None:
        """Register ``launcher`` as a candidate.

        If ``launcher`` already has a full sample budget (e.g., it came
        from the shared per-kernel pool with timings inherited from a
        prior autotuner), promote it directly against the cached best
        instead of queueing — so a state initialized with all-saturated
        launchers immediately converges without dispatching anything.

        ``raw`` is the autotuner-side raw / view that was placed in
        ``autotuner.launchers``. Stashed in ``_per_launcher_raw`` so
        ``on_discard_fn`` can find it later.

        Queue/promote happens *before* the pending-counter decrement
        under a single lock, so observers (``converged``,
        ``_next_launcher``'s outer block-or-return) can never see
        ``_pending_compilations == 0`` with the launcher missing from
        ``_queue`` / ``_certified_launcher``.
        """
        with self._mutation() as pending_discards:
            self._per_launcher_raw[launcher] = raw
            if launcher.num_available_timings >= max_timings_per_launcher:
                self._maybe_promote(launcher, pending_discards)
            else:
                self._queue.append(launcher)
                # Register the timing-update sub now (not just for
                # saturated launchers via ``_maybe_promote``) so the
                # provisional reflects this launcher as soon as its
                # first timing lands. Without this, the threshold-based
                # early-discard branch in ``_next_launcher`` has no
                # provisional reference during the formative-samples
                # window.
                self._register_provisional_sub(launcher)
            if self._pending_compilations > 0:
                self._pending_compilations -= 1
            self._convergence_callback_fired = False
        self._compile_signal.set()

    def register_compile(
        self,
        autotuner: CachingAutotuner,
        raw: RawLauncher,
        *,
        found_by_coordesc: bool = False,
    ) -> Launcher:
        """High-level registration: look up / create the shared
        ``Launcher`` for ``raw``, attach the chosen entry to
        ``autotuner.launchers``, and register with this state.

        Bundles the three-step dance (pool lookup → autotuner.launchers
        append → ``add_launcher``) so callers don't have to thread the
        per-autotuner ``chosen`` (raw or view) through manually.
        Returns the shared ``Launcher`` for further reference if
        needed.
        """
        from ._launcher import get_or_create_launcher

        launcher, chosen = get_or_create_launcher(autotuner, raw)
        if found_by_coordesc:
            chosen.found_by_coordesc = True
        with autotuner.lock:
            autotuner.launchers.append(chosen)
        self.add_launcher(launcher, chosen)
        return launcher

    # -- Internal mutation helpers (caller holds self._lock) ---------------

    def _maybe_promote(
        self,
        launcher: Launcher,
        pending_discards: list[
            tuple[Launcher, "RawLauncher | RawLauncherView"]
        ],
    ) -> None:
        """Compare a fully-sampled ``launcher`` against the cached best;
        the slower of the two is discarded.

        Caller must hold ``self._lock`` so the cert check-and-set is
        atomic — without it, two concurrent ``_maybe_promote`` calls
        could both see ``_certified_launcher is None`` and orphan one
        of the launchers.

        Appends ``(launcher, raw)`` for any discarded launchers to
        ``pending_discards`` for the caller to fire ``on_discard_fn``
        on after releasing the lock.
        """
        if self._certified_launcher is None:
            self._certified_launcher = launcher
            self._register_provisional_sub(launcher)
        elif launcher.timing < self._certified_launcher.timing:
            old_certified = self._certified_launcher
            self._certified_launcher = launcher
            self._discard(old_certified, pending_discards)
            self._register_provisional_sub(launcher)
        else:
            # ``launcher`` is the loser; ``_discard`` cleans it up and
            # we do NOT register a sub for it. (A sub registered here
            # would fire ``_maybe_update_provisional`` for an already-
            # discarded launcher, which can install it as
            # ``_provisional_launcher`` and leak through dispatch.)
            self._discard(launcher, pending_discards)

    def _register_provisional_sub(self, launcher: Launcher) -> None:
        """Caller must hold ``self._lock``. Wires
        ``_maybe_update_provisional`` as a timing-update callback on
        ``launcher`` and seeds the provisional with the launcher's
        current timing snapshot.

        No-op if a sub is already registered for this launcher in this
        state (e.g., the launcher was added unsaturated via
        ``add_launcher`` — which registers a sub — and later saturates
        via ``_next_launcher → _maybe_promote`` — which also tries to
        register one).

        Seeding with ``inf`` for a fresh launcher with no timings is
        intentional: ``_provisional_launcher`` becomes the launcher
        even though its representative timing is meaningless, but the
        threshold-discard branch in ``_next_launcher`` is gated on
        ``provisional.num_available_timings >= 1`` so an inf-timing
        provisional is filtered out from threshold checks until a real
        sample lands. The seed exists so subsequent timing updates
        compare against ``inf`` and immediately install a meaningful
        value.
        """
        if launcher in self._provisional_subs:
            return
        self._provisional_subs[launcher] = launcher.add_on_timing_update_fn(
            self._maybe_update_provisional
        )
        # No actual timing change here — pass the same value for old/new
        # so the rescan-on-slowdown branch isn't tripped on registration.
        current = launcher.timing
        self._maybe_update_provisional(launcher, current, current)

    def _discard(
        self,
        launcher: Launcher,
        pending_discards: list[
            tuple[Launcher, "RawLauncher | RawLauncherView"]
        ],
    ) -> None:
        """Drop ``launcher`` from tracking and queue an ``on_discard_fn``
        notification.

        Caller must hold ``self._lock``. Cancels the subscription
        synchronously: ``cancel()`` blocks until any in-flight
        ``_maybe_update_provisional`` invocation for this launcher
        completes (the callback doesn't take ``self._lock``, so this
        wait is safe under our lock).

        Appends ``(launcher, raw)`` to ``pending_discards`` instead of
        calling ``on_discard_fn`` directly, so the callback fires
        outside the lock — necessary because ``on_discard_fn`` takes
        ``autotuner.lock`` and the reverse direction
        ``autotuner.lock → self._lock`` exists in
        ``CachingAutotuner.precompile``.
        """
        sub = self._provisional_subs.pop(launcher, None)
        if sub is not None:
            sub.cancel()
        if launcher is self._provisional_launcher:
            self._recompute_provisional()
        self._maybe_converge_early()
        # ``pop`` (rather than ``get``) so the side-map shrinks as
        # launchers leave the state. Returns None if launcher was never
        # registered with a raw (defensive) or already popped.
        raw = self._per_launcher_raw.pop(launcher, None)
        if self._on_discard_fn is not None:
            pending_discards.append((launcher, raw))

    def _maybe_converge_early(self) -> None:
        """Promote the queue's last survivor as certified without
        waiting for ``max_timings_per_launcher``.

        Caller must hold ``self._lock``. Triggered after a discard: if
        the queue is down to a single launcher, no certified one
        exists, and no compiles are pending to bring more candidates,
        the survivor wins by process of elimination and we can skip
        the remaining confirmation dispatches.

        Triggers only from threshold-driven discards. The other
        ``_discard`` call site is inside ``_maybe_promote``, which has
        already set ``_certified_launcher`` before discarding the
        loser, so the ``is None`` guard is false and this is a no-op
        in that path. The pending-compilations guard prevents an early
        promote during streaming where another candidate is about to
        arrive.
        """
        if (
            self._certified_launcher is None
            and len(self._queue) == 1
            and self._pending_compilations == 0
        ):
            self._certified_launcher = self._queue.popleft()

    def _maybe_update_provisional(
        self, launcher: Launcher, old_timing: float, new_timing: float
    ) -> None:
        """Update ``_provisional_launcher`` in response to a timing change.

        Wired in as the per-launcher timing-update callback. Three cases:

        1. ``launcher`` is the current provisional and got slower
           (``new_timing > old_timing``): the provisional may no longer
           be the actual minimum, so rescan the queue and the certified
           launcher to refresh it.
        2. ``launcher`` is now strictly faster than the current
           provisional (or there is no provisional): take over.
        3. Otherwise: nothing to do.

        Together with ``_recompute_provisional`` on discard, this keeps
        ``_provisional_launcher`` in sync with the actual fastest
        launcher we know about at any point in time.

        Lock-free: called from the resolver thread without
        ``self._lock`` (and synchronously from ``_maybe_promote`` with
        the lock held). Reads/writes of ``_provisional_launcher`` race
        between the two — heuristic, self-correcting on the next event.
        """
        if launcher is self._provisional_launcher and new_timing > old_timing:
            self._recompute_provisional()
            return
        if (
            self._provisional_launcher is None
            or new_timing < self._provisional_launcher.timing
        ):
            self._provisional_launcher = launcher

    def _recompute_provisional(self) -> None:
        """Re-scan candidates and refresh ``_provisional_launcher``. O(n).

        Lock-free for the same reason as ``_maybe_update_provisional``:
        the snapshot of ``_queue`` / ``_certified_launcher`` may be
        stale, but the result is heuristic and corrects on the next
        event.
        """
        candidates: list[Launcher] = list(self._queue)
        if self._certified_launcher is not None:
            candidates.append(self._certified_launcher)
        self._provisional_launcher = (
            min(candidates, key=lambda l: l.timing) if candidates else None
        )

    # -- Read-only views ---------------------------------------------------

    @property
    def certified_launcher(self) -> Launcher | None:
        # Only meaningful at convergence: the stable, definitively best
        # launcher. Use ``current_certified`` for mid-autotune reads
        # (e.g., from a convergence callback).
        assert self.converged, "certified_launcher is only valid while converged"
        return self._certified_launcher

    @property
    def certified_timing(self) -> float:
        assert self.converged, "certified_timing is only valid while converged"
        if self._certified_launcher is None:
            return float("inf")
        return self._certified_launcher.timing

    @property
    def current_certified(self) -> Launcher | None:
        """Current certified launcher, without the converged-only
        assertion of ``certified_launcher``. Intended for use inside
        ``on_convergence_fn`` (where the state is mid-mutation between
        generations) and other in-progress reads."""
        return self._certified_launcher

    def chosen_for(
        self, launcher: Launcher
    ) -> "RawLauncher | RawLauncherView | None":
        """Return the autotuner-side raw / view registered for
        ``launcher`` via ``add_launcher``, or ``None`` if there's no
        entry (launcher was discarded or never registered)."""
        return self._per_launcher_raw.get(launcher)

    def replace_chosen(
        self,
        launcher: Launcher,
        raw: "RawLauncher | RawLauncherView",
    ) -> None:
        """Update the autotuner-side raw / view recorded for
        ``launcher``. Used by callers that swap a placeholder view for
        a freshly-compiled raw (e.g., the plugin's compile-on-finalize
        path)."""
        self._per_launcher_raw[launcher] = raw

    @property
    def converged(self) -> bool:
        # The queue drains as launchers either become best or get
        # discarded; ``_pending_compilations`` blocks convergence while
        # producers (the plugin's async-compile scheduler) still have
        # launchers in flight that haven't been added yet.
        return not self._queue and self._pending_compilations == 0

    # -- Dispatch ----------------------------------------------------------

    def _next_launcher(self) -> Launcher | None:
        """Pop the next launcher to dispatch, blocking when streaming
        compiles are still in flight and no fallback is available.

        Returns the next launcher for tuning mode, or ``None`` if the
        caller should fall back to ``_certified_launcher`` /
        ``_provisional_launcher``. Raises ``NoTritonConfigsError`` when
        there's nothing to dispatch and nothing more is coming
        (initial-set all-failed) — the unique state shape
        ``(_queue=[], _certified_launcher=None,
        _provisional_launcher=None, _pending_compilations=0)`` is only
        reachable that way, since every other path keeps at least one
        of cert / provisional set.

        Per-iteration locking: each queue-walk iteration takes
        ``self._lock`` (via ``_mutation``) so worker ``add_launcher``
        calls can squeeze in between iterations. The lock is released
        between iterations, including across ``_compile_signal.wait``.
        """
        seen_waiting: OrderedSet[int] = OrderedSet()
        # Track the queue size at the last "all-waiting" tally point so
        # we can detect worker-side ``add_launcher`` calls that grew
        # the queue between iterations and reset cycle detection
        # accordingly (otherwise we'd return None after cycling through
        # the originally-queued waiters even though new ready launchers
        # are sitting behind them).
        last_seen_queue_size: int = 0
        # Snapshot provisional once for stable threshold reference
        # across the walk (matches the prior unlocked design).
        with self._lock:
            provisional = self._provisional_launcher
        while True:
            wait_after = False
            with self._mutation() as pending_discards:
                if not self._queue:
                    if (
                        self._certified_launcher is not None
                        or self._provisional_launcher is not None
                    ):
                        return None
                    if self._pending_compilations == 0:
                        # Lazy import to avoid the circular dep with
                        # triton_heuristics.
                        from torch._inductor.runtime.triton_heuristics import (
                            NoTritonConfigsError,
                        )

                        raise NoTritonConfigsError(
                            "Incremental autotune: all initial configs "
                            "failed to compile"
                        )
                    wait_after = True
                else:
                    # Queue grew since our last cycle-detect bookkeeping
                    # — worker added new launchers we haven't examined.
                    # Reset seen_waiting so we don't bail before walking
                    # the new entries.
                    if len(self._queue) > last_seen_queue_size:
                        seen_waiting = OrderedSet()
                    launcher = self._queue.popleft()
                    available = launcher.num_available_timings

                    # 1) Done — promote against the cached best, discard the loser.
                    if available >= max_timings_per_launcher:
                        self._maybe_promote(launcher, pending_discards)
                        continue

                    # 2) Threshold-based early discard against the snapshotted
                    # temp best. Each side gets its own permissiveness based on
                    # its sample count; we multiply so a noisy candidate or a
                    # noisy baseline both relax the bar. Skip the baseline
                    # itself so we never discard our reference point, and
                    # require at least one sample on the baseline so the
                    # multiplier index is in range.
                    if (
                        provisional is not None
                        and launcher is not provisional
                        and available >= min_timings_before_filter
                        and provisional.num_available_timings >= 1
                    ):
                        cand_mult = max(
                            1.0,
                            1.0
                            + (initial_filter_threshold - 1.0)
                            * _THRESHOLD_MULTIPLIERS[available - 1],
                        )
                        provisional_mult = max(
                            1.0,
                            1.0
                            + (initial_filter_threshold - 1.0)
                            * _THRESHOLD_MULTIPLIERS[
                                provisional.num_available_timings - 1
                            ],
                        )
                        if (
                            launcher.timing
                            > cand_mult * provisional_mult * provisional.timing
                        ):
                            self._discard(launcher, pending_discards)
                            continue

                    # 3) Waiting — launched enough but not all resolved yet.
                    if launcher.num_total_timings >= max_timings_per_launcher:
                        if id(launcher) in seen_waiting:
                            self._queue.append(launcher)
                            return None
                        seen_waiting.add(id(launcher))
                        self._queue.append(launcher)
                        last_seen_queue_size = len(self._queue)
                        continue

                    # 4) Ready to dispatch.
                    return launcher
            # Only reachable for the wait case. Lock released; pending
            # discards (none in this branch) fired in __exit__.
            if wait_after:
                self._compile_signal.wait(timeout=1.0)
                self._compile_signal.clear()

    def __call__(self, *args: object, stream: object, **kwargs: object) -> object:
        # Converged path: fire the convergence callback once per
        # generation (re-armed by ``add_launcher`` /
        # ``register_pending_compilations``) and dispatch the certified
        # launcher.
        sub_to_cancel = None
        should_fire = False
        with self._lock:
            if (
                self.converged
                and self._certified_launcher is not None
                and not self._convergence_callback_fired
                and self._on_convergence_fn is not None
            ):
                self._convergence_callback_fired = True
                should_fire = True
                sub_to_cancel = self._provisional_subs.pop(
                    self._certified_launcher, None
                )

        if should_fire:
            # Cancel our timing-update subscription on the surviving
            # launcher so we don't keep this state alive via the global
            # pool. Then fire the convergence callback (which may queue
            # more compiles and re-arm ``_convergence_callback_fired``).
            # Both run with the lock released — ``cancel()`` blocks on
            # the subscription's ``_in_flight`` lock, and
            # ``on_convergence_fn`` calls back into
            # ``register_pending_compilations``, which takes
            # ``self._lock``.
            if sub_to_cancel is not None:
                sub_to_cancel.cancel()
            assert self._on_convergence_fn is not None
            self._on_convergence_fn(self)

        # ``_certified_launcher`` is never re-nulled, so the unlocked
        # read here is safe; the converged check below detects the case
        # where the callback added more work.
        if self.converged and self._certified_launcher is not None:
            return self._launch(
                self._certified_launcher,
                *args,
                stream=stream,
                time_if_warm=False,
                requeue=False,
                **kwargs,
            )

        # Tuning path. ``_next_launcher`` blocks if needed and raises
        # ``NoTritonConfigsError`` if everything failed. The loop
        # handles "invalid configuration" RuntimeErrors by discarding
        # the offending launcher and trying the next one, without
        # recursing (which would blow the stack on many invalid
        # configs in a row).
        while (launcher := self._next_launcher()) is not None:
            self._dispatch_count += 1
            # We *want* to time iff we're still inside the forced-timing
            # window for this launcher OR this dispatch lands on a
            # sampling beat. The Launcher additionally gates on its own
            # warm state inside its lock, so a freshly-switched front
            # never gets timed against on its first call. Either way we
            # requeue: more timings may still be needed and
            # ``_next_launcher`` will pull the launcher back out when
            # it's done.
            time_if_warm = (
                launcher.num_total_timings < force_timing_if_lt_n_timings
                or self._dispatch_count % timed_sampling_rate == 0
            )
            try:
                return self._launch(
                    launcher,
                    *args,
                    stream=stream,
                    time_if_warm=time_if_warm,
                    requeue=True,
                    **kwargs,
                )
            except RuntimeError as e:
                # Triton surfaces invalid kernel configs (e.g. requested
                # block size exceeds device limits) as a RuntimeError with
                # "invalid configuration" in the message. Drop the launcher
                # and try the next.
                if "invalid configuration" not in str(e).lower():
                    raise
                with self._mutation() as pending_discards:
                    self._discard(launcher, pending_discards)
                # Loop and try the next ready launcher.

        # ``_next_launcher`` returned None, so cert or provisional is
        # non-None (otherwise it would have raised).
        if self._certified_launcher is not None:
            return self._launch(
                self._certified_launcher,
                *args,
                stream=stream,
                time_if_warm=False,
                requeue=False,
                **kwargs,
            )
        return self._launch(
            self._provisional_launcher,
            *args,
            stream=stream,
            time_if_warm=False,
            requeue=False,
            **kwargs,
        )

    def _launch(
        self,
        launcher: Launcher | None,
        *args: object,
        stream: object,
        time_if_warm: bool,
        requeue: bool,
        **kwargs: object,
    ) -> object:
        """Run ``launcher`` (timed only if it's warm and ``time_if_warm``
        is set), optionally re-queueing it."""
        assert launcher is not None
        try:
            if self._pre_launch_fn is not None:
                self._pre_launch_fn(launcher, *args, stream=stream, **kwargs)
            result = launcher(
                *args, stream=stream, time_if_warm=time_if_warm, **kwargs
            )
        finally:
            if self._post_launch_fn is not None:
                self._post_launch_fn()
        if requeue:
            with self._lock:
                self._queue.append(launcher)
        return result
