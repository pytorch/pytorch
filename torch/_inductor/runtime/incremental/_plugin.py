from __future__ import annotations

import hashlib
import threading
from typing import TYPE_CHECKING

from torch._inductor.config import triton as inductor_triton_config
from torch._inductor.runtime.runtime_utils import triton_config_to_hashable
from torch._inductor.runtime.triton_heuristics import CachingAutotunerPlugin, DEFER

from ._launcher import get_or_create_launcher
from ._state import IncrementalAutotuneState
from ._utils import jk_passes, log
from .config import should_include


if TYPE_CHECKING:
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner

    from ._launcher import Launcher, RawLauncher, RawLauncherView


# Cross-recompilation registry: when two recompilations produce identical
# kernels (differing only in the generated name suffix), the second
# recompilation reuses the same ``IncrementalAutotuneState`` -- whether
# converged or still in progress. Keyed by a content hash strong enough
# that matching keys produce binary-identical launchers.
_shared_autotuner_registry: dict[str, IncrementalAutotuneState] = {}


def _kernel_content_key(autotuner: CachingAutotuner) -> str:
    """SHA-256 over normalized source, inductor/Triton metadata, and configs.

    Including ``triton_meta`` guarantees identical compiled binaries, so
    launchers from a matching shared state are directly interchangeable.
    """
    fn = autotuner.fn
    name = fn.__name__
    normalized_src = fn.src.replace(name, "triton_")
    normalized_inductor = str(sorted(autotuner.inductor_meta.items())).replace(
        name, "triton_"
    )
    normalized_triton = str(sorted(autotuner.triton_meta.items())).replace(
        name, "triton_"
    )
    config_keys = sorted(
        triton_config_to_hashable(launcher.config) for launcher in autotuner.launchers
    )

    hasher = hashlib.sha256()
    hasher.update(normalized_src.encode("utf-8"))
    hasher.update(normalized_inductor.encode("utf-8"))
    hasher.update(normalized_triton.encode("utf-8"))
    hasher.update(str(config_keys).encode("utf-8"))
    return hasher.hexdigest()


class IncrementalAutotunePlugin(CachingAutotunerPlugin):
    """Spread autotuning across real kernel invocations instead of blocking up
    front. Two cooperation modes:

    - Standalone: standard precompile populates ``autotuner.launchers``;
      ``pre_autotune`` snapshots them and initializes an
      ``IncrementalAutotuneState``. Subsequent dispatches go through the
      state until convergence; once converged the autotuner's launchers
      collapse to the single best launcher and the standard fast-path
      takes over.
    - Pipelining + incremental: ``pre_compile`` sees the streaming handle
      attached by ``AsyncCompile.triton`` and initializes the state up
      front. A background drain reads launchers off ``handle.launcher_q``
      as the worker streams them and feeds them into the state via
      ``register_compile``, so the dispatch path can start tuning before
      all configs have compiled.

    In either mode an existing ``IncrementalAutotuneState`` may be joined
    from the cross-recompilation registry: if it has already converged,
    the autotuner skips straight to the winner; otherwise it tags along
    and dispatches through the in-progress state.
    """

    def __init__(self) -> None:
        self._state: IncrementalAutotuneState | None = None
        self._is_shared: bool = False
        self._drain_thread: threading.Thread | None = None

    def pre_compile(self, autotuner) -> object:
        """Streaming combo entry point. If the pipelining plugin attached a
        handle to the autotuner, initialize the incremental state here and
        spawn a drain thread that feeds it from the handle's launcher_q.
        Otherwise DEFER -- the standalone path runs in ``pre_autotune``
        after standard precompile.
        """
        handle = getattr(autotuner, "_pipeline_caching_autotuner_handle", None)
        if handle is None:
            return DEFER
        if not self._should_apply(autotuner):
            return DEFER
        self._initialize_for_streaming(autotuner, handle)
        return DEFER

    def pre_autotune(self, autotuner, *args, stream, **kwargs):
        """Standalone entry point. Standard precompile has populated
        ``autotuner.launchers``; snapshot them into a fresh state, then
        dispatch this call through it. In the streaming combo the state
        is already initialized by ``pre_compile`` -- just dispatch.
        """
        if self._state is not None:
            return self._state(*args, stream=stream, **kwargs)
        if not self._should_apply(autotuner):
            return DEFER
        self._initialize_standard(autotuner)
        if self._state is None:
            return DEFER
        return self._state(*args, stream=stream, **kwargs)

    def pre_dispatch(self, autotuner, *args, stream, **kwargs):
        """Route dispatch through the active state, or DEFER if none.

        When the active state is one we joined from the registry and it
        converged on another autotuner, finalize locally (set the single
        launcher + write to the cache hook) and DEFER so the autotuner's
        standard single-launcher dispatch path runs.
        """
        if self._state is None:
            return DEFER
        if self._is_shared and self._state.converged:
            self._finalize_shared(autotuner)
            return DEFER
        return self._state(*args, stream=stream, **kwargs)

    @staticmethod
    def _should_apply(autotuner) -> bool:
        if not autotuner.inductor_meta.get("incremental_autotune", False):
            return False
        if inductor_triton_config.autotune_at_compile_time:
            log.warning(
                "Incremental autotune: skipping %s - autotune_at_compile_time=True",
                autotuner.fn.__name__,
            )
            return False
        if inductor_triton_config.cudagraphs:
            log.warning(
                "Incremental autotune: skipping %s - cudagraphs=True",
                autotuner.fn.__name__,
            )
            return False
        if not jk_passes():
            log.warning(
                "Incremental autotune: skipping %s - JK gate blocked",
                autotuner.fn.__name__,
            )
            return False
        if not should_include(autotuner.heuristic_type):
            log.debug(
                "Incremental autotune: skipping %s - heuristic_type %s not included",
                autotuner.fn.__name__,
                autotuner.heuristic_type,
            )
            return False
        log.debug("Incremental autotune: enabled for %s", autotuner.fn.__name__)
        return True

    def _make_callbacks(self, autotuner):
        """Build the (on_convergence, on_discard) pair for a state owning
        ``autotuner``. Convergence: collapse ``autotuner.launchers`` to the
        winner and fire ``save_cache_hook``. Discard: remove the chosen
        entry from ``autotuner.launchers``.
        """

        def on_convergence(state):
            launcher = state.current_certified
            assert launcher is not None
            chosen = state.chosen_for(launcher)
            assert chosen is not None
            with autotuner.lock:
                autotuner.launchers = [chosen]
                self._state = None
            log.info(
                "Incremental autotune converged for %s: %s (timing=%.3f ms)",
                autotuner.fn.__name__,
                chosen.config,
                launcher.timing,
            )
            if autotuner.save_cache_hook:
                autotuner.save_cache_hook(
                    chosen.config,
                    0,
                    found_by_coordesc=getattr(chosen, "found_by_coordesc", False),
                    triton_cache_hash=chosen.cache_hash,
                )

        def on_discard(launcher, raw):
            with autotuner.lock:
                try:
                    autotuner.launchers.remove(raw)
                except ValueError:
                    pass

        return on_convergence, on_discard

    def _try_join_shared(self, autotuner, content_key) -> bool:
        """If a shared state exists for ``content_key``, adopt it and return
        True (caller skips state creation). Converged shared states collapse
        ``autotuner.launchers`` to the winner; in-progress ones leave them
        empty and route dispatch through the shared state.
        """
        shared = _shared_autotuner_registry.get(content_key)
        if shared is None:
            return False
        if shared.converged:
            launcher = shared.current_certified
            assert launcher is not None
            chosen = shared.chosen_for(launcher)
            assert chosen is not None
            autotuner.launchers = [chosen]
            log.info(
                "Incremental autotune: %s using converged shared state id=%d "
                "(config=%s, timing=%.3f ms)",
                autotuner.fn.__name__,
                id(shared),
                chosen.config,
                launcher.timing,
            )
            return True
        autotuner.launchers = []
        self._state = shared
        self._is_shared = True
        log.info(
            "Incremental autotune: %s joining shared state id=%d (in progress)",
            autotuner.fn.__name__,
            id(shared),
        )
        return True

    def _initialize_standard(self, autotuner) -> None:
        """Standalone path: snapshot ``autotuner.launchers`` into a fresh
        state. Adopts a shared state when one is registered for this kernel.
        """
        content_key = (
            _kernel_content_key(autotuner) if hasattr(autotuner.fn, "src") else None
        )
        if content_key is not None and self._try_join_shared(autotuner, content_key):
            return

        raws = list(autotuner.launchers)
        autotuner.launchers = []

        on_convergence, on_discard = self._make_callbacks(autotuner)
        self._state = IncrementalAutotuneState(
            pre_launch_fn=autotuner._pre_launch,
            post_launch_fn=autotuner._post_launch,
            on_convergence_fn=on_convergence,
            on_discard_fn=on_discard,
        )
        for raw in raws:
            self._state.register_compile(autotuner, raw)
        log.info(
            "Incremental autotune: initializing %s with %d launchers (state id=%d)",
            autotuner.fn.__name__,
            len(raws),
            id(self._state),
        )
        if content_key is not None:
            _shared_autotuner_registry[content_key] = self._state

    def _initialize_for_streaming(self, autotuner, handle) -> None:
        """Pipelining combo path: bg drain is already feeding launchers into
        ``handle.launcher_q``. Set up a state expecting ``handle.num_configs``
        compiles and spawn a thread that drains the queue, registering each
        launcher (and counting failures as the END sentinel is reached).
        """
        # Bg drain also appends each launcher to ``autotuner.launchers``;
        # clear that so ``register_compile`` is the sole writer.
        with autotuner.lock:
            autotuner.launchers = []

        on_convergence, on_discard = self._make_callbacks(autotuner)
        self._state = IncrementalAutotuneState(
            pre_launch_fn=autotuner._pre_launch,
            post_launch_fn=autotuner._post_launch,
            on_convergence_fn=on_convergence,
            on_discard_fn=on_discard,
        )
        self._state.register_pending_compilations(handle.num_configs)
        log.info(
            "Incremental autotune: initializing %s for streaming with "
            "%d pending compilations (state id=%d)",
            autotuner.fn.__name__,
            handle.num_configs,
            id(self._state),
        )
        self._drain_thread = threading.Thread(
            target=self._drain_loop,
            args=(autotuner, handle),
            daemon=True,
            name="inductor-incremental-drain",
        )
        self._drain_thread.start()

    def _drain_loop(self, autotuner, handle) -> None:
        """Read launchers off ``handle.launcher_q`` until the END sentinel,
        feeding each into the state. Any compiles that didn't surface as a
        launcher are charged as failures so the state's pending counter
        drains to zero.
        """
        from torch._inductor.async_compile import _LAUNCHER_END

        state = self._state
        assert state is not None
        seen = 0
        try:
            while True:
                item = handle.launcher_q.get()
                if item is _LAUNCHER_END:
                    break
                state.register_compile(autotuner, item)
                seen += 1
        finally:
            # Either the bg drain finished normally (END sentinel) or it
            # crashed (drain_future has the exception). Either way: any
            # pending compilations we didn't see must be accounted for so
            # the state can converge.
            remaining = max(0, handle.num_configs - seen)
            for _ in range(remaining):
                state.note_compilation_failed()

    def _finalize_shared(self, autotuner) -> None:
        """Take ownership of the converged shared state's winner without
        mutating the shared state -- other autotuners may still be
        dispatching through it.
        """
        state = self._state
        assert state is not None and state.converged
        launcher = state.current_certified
        assert launcher is not None
        chosen = state.chosen_for(launcher)
        assert chosen is not None
        with autotuner.lock:
            autotuner.launchers = [chosen]
            self._state = None
            self._is_shared = False
        log.debug(
            "Incremental autotune: %s finalized from shared state "
            "(config=%s, timing=%.3f ms)",
            autotuner.fn.__name__,
            chosen.config,
            launcher.timing,
        )
        if autotuner.save_cache_hook:
            autotuner.save_cache_hook(
                chosen.config,
                0,
                found_by_coordesc=getattr(chosen, "found_by_coordesc", False),
                triton_cache_hash=chosen.cache_hash,
            )
