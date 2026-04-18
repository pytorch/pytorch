from __future__ import annotations

import functools
import weakref
from typing import TYPE_CHECKING

import torch._utils_internal

from ..triton_heuristics import CachingAutotunerPlugin, DEFER
from . import _cache
from ._launcher import Launcher
from ._state import IncrementalAutotuneState
from ._utils import log

if TYPE_CHECKING:
    from ..triton_heuristics import CachingAutotuner


# Bump this to enable incremental autotuning for users who have
# config.incremental_autotune = True (or the env var set to "1").
# The JK "pytorch/inductor:incremental_autotune_version" must be
# <= this value for the feature to activate.
_INCREMENTAL_AUTOTUNE_VERSION: int = 1


@functools.cache
def _jk_passes() -> bool:
    """Return True if the JK gate allows incremental autotuning."""
    try:
        val = torch._utils_internal.justknobs_getval_int(
            "pytorch/inductor:incremental_autotune_version"
        )
    except AttributeError:
        return True
    return _INCREMENTAL_AUTOTUNE_VERSION >= val


class IncrementalAutotunePlugin(CachingAutotunerPlugin):
    """Plugin that drives incremental autotuning for a CachingAutotuner.

    Holds the per-autotuner IncrementalAutotuneState. The state is created
    lazily on the first ``pre_autotune`` call and cleared by the
    on-convergence callback.
    """

    def __init__(self) -> None:
        self._state: IncrementalAutotuneState | None = None

    def pre_dispatch(
        self,
        autotuner: CachingAutotuner,
        *args: object,
        stream: object,
        **kwargs: object,
    ) -> object:
        if self._state is not None:
            return self._state.dispatch(*args, stream=stream, **kwargs)
        return DEFER

    def pre_autotune(
        self,
        autotuner: CachingAutotuner,
        *args: object,
        stream: object,
        **kwargs: object,
    ) -> object:
        self._state = self._init_state(autotuner)
        if self._state is not None:
            return self._state.dispatch(*args, stream=stream, **kwargs)
        return DEFER

    @staticmethod
    def _should_apply(autotuner: CachingAutotuner) -> bool:
        """Check whether incremental autotuning applies to ``autotuner``."""
        from torch._inductor import config as inductor_config
        from torch._inductor.config import triton as inductor_triton_config

        name: str = autotuner.fn.__name__
        if not inductor_config.incremental_autotune:
            return False
        if inductor_triton_config.autotune_at_compile_time:
            log.warning(
                "Incremental autotune: skipping %s — autotune_at_compile_time=True",
                name,
            )
            return False
        if inductor_triton_config.cudagraphs:
            log.warning(
                "Incremental autotune: skipping %s — cudagraphs=True",
                name,
            )
            return False
        if not _jk_passes():
            log.warning(
                "Incremental autotune: skipping %s — JK gate blocked",
                name,
            )
            return False
        log.info("Incremental autotune: enabled for %s", name)
        return True

    def _init_state(
        self, autotuner: CachingAutotuner
    ) -> IncrementalAutotuneState | None:
        """Create and seed an IncrementalAutotuneState for ``autotuner``.

        Returns None if incremental autotuning does not apply, or if the state
        immediately converged from shared stats (in which case the autotuner
        already has its single launcher set).
        """
        if not IncrementalAutotunePlugin._should_apply(autotuner):
            return None

        # Capture plugin and autotuner via weakref so the closures held by
        # ``state.on_convergence_fn`` / ``on_cleanup_fn`` don't form a cycle
        # (plugin._state -> state -> closure -> plugin / autotuner) that would
        # delay deterministic finalization of state.__del__ → on_cleanup.
        plugin_ref = weakref.ref(self)
        autotuner_ref = weakref.ref(autotuner)

        def on_convergence(state: IncrementalAutotuneState) -> None:
            plugin = plugin_ref()
            owner = autotuner_ref()
            if plugin is None or owner is None:
                return
            assert state.best_launcher is not None
            best = state.best_launcher
            owner.launchers = [best]
            plugin._state = None
            log.info(
                "Incremental autotune converged for %s: %s (timing=%.3f ms)",
                owner.fn.__name__,
                best.metadata.get("config"),
                state.best_timing,
            )

        def on_cleanup(state: IncrementalAutotuneState) -> None:
            if state.best_launcher is None:
                return
            owner = autotuner_ref()
            if owner is None:
                return
            best = state.best_launcher
            # Incremental autotune does not track wall time the same way
            # autotune_to_one_config does; report 0 to avoid skewing metrics.
            owner.autotune_time_taken_ns = 0
            if owner.save_cache_hook:
                owner.save_cache_hook(
                    best.metadata.get("config"),
                    0,
                    found_by_coordesc=getattr(
                        best.metadata.get("config"), "found_by_coordesc", False
                    ),
                    triton_cache_hash=best.metadata.get("cache_hash"),
                )

        state = IncrementalAutotuneState(
            on_convergence_fn=on_convergence,
            on_cleanup_fn=on_cleanup,
            pre_launch_fn=autotuner._pre_launch,
            post_launch_fn=autotuner._post_launch,
        )

        # Build the per-state launcher list from the shared per-config pool:
        # configs already seen by another autotuner reuse the existing Launcher
        # (and its accumulated timings); novel configs add fresh Launchers to
        # the pool. ``pool`` is None if the kernel can't be content-keyed.
        pool = _cache.get_launcher_pool(autotuner)
        if pool is None:
            launchers = [
                Launcher(
                    fn=raw_launcher,
                    config=raw_launcher.config,
                    cache_hash=raw_launcher.cache_hash,
                )
                for raw_launcher in autotuner.launchers
            ]
        else:
            launchers = [
                _cache.get_or_create_launcher(pool, raw_launcher, Launcher)
                for raw_launcher in autotuner.launchers
            ]
        state.init_fresh(launchers)
        log.info(
            "Incremental autotune: initializing %s with %d launchers "
            "(state id=%d, pool size=%d)",
            autotuner.fn.__name__,
            len(launchers),
            id(state),
            len(pool) if pool is not None else 0,
        )

        autotuner.launchers = []

        if state.converged and state.best_launcher:
            best = state.best_launcher
            autotuner.launchers = [best]
            state.shutdown()
            log.info(
                "Incremental autotune: %s immediately converged from shared "
                "stats (config=%s, timing=%.3f ms)",
                autotuner.fn.__name__,
                best.metadata.get("config"),
                state.best_timing,
            )
            return None

        return state
