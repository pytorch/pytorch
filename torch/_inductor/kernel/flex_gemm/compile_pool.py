"""Compile FlexGEMM autotune candidates in Inductor's compile workers.

NOTE [FlexGEMM compile workers]
CuTeDSL compilation is not thread-safe, so in-process candidate compiles are
serialized by ``CUTEDSL_COMPILE_LOCK`` and N candidates cost N compiles of wall
time. QuACK's ``jit_cache`` has a defer-and-retry protocol for this: with a
compile pool active, a cold miss is submitted to the pool and raises
``CompilePending``; the worker exports the ``.o`` into QuACK's disk cache and
the retry just loads it. QuACK's own pool ships the EpiMod to workers by value
(cloudpickle). FlexGEMM instead ships the *recipe*: the generated kernel module
(already on disk in the Inductor cache) plus the template config, and the worker
rebuilds the EpiMod with the same semantic digest. Every failure on this path
falls back to the in-process compile.
"""

from __future__ import annotations

import base64
import dataclasses
import functools
import logging
import pickle
import time
from typing import Any, TYPE_CHECKING

from torch._inductor import config
from torch._inductor.async_compile import _pycodecache_kernel_compile_env, AsyncCompile
from torch._logging import warning_once
from torch._vendor.quack.cache import async_compile as quack_async


if TYPE_CHECKING:
    import torch
    from torch._inductor.kernel.flex_gemm.template import FlexGemmEpilogueConfig


log = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class FlexGemmCompileRecipe:
    """What a compile worker needs to rebuild one choice's EpiMod."""

    module_key: str
    module_path: str
    config: FlexGemmEpilogueConfig
    input_dtypes: tuple[torch.dtype, ...]


def install_flex_gemm_epimod(
    expected_digest: str, recipe: FlexGemmCompileRecipe
) -> None:
    """Worker-side payload installer: rebuild and register the EpiMod."""
    from torch._inductor.codecache import PyCodeCache
    from torch._vendor.quack.gemm_runtime.identity import register_local_epi_mod

    module = PyCodeCache.load_by_key_path(
        recipe.module_key, recipe.module_path, set_sys_modules=False
    )
    config = recipe.config
    epimod = config.epimod(
        getattr(module, config.epilogue_name),
        recipe.input_dtypes,
        lambda name: getattr(module, name),
    )
    if epimod.semantic_digest != expected_digest:
        raise RuntimeError(
            f"FlexGEMM compile worker rebuilt EpiMod {epimod.semantic_digest[:12]}, "
            f"expected {expected_digest[:12]}"
        )
    register_local_epi_mod(expected_digest, epimod)


# Inductor's workers are persistent and shared across tasks; pin once per process.
_pin_worker_arch = functools.cache(quack_async.pin_worker_arch)


def _compile_in_worker(
    quack_arch: str | None,
    cute_dsl_arch: str | None,
    extra_env: dict[str, str | None],
    *args: Any,
) -> str | None:
    from torch._inductor.runtime.compile_tasks import (
        _apply_subprocess_env_and_clear_caches,
    )
    from torch._vendor.quack.gemm_runtime.host import _compile_gemm_epi

    _apply_subprocess_env_and_clear_caches(extra_env)
    _pin_worker_arch(quack_arch, cute_dsl_arch)
    # The exported .o is the only result; an in-memory hit from an earlier task
    # (e.g. under a previous cache root) would skip the export.
    _compile_gemm_epi.cache_clear()
    return quack_async._pool_worker(*args)


@functools.cache
def workers_ready() -> bool:
    """Whether Inductor's compile-worker pool is usable, decided once per process.

    The blocking wait covers the pool warmup race; a negative answer (or a
    warmup timeout) must not be re-waited for every candidate.
    """
    try:
        return AsyncCompile.wait_process_pool_ready()
    except Exception as e:
        warning_once(log, "Inductor compile worker pool unavailable (%s)", e)
        return False


class InductorCompilePool(quack_async.CompilePool):
    """QuACK compile pool for one choice, backed by Inductor's compile workers.

    ``submit`` replaces QuACK's by-value payload with ``recipe`` and returns
    False on any submission problem so ``jit_cache`` compiles in-process.
    """

    def __init__(self, recipe: FlexGemmCompileRecipe) -> None:
        # pyrefly: ignore [bad-argument-type]
        super().__init__(executor=AsyncCompile.process_pool())
        self.recipe = recipe

    def submit(self, sha: str, fn: Any, args: tuple, kwargs: dict, o_path: Any) -> bool:
        try:
            payload = quack_async.PoolPayload(
                __name__,
                install_flex_gemm_epimod.__name__,
                args[0].semantic_digest,
                self.recipe,
            )
            key_b64 = base64.b64encode(pickle.dumps((args, kwargs))).decode("ascii")
            payloads_b64 = base64.b64encode(pickle.dumps([payload])).decode("ascii")
            self._futures[sha] = self._executor.submit(
                _compile_in_worker,
                *quack_async._detect_arch_env(),
                _pycodecache_kernel_compile_env(),
                fn.__module__,
                fn.__qualname__,
                key_b64,
                str(o_path),
                payloads_b64,
            )
            self.n_submitted += 1
        except Exception as e:
            warning_once(
                log,
                "FlexGEMM compile worker submission failed (%s); compiling in-process",
                e,
            )
            return False
        return True

    def wait(self, sha: str) -> bool:
        """Block until the worker compiling ``sha`` finishes; False if it failed.

        The deadline stays well inside Inductor's precompile timeout so the
        in-process fallback still counts.
        """
        deadline = time.monotonic() + config.precompilation_timeout_seconds / 2
        while (state := self.poll(sha)[0]) == "pending" and time.monotonic() < deadline:
            time.sleep(0.01)
        if state != "done":
            log.warning(
                "FlexGEMM worker precompile %s (%s); compiling in-process",
                state,
                self.poll(sha)[1],
            )
            return False
        return True
