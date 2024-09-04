from __future__ import annotations

import dataclasses
import hashlib
import logging
import os
import os.path
from typing import Dict, List, Optional, Tuple
from typing_extensions import override

import torch
from torch.utils._triton import has_triton_package

from ..remote_cache import (
    JsonDataTy,
    RemoteCache,
    RemoteCacheBackend,
    RemoteCacheJsonSerde,
)


if has_triton_package():
    from triton import Config

log = logging.getLogger(__name__)


_InductorMetaTy = Dict[str, object]


@dataclasses.dataclass
class AutotuneCache:
    configs_hash: str
    filename: str
    local_cache: Optional[Tuple[RemoteCache[JsonDataTy], str]] = None
    remote_cache: Optional[Tuple[RemoteCache[JsonDataTy], str]] = None

    # Create a AutotuneCache. Returns None if none of the caches can be used.
    @staticmethod
    def create(
        inductor_meta: _InductorMetaTy, filename: str, configs_hash: str
    ) -> Optional[AutotuneCache]:
        cache = AutotuneCache(configs_hash, filename)
        cache._setup_local_cache(inductor_meta, filename)
        cache._setup_remote_autotune_cache(inductor_meta, filename)
        if cache.local_cache or cache.remote_cache:
            return cache
        else:
            return None

    # Read the best config options from the most local cache and return it.
    def _read(self, inductor_meta: _InductorMetaTy) -> Optional[Dict[str, JsonDataTy]]:
        if local_cache := self.local_cache:
            cache, key = local_cache
            if best_config := cache.get(key):
                if isinstance(best_config, dict):
                    return best_config

        if remote_cache := self.remote_cache:
            cache, key = remote_cache
            if best_config := cache.get(key):
                if isinstance(best_config, dict):
                    return best_config

        return None

    # Read the best config options from the most local cache and figure out
    # which `configs` represents that option.
    def read_best(
        self, inductor_meta: _InductorMetaTy, configs: List[Config]
    ) -> Optional[Config]:
        if best := self._read(inductor_meta):
            return _load_cached_autotuning(
                best, self.configs_hash, configs, inductor_meta
            )
        return None

    # Set up local filesystem caching information
    def _setup_local_cache(self, inductor_meta: _InductorMetaTy, filename: str) -> None:
        if not inductor_meta.get("autotune_local_cache", True):
            return

        cache_filename = os.path.splitext(filename)[0] + ".best_config"
        local_cache = RemoteCache(_LocalAutotuneCacheBackend(), RemoteCacheJsonSerde())
        self.local_cache = (local_cache, cache_filename)

    # Set up remote caching information
    def _setup_remote_autotune_cache(
        self, inductor_meta: _InductorMetaTy, filename: str
    ) -> None:
        if not _should_use_remote_autotune_cache(inductor_meta):
            return

        remote_cache = _create_cache(
            inductor_meta,
            self.configs_hash,
            "FbRemoteAutotuneCache",
            "RemoteAutotuneCache",
            "autotune-best-config-v2",
        )
        if not remote_cache:
            return

        # we already sha256 hash the source contents
        remote_cache_key = os.path.basename(filename)
        self.remote_cache = (remote_cache, remote_cache_key)

    # Save the config in the caches
    def save(
        self, config: Config, time_taken_ns: int, found_by_coordesc: bool = False
    ) -> None:
        data = {
            **config.kwargs,
            "num_warps": config.num_warps,
            "num_stages": config.num_stages,
            "configs_hash": self.configs_hash,
            "found_by_coordesc": found_by_coordesc,
            "time_taken_ms": time_taken_ns // 1000000,  # Convert from NS to MS
        }

        if local_cache := self.local_cache:
            cache, key = local_cache
            cache.put(key, data)

            if log.isEnabledFor(logging.DEBUG):
                type_str = "coordesc" if found_by_coordesc else "heuristic"
                log.debug("Save %s tuning result to %s", type_str, key)

        if remote_cache := self.remote_cache:
            cache, key = remote_cache
            cache.put(key, data)


def _should_use_remote_autotune_cache(inductor_meta: Dict[str, object]) -> bool:
    if (config := inductor_meta.get("autotune_remote_cache")) is not None:
        return bool(config)
    if not inductor_meta.get("is_fbcode"):
        return False
    if torch._utils_internal.is_fb_unit_test():
        return False
    if inductor_meta.get("is_hip"):
        return False

    try:
        from torch._inductor.fb.remote_cache import REMOTE_CACHE_VERSION
    except ModuleNotFoundError:
        return False

    return REMOTE_CACHE_VERSION >= torch._utils_internal.justknobs_getval_int(
        "pytorch/remote_cache:autotune_memcache_version"
    )


def _load_cached_autotuning(
    best_config: Dict[str, JsonDataTy],
    configs_hash: str,
    configs: List[Config],
    inductor_meta: Dict[str, object],
) -> Optional[Config]:
    if best_config is None:
        return None
    if best_config.pop("configs_hash", None) != configs_hash:
        return None

    # Remove time taken for comparison
    best_config.pop("time_taken_ms", None)

    if inductor_meta.get("coordinate_descent_tuning") and best_config.pop(
        "found_by_coordesc", False
    ):
        num_warps = best_config.pop("num_warps")
        num_stages = best_config.pop("num_stages")
        triton_config = Config(best_config, num_warps=num_warps, num_stages=num_stages)
        triton_config.found_by_coordesc = True
        return triton_config

    matching_configs = [
        cfg
        for cfg in configs
        if all(val == best_config.get(key) for key, val in cfg.kwargs.items())
        and cfg.num_warps == best_config.get("num_warps")
        and cfg.num_stages == best_config.get("num_stages")
    ]
    if len(matching_configs) != 1:
        return None

    return matching_configs[0]


def _create_cache(
    inductor_meta: Dict[str, object],
    configs_hash: str,
    fb_cache_cls: str,
    oss_cache_cls: str,
    salt: str,
) -> Optional[RemoteCache[JsonDataTy]]:
    backend_hash = inductor_meta.get("backend_hash", None)
    if backend_hash is None:
        log.debug(
            "backend_hash is not passed on the inductor_meta, unable to use autotune remote cache"
        )
        return None

    assert isinstance(backend_hash, str)

    key = backend_hash + configs_hash + salt
    key = hashlib.sha256(key.encode("utf-8")).hexdigest()

    try:
        if inductor_meta.get("is_fbcode"):
            import torch._inductor.fb.remote_cache

            cache_cls = getattr(torch._inductor.fb.remote_cache, fb_cache_cls)
            return cache_cls(key)
        else:
            import torch._inductor.remote_cache

            cache_cls = getattr(torch._inductor.remote_cache, oss_cache_cls)
            return cache_cls(key)
    except Exception:
        log.warning("Unable to create a remote cache", exc_info=True)
        return None


class _LocalAutotuneCacheBackend(RemoteCacheBackend[bytes]):
    @override
    def get(self, key: str) -> Optional[bytes]:
        try:
            with open(key, "rb") as fd:
                return fd.read()
        except FileNotFoundError:
            return None

    @override
    def put(self, key: str, data: bytes) -> None:
        with open(key, "wb") as fd:
            fd.write(data)
