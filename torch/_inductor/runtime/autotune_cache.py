from __future__ import annotations

import dataclasses
import hashlib
import logging
import os
import os.path
import re
from typing import Dict, List, Optional, TYPE_CHECKING
from typing_extensions import override

import torch
from torch.utils._triton import has_triton

from ..remote_cache import (
    create_cache,
    JsonDataTy,
    RemoteCache,
    RemoteCacheBackend,
    RemoteCacheJsonSerde,
)
from .triton_compat import Config


if TYPE_CHECKING:
    from ..remote_cache import Sample

log = logging.getLogger(__name__)


_InductorMetaTy = Dict[str, object]


def inductor_meta_from_config() -> _InductorMetaTy:
    from torch._inductor import config

    backend_hash = None
    if has_triton():
        try:
            backend_hash = torch.utils._triton.triton_hash_with_backend()
        except RuntimeError:
            # This can get the error:
            #   RuntimeError: 0 active drivers ([]). There should only be one.
            pass

    is_hip = None
    if torch.version.hip is not None:
        is_hip = True

    return {
        "autotune_local_cache": config.autotune_local_cache,
        "autotune_remote_cache": config.autotune_remote_cache,
        "backend_hash": backend_hash,
        "bundled_autotune_remote_cache": config.bundled_autotune_remote_cache,
        "coordinate_descent_tuning": config.coordinate_descent_tuning,
        "is_fbcode": config.is_fbcode(),
        "is_hip": is_hip,
    }


@dataclasses.dataclass
class AutotuneCache:
    configs_hash: str
    local_cache: Optional[tuple[RemoteCache[JsonDataTy], str]] = None
    remote_cache: Optional[tuple[RemoteCache[JsonDataTy], str]] = None

    # Create a AutotuneCache. Returns None if none of the caches can be used.
    @staticmethod
    def create(
        inductor_meta: _InductorMetaTy, filename: str, configs_hash: str
    ) -> Optional[AutotuneCache]:
        cache = AutotuneCache(configs_hash)
        key = AutotuneCache._prepare_key(filename)
        cache._setup_local_cache(inductor_meta, os.path.dirname(filename), key)
        cache._setup_remote_autotune_cache(inductor_meta, key)
        if cache.local_cache or cache.remote_cache:
            return cache
        else:
            return None

    @staticmethod
    def _prepare_key(filename: str) -> str:
        from torch.compiler import config as cconfig

        # base of filename is already sha256 hash the source contents
        key = f"{os.path.basename(filename)}:{cconfig.cache_key_tag}"
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    # Read the best config options from the most local cache and return it.
    def _read(self) -> Optional[Dict[str, JsonDataTy]]:
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
        if best := self._read():
            return _load_cached_autotuning(
                best, self.configs_hash, configs, inductor_meta
            )
        return None

    # Set up local filesystem caching information
    def _setup_local_cache(
        self, inductor_meta: _InductorMetaTy, dirname: str, cache_key: str
    ) -> None:
        if not inductor_meta.get("autotune_local_cache", True):
            return

        cache_filename = f"{dirname}/{cache_key}.best_config"
        local_cache = LocalAutotuneCache()
        self.local_cache = (local_cache, cache_filename)

    # Set up remote caching information
    def _setup_remote_autotune_cache(
        self, inductor_meta: _InductorMetaTy, cache_key: str
    ) -> None:
        if not _should_use_remote_autotune_cache(inductor_meta):
            return

        if (backend_hash := inductor_meta.get("backend_hash", None)) is None:
            log.debug(
                "backend_hash is not passed on the inductor_meta, unable to use autotune remote cache"
            )
            return
        assert isinstance(backend_hash, str)

        is_fbcode = bool(inductor_meta.get("is_fbcode", False))

        salt = "autotune-best-config-v2"
        key = backend_hash + self.configs_hash + salt
        key = hashlib.sha256(key.encode("utf-8")).hexdigest()

        remote_cache = create_cache(
            key,
            is_fbcode,
            "FbRemoteAutotuneCache",
            "RemoteAutotuneCache",
        )
        if not remote_cache:
            return

        self.remote_cache = (remote_cache, cache_key)

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
            AutotuneCacheBundler.put(key, data)

            if log.isEnabledFor(logging.DEBUG):
                type_str = "coordesc" if found_by_coordesc else "heuristic"
                log.debug("Save %s tuning result to %s", type_str, key)

        if remote_cache := self.remote_cache:
            cache, key = remote_cache
            cache.put(key, data)


class _AutotuneCacheBundlerImpl:
    """
    Caches a set of LocalAutotuneCacheBackend entries together in a single
    cache.
    """

    _key: str
    _cache: RemoteCache[JsonDataTy]

    # All known entries from LocalAutotuneCache.put()
    _entries: Dict[str, JsonDataTy]

    def end_compile(self) -> None:
        # TODO: Do we need to compute time_taken_ms and encode that somehow?
        if self._entries:
            self._cache.put(self._key, self._entries)

    def put(self, basename: str, data: JsonDataTy) -> None:
        # Do we need to worry about duplicates? We only have a single local fs
        # entry - so probably not.
        self._entries[basename] = data

    def __init__(self, key: str, cache: RemoteCache[JsonDataTy]) -> None:
        self._key = key
        self._cache = cache
        self._entries = {}

    def sync(self) -> None:
        # We don't currently use this - but we could async load starting at
        # `begin_compile` and wait for the load to be finished here.
        pass

    @classmethod
    def _should_use_bundled_autotune_remote_cache(
        cls, inductor_meta: _InductorMetaTy
    ) -> bool:
        # The bundled autotune cache is only available if you've also got local
        # caching enabled (because we feed the bundled data to the local cache).
        if not inductor_meta.get("autotune_local_cache", True):
            return False

        # Check if the we're enabled via config
        if (
            bundled_autotune_remote_cache := inductor_meta.get(
                "bundled_autotune_remote_cache"
            )
        ) is not None:
            return bool(bundled_autotune_remote_cache)

        if not cls._get_is_fbcode(inductor_meta):
            return False
        if torch._utils_internal.is_fb_unit_test():
            return False
        if inductor_meta.get("is_hip"):
            return False

        try:
            from torch._inductor.fb.remote_cache import REMOTE_CACHE_VERSION
        except ModuleNotFoundError:
            return False

        jk = torch._utils_internal.justknobs_getval_int(
            "pytorch/remote_cache:bundled_autotune_remote_cache_version"
        )
        return REMOTE_CACHE_VERSION >= jk

    def _load_cache(self) -> bool:
        from torch._inductor import codecache

        # The single key is defined on construction of the cache.
        entries = self._cache.get(self._key)
        if entries is None or not isinstance(entries, dict):
            # We couldn't load the cache - so mark _entries as non-None so we
            # store local cache values.
            return False

        # Go through the entries we got from the cache and save them locally.
        time_saved_ns = 0
        for basename, data in entries.items():
            # Reconstruct the final filename (see put())
            root, ext = _splitext_nodot(basename)
            _, _, filename = codecache.get_path(root, ext)
            if isinstance(data, dict) and (tsns := data.get("time_saved_ns")):
                time_saved_ns += int(tsns)  # type: ignore[arg-type]
            local_cache = LocalAutotuneCache()
            local_cache.put(filename, data)

        codecache.add_ephemeral_timeout_increase_for_distributed(time_saved_ns)

        return True

    @staticmethod
    def _get_is_fbcode(inductor_meta: _InductorMetaTy) -> bool:
        return bool(inductor_meta.get("is_fbcode", False))

    @staticmethod
    def _get_backend_hash(inductor_meta: _InductorMetaTy) -> str:
        backend_hash = inductor_meta["backend_hash"]
        assert isinstance(backend_hash, str)
        return backend_hash


class AutotuneCacheBundler:
    _bundler: Optional[_AutotuneCacheBundlerImpl] = None

    def __init__(self) -> None:
        pass

    # Call this before we start any autotune computation for an inductor python
    # file. On a cache hit it copies the individual results into the local
    # autotune caches.
    @classmethod
    def begin_compile(
        cls,
        inductor_meta: _InductorMetaTy,
        *,
        code: Optional[str] = None,
        code_hash: Optional[str] = None,
    ) -> None:
        assert cls._bundler is None

        if code is not None:
            assert code_hash is None, "Cannot specify both code and code_hash"
            code_hash = _comment_stripped_hash(code)
        assert code_hash is not None

        if not _AutotuneCacheBundlerImpl._should_use_bundled_autotune_remote_cache(
            inductor_meta
        ):
            return

        cache = create_cache(
            "bundled-autotune-v1",
            _AutotuneCacheBundlerImpl._get_is_fbcode(inductor_meta),
            "FbRemoteBundledAutotuneCache",
            "RemoteBundledAutotuneCache",
        )
        if not cache:
            return

        # We're starting a compilation phase. We have a cache key for the code
        # we're compiling. We'll get the individual autotune bundles later (via
        # self.put()). For now create the AutotuneCacheBundler and try to load
        # from the cache.

        salt = "bundled-autotune-best-configs-v1"
        backend_hash = _AutotuneCacheBundlerImpl._get_backend_hash(inductor_meta)
        # TODO: The autotune cache includes configs_hash in the key. The problem
        # is that the configs_hash includes info from the individual pointwise()
        # calls (size_hints, for example) which we can't know yet. I *think*
        # that info is basically present in the `code_hash` (since it's a
        # parameter to the pointwise decorator) - but is there other info we
        # need to include from inductor_meta?
        key = code_hash + backend_hash + salt
        key = hashlib.sha256(key.encode("utf-8")).hexdigest()

        bundler = _AutotuneCacheBundlerImpl(key, cache)
        if not bundler._load_cache():
            # We couldn't load from the cache - so save the data so we can store
            # the saved autotunes.
            cls._bundler = bundler

        # If we get a cache hit don't bother saving any of the individual
        # autotune results.

    # Call this after all individual autotune results are finished for a
    # inductor python file. If we gathered any individual results then we bundle
    # those and put it into the cache.
    @classmethod
    def end_compile(cls) -> None:
        if bundler := cls._bundler:
            cls._bundler = None
            bundler.end_compile()

    @classmethod
    def sync(cls) -> None:
        if bundler := cls._bundler:
            bundler.sync()

    @classmethod
    def put(cls, filename: str, data: JsonDataTy) -> None:
        if bundler := cls._bundler:
            # The filename comes in as something like
            # "/tmp/tmp{random}/{aa}/{basename}.py" (where aa is
            # basename[1:3]). Strip it down and make sure that it looks like a path
            # we could reconstruct (because it's possible for the caller to
            # customize the path).
            basename = os.path.basename(filename)

            # TODO: check cache_dir() vs filename, then strip dirname
            bundler.put(basename, data)


# Remove the comments from the code (which include things like run ids and file
# paths) and then hash the result.
def _comment_stripped_hash(code: str) -> str:
    code = re.sub(r"#.*$", "", code, count=0, flags=re.MULTILINE)
    return torch._inductor.codecache.code_hash(code)


def _should_use_remote_autotune_cache(inductor_meta: _InductorMetaTy) -> bool:
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
    inductor_meta: _InductorMetaTy,
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


class _LocalAutotuneCacheBackend(RemoteCacheBackend[bytes]):
    @override
    def _get(self, key: str) -> Optional[bytes]:
        try:
            with open(key, "rb") as fd:
                return fd.read()
        except FileNotFoundError:
            return None

    @override
    def _put(self, key: str, data: bytes) -> None:
        os.makedirs(os.path.dirname(key), exist_ok=True)
        with open(key, "wb") as fd:
            fd.write(data)


class LocalAutotuneCache(RemoteCache[JsonDataTy]):
    def __init__(self) -> None:
        backend = _LocalAutotuneCacheBackend()
        serde = RemoteCacheJsonSerde()
        super().__init__(backend, serde)

    @override
    def _get(self, key: str, sample: Optional[Sample]) -> Optional[JsonDataTy]:
        AutotuneCacheBundler.sync()
        result = super()._get(key, sample)
        if result is not None:
            # What? Why are we doing a put() here? Imagine we have a new model
            # that reuses some existing kernels that have already been
            # compiled. If we didn't do a `put` here (on cache hit) then the new
            # model would only bundle *newly* compiled kernels, not existing
            # kernels that were already compiled and cached.
            AutotuneCacheBundler.put(key, result)
        return result

    @override
    def _put(self, key: str, value: JsonDataTy, sample: Optional[Sample]) -> None:
        AutotuneCacheBundler.put(key, value)
        super()._put(key, value, sample)


def _splitext_nodot(basename: str) -> tuple[str, str]:
    root, ext = os.path.splitext(basename)
    if ext:
        ext = ext[1:]
    return root, ext
