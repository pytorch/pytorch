"""
PyTorch Inductor Autotuning Cache System

This module implements a caching system for autotuning configurations in PyTorch's Inductor compiler.
It provides mechanisms to store and retrieve optimal kernel configurations both locally and remotely,
which significantly speeds up compilation by reusing previously discovered optimal parameters.

The caching system includes:
- Local filesystem caching for individual machine reuse
- Remote caching for sharing optimizations across machines
- Bundled caching to efficiently store multiple related configurations
- Cache invalidation based on PyTorch versions and backend changes
- Serialization/deserialization support for worker processes

Key components:
- AutotuneCache: Main class for managing cache access and storage
- AutotuneCacheBundler: Bundles multiple cache entries for efficient storage
- LocalAutotuneCache: Handles filesystem-based caching
- _LocalAutotuneCacheBackend: Low-level file operations for cache storage
- AutotuneCacheArtifact: Integration with PyTorch's artifact system

This caching system is critical for performance as it eliminates the need to re-run
expensive autotuning operations when the same kernels are compiled multiple times.
"""

from __future__ import annotations

import dataclasses
import hashlib
import logging
import os
import os.path
import re
from typing import Any, TYPE_CHECKING
from typing_extensions import override

import torch
from torch._inductor.runtime.runtime_utils import cache_dir
from torch.compiler._cache import (
    CacheArtifact,
    CacheArtifactFactory,
    CacheArtifactManager,
)
from torch.utils._triton import has_triton
from ..remote_cache import (
    create_cache,
    JsonDataTy,
    RemoteCache,
    RemoteCacheBackend,
    RemoteCacheJsonSerde,
)
from .triton_compat import Config, HAS_WARP_SPEC


if TYPE_CHECKING:
    from ..remote_cache import Sample

log = logging.getLogger(__name__)


_InductorMetaTy = dict[str, object]


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


@CacheArtifactFactory.register
class AutotuneCacheArtifact(CacheArtifact):
    @override
    def populate_cache(self) -> None:
        autotune_cache = _LocalAutotuneCacheBackend()
        key = os.path.join(cache_dir(), self.key)
        autotune_cache._put(key, self.content)

    @override
    @staticmethod
    def type() -> str:
        return "autotune"

    @override
    @staticmethod
    def encode(content: JsonDataTy) -> bytes:
        assert not isinstance(content, bytes)
        serde = RemoteCacheJsonSerde()
        content_bytes = serde.encode(content)
        assert isinstance(content_bytes, bytes)
        return content_bytes


@dataclasses.dataclass
class AutotuneCache:
    configs_hash: str
    local_cache: tuple[RemoteCache[JsonDataTy], str] | None = None
    remote_cache: tuple[RemoteCache[JsonDataTy], str] | None = None

    # Create a AutotuneCache. Returns None if none of the caches can be used.
    @staticmethod
    def create(
        inductor_meta: _InductorMetaTy, filename: str, configs_hash: str
    ) -> AutotuneCache | None:
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
    def _read(self) -> dict[str, JsonDataTy] | None:
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
        self, inductor_meta: _InductorMetaTy, configs: list[Config]
    ) -> Config | None:
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

        from ..codecache import torch_key

        """
        [Note: torch_key in autotune cache key]
        Include torch_key() in the cache key so that different versions
        of torch result in cache invalidation. This is important in case
        of changes to the best_config format or other code changes that
        are not backward compatible w.r.t. the cache.
        """
        hasher = hashlib.sha256()
        hasher.update(cache_key.encode("utf-8"))
        hasher.update(torch_key())
        updated_cache_key = hasher.hexdigest()

        cache_filename = f"{dirname}/{updated_cache_key}.best_config"
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

        from ..codecache import torch_key

        is_fbcode = bool(inductor_meta.get("is_fbcode", False))

        salt = "autotune-best-config-v2"
        # re: torch_key - see [Note: torch_key in autotune cache key]
        key = torch_key().hex() + backend_hash + self.configs_hash + salt
        key = hashlib.sha256(key.encode("utf-8")).hexdigest()

        remote_cache = create_cache(
            key,
            is_fbcode,
            "FbRemoteAutotuneCache",
            "RemoteAutotuneCache",
        )
        if not remote_cache:
            return

        # Save the args passed to create_cache
        # in case AutotuneCache needs to be pickled
        self.remote_cache_full_key = key
        self.is_fbcode = is_fbcode
        self.remote_cache = (remote_cache, cache_key)

    # The AutotuneCache may be serialized/deserialized if we're using
    # AsyncCompile worker processes to run triton compilation.
    # This is because AutotuneCache instances are created on the worker
    # process, but we need to run AutotuneCache.save on the parent process
    # when actually doing autotuning.
    def __getstate__(self) -> dict[str, Any]:
        # The remote cache handles themselves may not be serializable
        # So clear it and reconstruct it on setstate
        remote_cache = getattr(self, "remote_cache", None)
        return {
            **self.__dict__,
            # Save the cache_key portion
            "remote_cache": remote_cache and remote_cache[1],
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        # Reconstruct the remote cache on the parent class
        self.__dict__.update(state)
        if self.remote_cache is not None:
            assert isinstance(self.remote_cache, str)
            assert hasattr(self, "remote_cache_full_key")
            assert hasattr(self, "is_fbcode")
            cache_key = self.remote_cache
            remote_cache = create_cache(
                self.remote_cache_full_key,
                self.is_fbcode,
                "FbRemoteAutotuneCache",
                "RemoteAutotuneCache",
            )
            if remote_cache is not None:
                self.remote_cache = (remote_cache, cache_key)
            else:
                log.warning("Warning, failed to recreate remote cache after pickling")
                self.remote_cache = None

    # Save the config in the caches
    def save(
        self,
        config: Config,
        time_taken_ns: int,
        found_by_coordesc: bool = False,
        triton_cache_hash: str | None = None,
    ) -> None:
        data = {
            # pyrefly: ignore [missing-attribute]
            **config.kwargs,
            # pyrefly: ignore [missing-attribute]
            "num_warps": config.num_warps,
            # pyrefly: ignore [missing-attribute]
            "num_stages": config.num_stages,
            "configs_hash": self.configs_hash,
            "found_by_coordesc": found_by_coordesc,
            "time_taken_ms": time_taken_ns // 1000000,  # Convert from NS to MS
            "triton_cache_hash": triton_cache_hash,
        }
        if HAS_WARP_SPEC:
            data.update(
                {
                    "num_consumer_groups": getattr(config, "num_consumer_groups", 0),
                    "num_buffers_warp_spec": getattr(
                        config, "num_buffers_warp_spec", 0
                    ),
                }
            )

        if local_cache := self.local_cache:
            cache, key = local_cache
            cache.put(key, data)
            AutotuneCacheBundler.put(key, data)
            autotune_artifact_key = os.path.join(*key.split(os.sep)[-2:])
            CacheArtifactManager.record_artifact(
                AutotuneCacheArtifact.type(), autotune_artifact_key, data
            )

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
    _entries: dict[str, JsonDataTy]

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
    _bundler: _AutotuneCacheBundlerImpl | None = None

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
        code: str | None = None,
        code_hash: str | None = None,
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
    best_config: dict[str, JsonDataTy],
    configs_hash: str,
    configs: list[Config],
    inductor_meta: _InductorMetaTy,
) -> Config | None:
    if best_config is None:
        return None
    if best_config.pop("configs_hash", None) != configs_hash:
        return None

    # Remove time taken for comparison
    best_config.pop("time_taken_ms", None)

    best_config.pop("triton_cache_hash", None)

    if inductor_meta.get("coordinate_descent_tuning") and best_config.pop(
        "found_by_coordesc", False
    ):
        num_warps = best_config.pop("num_warps")
        num_stages = best_config.pop("num_stages")

        # Extract common arguments
        config_args = {
            "num_warps": num_warps,
            "num_stages": num_stages,
        }

        if HAS_WARP_SPEC:
            config_args.update(
                {
                    "num_consumer_groups": best_config.pop("num_consumer_groups", 0),
                    "num_buffers_warp_spec": best_config.pop(
                        "num_buffers_warp_spec", 0
                    ),
                }
            )

        # Create the triton_config with the appropriate arguments
        # pyrefly: ignore [bad-argument-count]
        triton_config = Config(best_config, **config_args)
        # pyrefly: ignore [missing-attribute]
        triton_config.found_by_coordesc = True
        return triton_config

    matching_configs = [
        cfg
        for cfg in configs
        # pyrefly: ignore [missing-attribute]
        if all(val == best_config.get(key) for key, val in cfg.kwargs.items())
        # pyrefly: ignore [missing-attribute]
        and cfg.num_warps == best_config.get("num_warps")
        # pyrefly: ignore [missing-attribute]
        and cfg.num_stages == best_config.get("num_stages")
    ]
    if len(matching_configs) != 1:
        return None

    return matching_configs[0]


class _LocalAutotuneCacheBackend(RemoteCacheBackend[bytes]):
    @override
    def _get(self, key: str) -> bytes | None:
        try:
            with open(key, "rb") as fd:
                return fd.read()
        except FileNotFoundError:
            return None

    @override
    def _put(self, key: str, data: bytes) -> None:
        os.makedirs(os.path.dirname(key), exist_ok=True)
        from torch._inductor import codecache

        codecache.write_atomic(key, data)


class LocalAutotuneCache(RemoteCache[JsonDataTy]):
    def __init__(self) -> None:
        backend = _LocalAutotuneCacheBackend()
        serde = RemoteCacheJsonSerde()
        super().__init__(backend, serde)

    @override
    def _get(self, key: str, sample: Sample | None) -> JsonDataTy | None:
        AutotuneCacheBundler.sync()
        result = super()._get(key, sample)
        if result is not None:
            assert isinstance(result, dict)
            # What? Why are we doing a put() here? Imagine we have a new model
            # that reuses some existing kernels that have already been
            # compiled. If we didn't do a `put` here (on cache hit) then the new
            # model would only bundle *newly* compiled kernels, not existing
            # kernels that were already compiled and cached.
            AutotuneCacheBundler.put(key, result)
            autotune_artifact_key = os.path.join(*key.split(os.sep)[-2:])
            CacheArtifactManager.record_artifact(
                AutotuneCacheArtifact.type(), autotune_artifact_key, result
            )
        return result

    @override
    def _put(self, key: str, value: JsonDataTy, sample: Sample | None) -> None:
        AutotuneCacheBundler.put(key, value)
        super()._put(key, value, sample)


def _splitext_nodot(basename: str) -> tuple[str, str]:
    root, ext = os.path.splitext(basename)
    if ext:
        ext = ext[1:]
    return root, ext
