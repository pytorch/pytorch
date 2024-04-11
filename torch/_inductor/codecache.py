from __future__ import annotations

import base64
import copyreg
import dataclasses
import functools
import hashlib
import importlib
import io
import json
import logging
import multiprocessing
import os
import pathlib
import pickle
import pkgutil
import platform
import re
import shlex
import shutil
import signal
import subprocess
import sys
import sysconfig
import tempfile
import textwrap
import threading
import warnings
from bisect import bisect_right
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from copy import copy
from ctypes import c_void_p, cdll, CDLL
from functools import partial
from pathlib import Path
from threading import Thread
from time import sleep, time
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TYPE_CHECKING, Union

import torch

from torch._dynamo.device_interface import (
    get_interface_for_device,
    get_registered_device_interfaces,
)
from torch._dynamo.utils import counters, dynamo_timed
from torch._inductor import config, exc, metrics
from torch._inductor.codegen.cuda import cuda_env
from torch._inductor.utils import cache_dir, developer_warning, is_linux
from torch._subclasses.fake_tensor import (
    extract_tensor_metadata,
    FakeTensor,
    TensorMetadata,
)
from torch.fx.experimental.symbolic_shapes import has_hint, hint_int, ShapeEnv

if TYPE_CHECKING:
    from torch._inductor.graph import GraphLowering
    from torch._inductor.ir import ChoiceCaller

from torch.hub import _Faketqdm, tqdm

_HERE = os.path.abspath(__file__)
_TORCH_PATH = os.path.dirname(os.path.dirname(_HERE))
_LINKER_SCRIPT = os.path.join(_TORCH_PATH, "_inductor/script.ld")

if config.is_fbcode():
    from triton.fb import build_paths
    from triton.fb.build import _run_build_command

    from torch._inductor.fb.utils import (
        log_global_cache_errors,
        log_global_cache_stats,
        log_global_cache_vals,
        use_global_cache,
    )
else:

    def log_global_cache_errors(*args, **kwargs):
        pass

    def log_global_cache_stats(*args, **kwargs):
        pass

    def log_global_cache_vals(*args, **kwargs):
        pass

    def use_global_cache() -> bool:
        return False


output_code_log = torch._logging.getArtifactLogger(__name__, "output_code")

LOCK_TIMEOUT = 600

# timing metrics for time spent in the compilation
_cumulative_compile_time = 0.0
_t0: Optional[float] = None


def _compile_start() -> None:
    global _t0
    if _t0 is None:
        _t0 = time()


def _compile_end() -> None:
    global _cumulative_compile_time, _t0
    if _t0 is not None:
        t1 = time()
        _cumulative_compile_time += t1 - _t0
        _t0 = None
        # print("CUMULATIVE COMPILE TIME", _cumulative_compile_time)


log = logging.getLogger(__name__)


def cpp_wrapper_cache_dir(name: str) -> str:
    cu_str = (
        "cpu"
        if torch.version.cuda is None
        else f'cu{torch.version.cuda.replace(".", "")}'
    )
    python_version = f"py{sys.version_info.major}{sys.version_info.minor}"
    build_folder = f"{python_version}_{cu_str}"

    cpp_wrapper_dir = os.path.join(cache_dir(), build_folder)
    cpp_wrapper_build_directory = os.path.join(cpp_wrapper_dir, name)
    os.makedirs(cpp_wrapper_build_directory, exist_ok=True)
    return cpp_wrapper_build_directory


def get_cpp_wrapper_cubin_path_name():
    return "cubin_path" if torch.version.hip is None else "hsaco_path"


class CacheBase:
    @staticmethod
    @functools.lru_cache(None)
    def get_system() -> Dict[str, Any]:
        try:
            import triton

            triton_version = triton.__version__
        except ModuleNotFoundError:
            triton_version = None

        try:
            system: Dict[str, Any] = {
                "device": {
                    "name": torch.cuda.get_device_properties(
                        torch.cuda.current_device()
                    ).name,
                },
                "version": {
                    "cuda": torch.version.cuda,
                    "triton": triton_version,
                },
            }
        except (AssertionError, RuntimeError):
            # If cuda is not installed, none of the above config is relevant.
            system = {}

        system["hash"] = hashlib.sha256(
            json.dumps(system, sort_keys=True).encode("utf-8")
        ).hexdigest()

        return system

    @staticmethod
    @functools.lru_cache(None)
    def get_local_cache_path() -> Path:
        return Path(os.path.join(cache_dir(), "cache", CacheBase.get_system()["hash"]))

    @staticmethod
    @functools.lru_cache(None)
    def get_global_cache_path() -> Optional[Path]:
        return (
            Path(os.path.join(config.global_cache_dir, CacheBase.get_system()["hash"]))
            if config.global_cache_dir is not None
            else None
        )

    def __init__(self) -> None:
        if not torch.cuda.is_available():
            return

        self.system = CacheBase.get_system()

        self.local_cache_path = CacheBase.get_local_cache_path()
        self.global_cache_path = CacheBase.get_global_cache_path()

    def get_local_cache(self) -> Dict[str, Any]:
        if not self.local_cache_path.is_file():
            return {}
        with open(self.local_cache_path) as local_cache_fp:
            local_cache = json.load(local_cache_fp)
        return local_cache["cache"]

    def update_local_cache(self, local_cache: Dict[str, Any]) -> None:
        if not os.path.exists(self.local_cache_path.parent):
            os.makedirs(self.local_cache_path.parent, exist_ok=True)

        write_atomic(
            str(self.local_cache_path),
            json.dumps({"system": self.system, "cache": local_cache}, indent=4),
        )


class LocalCache(CacheBase):
    def lookup(self, *keys: str) -> Optional[Dict[str, Any]]:
        cache = self.get_local_cache()

        sub_cache = cache
        for key in keys:
            if key in cache:
                sub_cache = cache[key]
            else:
                return None

        return sub_cache

    def set_value(self, *keys: str, value: Any) -> None:
        cache = self.get_local_cache()

        sub_cache = cache
        for key in keys[0:-1]:
            sub_cache.setdefault(key, {})
            sub_cache = sub_cache[key]
        sub_cache[keys[-1]] = value

        self.update_local_cache(cache)


class PersistentCache(CacheBase):
    @functools.lru_cache(None)
    def get_global_cache(self):
        if self.global_cache_path is None or not self.global_cache_path.is_file():
            return {}
        with open(self.global_cache_path) as global_cache_fp:
            global_cache = json.load(global_cache_fp)
        return global_cache["cache"]

    def lookup(
        self,
        choices: List[ChoiceCaller],
        op: str,
        inputs: str,
        benchmark: Callable[[Any], Dict[ChoiceCaller, float]],
    ) -> Dict[ChoiceCaller, float]:
        """
        Check to see if we have benchmarked the given choice callers. For each
        choice caller:

            1. Check global_cache[op][inputs][choice][precision], return benchmark if cached.
            2. Check local_cache[op][inputs][choice][precision], return benchmark if cached.
            3.
                a. `max_autotune_gemm=True`: benchmark the choice, update
                    local_cache[op][inputs][choice], and return the benchmark.
                b. `max_autotune_gemm=False`: don't benchmark the choice, return nothing.
        """
        precision = torch.get_float32_matmul_precision()

        log_stats = partial(log_global_cache_stats, self.system, op, inputs, precision)
        log_vals = partial(log_global_cache_vals, self.system, op, inputs, precision)
        log_errors = partial(
            log_global_cache_errors, self.system, op, inputs, precision
        )
        timings = {}

        def check_cache(cache, callback=None) -> bool:
            """Check if `cache` contains data for all the choices"""
            hit = True
            for choice in choices:
                choice_hash = choice.hash_key()
                if choice_hash in cache.get(op, {}).get(inputs, {}).get(precision, {}):
                    # cache hit
                    timings[choice] = cache[op][inputs][precision][choice_hash]
                else:
                    # cache miss
                    hit = False
                    break
            if callback:
                callback(cached=hit)
            return hit

        if config.max_autotune or config.max_autotune_gemm:
            local_cache = self.get_local_cache()
            # check local cache first since it is data specific to the current machine
            if not check_cache(local_cache) and not (
                use_global_cache()
                and check_cache(self.get_global_cache(), callback=log_stats)
            ):
                try:
                    # re-benchmark everything to try to get consistent numbers from the same machine
                    timings = benchmark(choices)
                    assert all(choice in timings for choice in choices)
                    local_cache.setdefault(op, {})
                    local_cache[op].setdefault(inputs, {}).setdefault(precision, {})
                    for choice, timing in timings.items():
                        local_cache[op][inputs][precision][choice.hash_key()] = timing
                except RuntimeError as e:
                    # catch and log autotuning failures
                    log_errors(e)
                    raise e

                self.update_local_cache(local_cache)

                timings_to_log = {
                    choice.hash_key(): timings[choice] for choice in choices
                }
                log_vals(timings_to_log)
        elif use_global_cache():
            # only check global cache, not local one
            check_cache(self.get_global_cache(), callback=log_stats)
            # may have a partial cache hit, where not everything is benchmarked

        return timings


def get_lock_dir() -> str:
    lock_dir = os.path.join(cache_dir(), "locks")
    if not os.path.exists(lock_dir):
        os.makedirs(lock_dir, exist_ok=True)
    return lock_dir


def sha256_hash(data: bytes) -> str:
    # [:51] to strip off the "Q====" suffix common to every hash value.
    return base64.b32encode(hashlib.sha256(data).digest())[:51].decode("utf-8").lower()


def code_hash(code: Union[str, bytes], extra: str = ""):
    hashing_str = code if isinstance(code, bytes) else code.encode("utf-8")
    if extra != "":
        hashing_str = hashing_str + b"||" + extra.encode("utf-8")
    return "c" + sha256_hash(hashing_str)


def get_path(
    basename: str, extension: str, specified_dir: str = ""
) -> Tuple[str, str, str]:
    if specified_dir:
        if os.path.isabs(specified_dir):
            subdir = specified_dir
        else:
            subdir = os.path.join(cache_dir(), specified_dir)
    else:
        subdir = os.path.join(cache_dir(), basename[1:3])
    path = os.path.join(subdir, f"{basename}.{extension}")
    return basename, subdir, path


def get_hash(content: Union[str, bytes], extra: str = "", hash_type: str = "code"):
    if hash_type == "code":
        return code_hash(content, extra)
    if hash_type in ["cubin", "hsaco"]:
        return code_hash(repr(content))
    raise AssertionError(f"Unknown hash type {hash_type}")


def write(
    content: Union[str, bytes],
    extension: str,
    extra: str = "",
    hash_type: str = "code",
    specified_dir: str = "",
) -> Tuple[str, str]:
    # use striped content to compute hash so we don't end up with different
    # hashes just because the content begins/ends with differnet number of
    # spaces.
    key: str = get_hash(content.strip(), extra, hash_type)
    basename, subdir, path = get_path(key, extension, specified_dir)
    if not os.path.exists(subdir):
        os.makedirs(subdir, exist_ok=True)
    if not os.path.exists(path):
        write_atomic(path, content)
    return basename, path


def write_atomic(path: str, content: Union[str, bytes]) -> None:
    # Write into temporary file first to avoid conflicts between threads
    # Avoid using a named temporary file, as those have restricted permissions
    assert isinstance(
        content, (str, bytes)
    ), "Only strings and byte arrays can be saved in the cache"
    path = pathlib.Path(path)
    tmp_path = path.parent / f".{os.getpid()}.{threading.get_ident()}.tmp"
    write_mode = "w" if isinstance(content, str) else "wb"
    with tmp_path.open(write_mode) as f:
        f.write(content)
    tmp_path.rename(path)


@dataclasses.dataclass
class TensorMetadataAndValues:
    """
    TensorMetadata plus the elements as a list of raw values.
    Used for hashing inlined constants.
    """

    tensor_metadata: TensorMetadata
    values: List[Any]


def _ident(x: Any) -> Any:
    return x


def _reduce_fake_tensor(t):
    """
    See FxGraphCachePickler. Custom reducer to pickle FakeTensors.
    """
    metadata = extract_tensor_metadata(t)
    return (_ident, (metadata,))


def _reduce_tensor(t):
    """
    See FxGraphCachePickler. Custom reducer to pickle Tensors.
    If we see tensors, we know they're constants stored as attributes on
    the GraphModule. Include the values in the key calculation. Small
    tensors will be inlined, so we can't serve the same cache entry for
    different values anyway. Large constants are treated as parameters,
    so we could conceivably reuse a cache entry. To do that, however,
    PyCodeCache would need more complexity to create a new module from its
    cache, but with the right constants attached as attributes.
    """
    if t.is_mkldnn:
        # TODO: These tensors don't currently pickle, so we can't cache a
        # compiled graph containing them. Just fail now. If mkldnn tensors
        # get pickling support, we can remove this.
        raise BypassFxGraphCache()

    # Very large tensors could be expensive to copy to cpu and hash. Let's
    # at least report if we find slowness.
    start = time()
    values = t.tolist()
    elapsed = time() - start
    if elapsed > 1.0:
        warnings.warn(
            f"FX graph cache handling of a large constant took {elapsed:.1}s. Please file an issue."
        )

    metadata = extract_tensor_metadata(t)
    return (_ident, (TensorMetadataAndValues(metadata, values),))


def _reduce_symint(s):
    """
    See FxGraphCachePickler. Custom reducer to pickle SymInts.
    """
    # For hashing purposes, we only care about the name of the symbol and
    # not the backed value. We evaluate guards stored with a cached graph
    # to ensure a cached entity with SymInt args is safe to reuse.
    return (_ident, (str(s),))


class FxGraphCachePickler(pickle.Pickler):
    """
    Custom pickler to customize the pickling of some objects (Tensors), only for the
    purpose of computing a hash for keying into the FxGraphCache. Tensors contain
    objects that don't pickle and/or vary between runs, and we want to capture the
    data that allow us to compute a stable, but safe hash.
    """

    dispatch_table = copyreg.dispatch_table.copy()
    dispatch_table[FakeTensor] = _reduce_fake_tensor
    dispatch_table[torch.Tensor] = _reduce_tensor
    dispatch_table[torch.SymInt] = _reduce_symint

    @staticmethod
    def dumps(obj) -> bytes:
        """
        Pickle an object using the FxGraphCachePickler.
        """
        with io.BytesIO() as stream:
            pickler = FxGraphCachePickler(stream)
            pickler.dump(obj)
            return stream.getvalue()

    @staticmethod
    def get_hash(obj: Any) -> str:
        """
        Serialize an object using the FxGraphCachePickler and return a hash
        of the pickled object.
        """
        serialized_data = FxGraphCachePickler.dumps(obj)
        return sha256_hash(serialized_data)


@functools.lru_cache(None)
def get_inductor_code_hash() -> bytes:
    """
    Compute a hash of all inductor code modules. Used by the FxGraph cache
    so any inductor code changes would result in new cache keys.
    """
    inductor_root = os.path.dirname(__file__)

    contents: Dict[str, bytes] = {}
    for lib in pkgutil.iter_modules([inductor_root]):
        spec = lib.module_finder.find_spec(lib.name, None)
        assert spec is not None
        module = spec.origin
        assert module is not None
        with open(module, "rb") as f:
            contents[module] = f.read()

    return hashlib.sha256(pickle.dumps(contents)).digest()


@dataclasses.dataclass
class OrderedSetHolder:
    """
    See FxGraphHashDetails. Holds a sorted list to support stable hashing
    of set kwargs.
    """

    items: List[Any]


class BypassFxGraphCache(Exception):
    """
    Exception to indicate that the FxGraphCache should be bypassed.
    """

    pass


class FxGraphHashDetails:
    """
    Object to capture all the details for a compiled FX graph relevant to computing
    a safe and stable cache key.
    """

    # Excluded kwargs param that are not stable between runs
    EXCLUDED_KWARGS = ["graph_id"]

    def __init__(
        self,
        gm: torch.fx.GraphModule,
        example_inputs: List[torch.Tensor],
        fx_kwargs: Dict[str, Any],
    ):
        self.gm = gm
        self.example_inputs = example_inputs

        # Order kwargs so hashing is stable to changes in kwarg order.
        self.fx_kwargs = {}
        for k in sorted(fx_kwargs):
            if k not in self.EXCLUDED_KWARGS:
                if type(fx_kwargs[k]) is set:
                    # Special case to handle set params. Python sets can't be
                    # ordered, so sort the elements and store them in a proxy.
                    self.fx_kwargs[k] = OrderedSetHolder(sorted(fx_kwargs[k]))
                else:
                    self.fx_kwargs[k] = fx_kwargs[k]

        # 'Deterministic algorithms' can affect codegen via lowering to cuda kernels.
        self.deterministic_algorithms_settings = (
            torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled(),
            torch.utils.deterministic.fill_uninitialized_memory,  # type: ignore[attr-defined]
        )

        # Global settings affecting matmul codegen.
        self.cuda_matmul_settings = (
            torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        )

        # Also hash on various system info (including the triton compiler version).
        self.torch_version = torch.__version__
        self.system_info = CacheBase.get_system()

        # And the inductor configuration and code.
        self.inductor_code_hash = get_inductor_code_hash()
        try:
            self.inductor_config = config.save_config()
        except (TypeError, AttributeError) as e:
            # Some configs options are callables, e.g., post_grad_custom_pre_pass,
            # and may not pickle.
            log.debug("Can't pickle inductor config: %s", e)
            raise BypassFxGraphCache() from e

    def debug_str(self) -> str:
        """
        Get a printable string describing in more detail all the attributes
        comprising this object. Useful for debugging when one graph hashes
        to a different value than another.
        """

        def get_str(obj) -> str:
            if isinstance(obj, torch.Tensor):
                return str(extract_tensor_metadata(obj))
            elif isinstance(obj, bytes):
                return "<bytes>"
            else:
                return str(obj)

        lines = []
        for attr, obj in vars(self).items():
            if isinstance(obj, list):
                for ii in range(len(obj)):
                    h = FxGraphCachePickler.get_hash(obj[ii])
                    lines.append(f"[{h}] {attr}[{ii}]: {get_str(obj[ii])}")
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    h = FxGraphCachePickler.get_hash(v)
                    lines.append(f"[{h}] {attr}[{k}]: {get_str(v)}")
            else:
                h = FxGraphCachePickler.get_hash(obj)
                lines.append(f"[{h}] {attr}: {get_str(obj)}")
        return "\n".join(lines)


def compiled_fx_graph_hash(
    gm: torch.fx.GraphModule,
    example_inputs: List[torch.Tensor],
    fx_kwargs: Dict[str, Any],
) -> str:
    """
    Generate a unique hash of the FX graph for caching.
    """
    details = FxGraphHashDetails(gm, example_inputs, fx_kwargs)
    # The prefix distinguishes among the other kinds of objects we
    # cache in this module.
    key = "f" + FxGraphCachePickler.get_hash(details)
    log.debug("FX graph cache hash details for key %s:\n%s", key, details.debug_str())
    return key


class FxGraphCache:
    """
    Supports caching and reusing compiled Fx graphs.

    The overall strategy is as follows:
    - This cache stores entries on disk. When saving an entry, we can't
      serialize callables (that could be C++, Triton, etc.), so we serialize
      their own disk cache location. We then recreate the compiled artifact
      after fetching from disk.
    - For indexing the cache, we gather the fields relevant to identifying an
      FxGraph (the graph module, graph inputs, system settings etc.) into an
      FxGraphCacheDetails object, pickle it, and compute a hash for the key.
      See FxGraphCachePickler.
    - Among the metadata we store, we also include a guards expression that's
      appropriate for validating any symbols for Tensor arguments that have
      symbolic bounds. On cache lookup then, we evaluate those guards in the
      current context to validate that a cached entry can be served.
    - A given graph could have multiple compiled versions, corresponding to
      different sets of guards. Therefore, we store cache entries in the form:
          <temp dir>/<fx graph hash>/<serialized metatdata>
    - On lookup, we compute the key from the graph details, iterate over all
      leaf files in the corresponding subdirectory, deserialize the entry, and
      evaluate its guards expression. If the evaluation succeeds, we have a
      cache hit. If it fails, we compile the graph and store a new entry.
    - Finally, on a cache hit, we need to make sure any guards that would
      have been created during compilation are added to the current context.
    """

    # TODO(masnesral): Investigate whether it's beneficial to store compiled graphs
    # in an in-memory cache after loading from disk.
    @staticmethod
    def _get_tmp_dir() -> str:
        """
        Get the toplevel temporary directory for storing compiled graphs.
        """
        return os.path.join(cache_dir(), "fxgraph")

    @staticmethod
    def _get_tmp_dir_for_key(key: str) -> str:
        """
        Return the disk location for a given cache key.
        """
        return os.path.join(FxGraphCache._get_tmp_dir(), key[1:3], key)

    @staticmethod
    def _filter_symints(inputs: List[Any]) -> List[torch.SymInt]:
        """
        Get the SymInt objects from the input list.
        """
        return [s for s in inputs if isinstance(s, torch.SymInt)]

    @staticmethod
    def _get_shape_env() -> Optional[ShapeEnv]:
        """
        Helper to get the shape env from the tracing context.
        """
        ctx = torch._guards.TracingContext.try_get()
        if not ctx:
            return None
        return ctx.fake_mode.shape_env

    @staticmethod
    def _lookup_graph(
        key: str,
        example_inputs: List[torch.Tensor],
    ) -> Optional[CompiledFxGraph]:
        """
        Lookup a compiled graph in the cache by key. On a hit, return the
        deserialized CompiledFxGraph object. On a miss, return None.
        """
        subdir = FxGraphCache._get_tmp_dir_for_key(key)
        if not os.path.exists(subdir):
            return None

        shape_env = FxGraphCache._get_shape_env()
        assert shape_env is not None

        symints = FxGraphCache._filter_symints(example_inputs)
        assert all(has_hint(s) for s in symints)
        hints = [hint_int(s) for s in symints]

        # Iterate over any entries in the subdir for this key and evaluate
        # their guards to determine whether there's a hit.
        graph = None

        for path in sorted(os.listdir(subdir)):
            with open(os.path.join(subdir, path), "rb") as f:
                candidate: CompiledFxGraph = pickle.load(f)

            if not candidate.guards_expr:
                # No guards to evaluate, so this is a hit.
                graph = candidate
                break

            # Evaluate the guard expression in the current context.
            # If there's not a cache hit, we don't want the evaluation to
            # affect the current env, e.g., cause the creation of new guards,
            # so we evaluate with the hints instead of the symbols.
            hit = bool(
                shape_env.evaluate_guards_expression(candidate.guards_expr, hints)
            )
            log.debug(
                "fx graph cache key %s evaluating guards [%s] with values %s => hit=%s",
                key,
                candidate.guards_expr,
                hints,
                hit,
            )
            if hit:
                graph = candidate
                break

        if graph is None:
            return None

        # See _save_graph(); we don't store the callable in the cache entry so
        # recreate it here from the PyCodeCache disk cache.
        try:
            graph.current_callable = PyCodeCache.load_by_key_path(
                graph.cache_key,
                graph.artifact_path,
                graph.cache_linemap,
                graph.constants,
            ).call
        except OSError:
            # Not expected, but in case the PyCodeCache entry is removed from
            # underneath us, treat it as a cache miss and recompile.
            log.error("Failed to load cached artifact: %s", graph.artifact_path)
            return None

        # Now re-evaluate with the symints to add any guards to the current env.
        if graph.guards_expr:
            check = bool(
                shape_env.evaluate_guards_expression(graph.guards_expr, symints)
            )
            assert check is True
            log.debug(
                "fx graph cache key %s post-load guards: %s", key, shape_env.guards
            )

        # Increment the cached metrics by the amounts recorded when the FX
        # graph was compiled for this cache entry. Pretending these counters
        # were incremented normally is useful for testing with the cache enabled.
        metrics.CachedMetricsHelper.apply_deltas(graph.metrics_deltas)

        return graph

    @staticmethod
    def _save_graph(
        key: str, compiled_graph: CompiledFxGraph, example_inputs: List[torch.Tensor]
    ):
        """
        Store a serialized CompiledFxGraph on disk.
        """
        disk_compiled_graph = copy(compiled_graph)
        # We can't really serialize callables that may be C++/Triton/etc.,
        # so we serialize their PyCodeCache disk cache location instead.
        # TODO: This could be better if we're ever able to serialize compiled
        # models to disk.
        disk_compiled_graph.current_callable = None

        # Before serializing, compute the guard expression that will be used to
        # ensure that a CompiledFxGraph is valid when loaded from the cache. It's
        # sufficient to consider only the SymInt args to the fx graph since the
        # Tensor shapes are already captured in the hash for the cache key. Any
        # Tensor arg with a symbolic shape will have a SymInt arg for the graph.
        shape_env = FxGraphCache._get_shape_env()
        assert shape_env is not None
        symints = FxGraphCache._filter_symints(example_inputs)
        disk_compiled_graph.guards_expr = shape_env.produce_guards_expression(symints)

        try:
            content = pickle.dumps(disk_compiled_graph)
        except Exception as e:
            log.debug("fx graph cache unable to serialize compiled graph: %s", e)
            counters["inductor"]["fxgraph_cache_pickle_error"] += 1
            return

        subdir = FxGraphCache._get_tmp_dir_for_key(key)
        if not os.path.exists(subdir):
            os.makedirs(subdir, exist_ok=True)

        # Use a hash of the serialized CompiledFxGraph to get a unique file
        # name. The specific name doesn't matter since a lookup involves
        # iterating over all entries in the parent subdir.
        path = os.path.join(subdir, sha256_hash(content))
        write_atomic(path, content)

    @staticmethod
    def _check_can_cache():
        """
        Check some conditions that would preclude caching and raise BypassFxGraphCache
        to bypass in case caching is not possible.
        """
        if config.freezing or config.aot_inductor.use_runtime_constant_folding:
            # Freezing can embed constants that wouldn't be static across runs.
            raise BypassFxGraphCache()

        if FxGraphCache._get_shape_env() is None:
            # The treatment of guards in the caching implementation requires that
            # we have a shape env.
            log.debug("fx graph cache no shape env")
            raise BypassFxGraphCache()

    @staticmethod
    def load(
        compile_fx_fn: Callable[..., Any],
        gm: torch.fx.GraphModule,
        example_inputs: List[torch.Tensor],
        fx_kwargs: Dict[str, Any],
    ):
        """
        Load a compiled graph from the cache. If a cached entry does not exist,
        compile the graph and save it to the cache.
        """
        from filelock import FileLock

        compiled_graph = None
        try:
            FxGraphCache._check_can_cache()
            key = compiled_fx_graph_hash(gm, example_inputs, fx_kwargs)

            lock_path = os.path.join(get_lock_dir(), key + ".lock")
            with FileLock(lock_path, timeout=LOCK_TIMEOUT):
                compiled_graph = FxGraphCache._lookup_graph(key, example_inputs)
                if compiled_graph is None:
                    log.debug("fx graph cache miss for key %s", key)
                    counters["inductor"]["fxgraph_cache_miss"] += 1
                    compiled_graph = compile_fx_fn(gm, example_inputs, **fx_kwargs)
                    FxGraphCache._save_graph(key, compiled_graph, example_inputs)
                else:
                    log.debug("fx graph cache hit for key %s", key)
                    counters["inductor"]["fxgraph_cache_hit"] += 1
        except BypassFxGraphCache:
            counters["inductor"]["fxgraph_cache_bypass"] += 1

        if not compiled_graph:
            compiled_graph = compile_fx_fn(gm, example_inputs, **fx_kwargs)

        return compiled_graph

    @staticmethod
    def clear():
        """
        Clear out the on-disk cache.
        """
        try:
            shutil.rmtree(FxGraphCache._get_tmp_dir())
        except FileNotFoundError:
            pass


@dataclasses.dataclass
class CompiledFxGraph:
    """
    Class holding a compiled FX graph. This is the object serialized on disk
    to support FxGraph caching.
    """

    current_callable: Optional[Callable[..., Any]]
    cache_key: str
    artifact_path: str
    cache_linemap: Optional[List[Tuple[int, str]]]
    device_types: Set[str]
    device_idxs: Set[int]
    mutated_inputs: Set[str]
    mutated_input_idxs: Set[int]
    constants: Dict[str, torch.Tensor]
    output_strides: Optional[List[Optional[Tuple[int, ...]]]]
    disabled_cudagraphs_reason: Optional[str]
    metrics_deltas: metrics.CachedMetricsDeltas
    # This is a string representation of an expression we serialize
    # with the object so the guards can be evaluated in a different
    # context in order to verify the validity of serving a cached
    # fx graph. The expression must be generated by:
    # ShapeEnv.produce_guards_expression()
    guards_expr: Optional[str]

    _boxed_call: Optional[bool] = None

    def __init__(
        self,
        current_callable: Optional[Callable[..., Any]],
        graph: GraphLowering,
        output_strides: List[Optional[Tuple[int, ...]]],
        disabled_cudagraphs_reason: Optional[str],
        metrics_deltas: metrics.CachedMetricsDeltas,
    ):
        self.current_callable = current_callable
        self.cache_key = graph.cache_key
        self.artifact_path = graph.cache_path
        self.cache_linemap = graph.cache_linemap
        self.device_types = graph.device_types
        self.device_idxs = graph.device_idxs
        self.mutated_inputs = graph.mutated_inputs
        self.mutated_input_idxs = set(graph.mutated_input_idxs)
        self.constants = graph.constants
        self.output_strides = output_strides
        self.disabled_cudagraphs_reason = disabled_cudagraphs_reason
        self.metrics_deltas = metrics_deltas
        self.guards_expr = None

    def __call__(self, inputs: List[Any]) -> Any:
        assert self.current_callable is not None
        return self.current_callable(inputs)


def cpp_compiler() -> str:
    if config.is_fbcode():
        return build_paths.cc()
    if isinstance(config.cpp.cxx, (list, tuple)):
        search = tuple(config.cpp.cxx)
    else:
        search = (config.cpp.cxx,)
    return cpp_compiler_search(search)


@functools.lru_cache(1)
def cpp_compiler_search(search: str) -> str:
    for cxx in search:
        try:
            if cxx is None:
                # gxx package is only available for Linux
                # according to https://anaconda.org/conda-forge/gxx/
                if sys.platform != "linux":
                    continue
                # Do not install GXX by default
                if not os.getenv("TORCH_INDUCTOR_INSTALL_GXX"):
                    continue
                from filelock import FileLock

                lock_dir = get_lock_dir()
                lock = FileLock(
                    os.path.join(lock_dir, "g++.lock"), timeout=LOCK_TIMEOUT
                )
                with lock:
                    cxx = install_gcc_via_conda()
            subprocess.check_output([cxx, "--version"])
            return cxx
        except (subprocess.SubprocessError, FileNotFoundError, ImportError):
            continue
    raise exc.InvalidCxxCompiler()


def install_gcc_via_conda() -> str:
    """On older systems, this is a quick way to get a modern compiler"""
    prefix = os.path.join(cache_dir(), "gcc")
    cxx_path = os.path.join(prefix, "bin", "g++")
    if not os.path.exists(cxx_path):
        log.info("Downloading GCC via conda")
        conda = os.environ.get("CONDA_EXE", "conda")
        if conda is None:
            conda = shutil.which("conda")
        if conda is not None:
            subprocess.check_call(
                [
                    conda,
                    "create",
                    f"--prefix={prefix}",
                    "--channel=conda-forge",
                    "--quiet",
                    "-y",
                    "python=3.8",
                    "gxx",
                ],
                stdout=subprocess.PIPE,
            )
    return cxx_path


def is_gcc() -> bool:
    if sys.platform == "darwin" and is_apple_clang():
        return False
    return bool(re.search(r"(gcc|g\+\+)", cpp_compiler()))


@functools.lru_cache(None)
def is_apple_clang() -> bool:
    cxx = cpp_compiler()
    version_string = subprocess.check_output([cxx, "--version"]).decode("utf8")
    return "Apple" in version_string.splitlines()[0]


def is_clang() -> bool:
    # Mac OS apple clang maybe named as gcc, need check compiler info.
    if sys.platform == "darwin":
        return is_apple_clang()
    return bool(re.search(r"(clang|clang\+\+)", cpp_compiler()))


class VecISA:
    _bit_width: int
    _macro: str
    _arch_flags: str
    _dtype_nelements: Dict[torch.dtype, int]

    # Note [Checking for Vectorized Support in Inductor]
    # TorchInductor CPU vectorization reuses PyTorch vectorization utility functions
    # Hence, TorchInductor would depend on Sleef* to accelerate mathematical functions
    # like exp, pow, sin, cos and etc.
    # But PyTorch and TorchInductor might use different compilers to build code. If
    # PyTorch uses gcc-7/g++-7 to build the release package, the libtorch_cpu.so
    # will not expose the Sleef* AVX512 symbols since gcc-7/g++-7 cannot pass
    # avx512 check in CMake - FindAVX.cmake. But TorchInductor install the latest
    # gcc/g++ compiler by default while it could support the AVX512 compilation.
    # Therefore, there would be a conflict sleef version between PyTorch and
    # TorchInductor. Hence, we dry-compile the following code to check whether current
    # HW platform and PyTorch both could support AVX512 or AVX2. And suppose ARM
    # also needs the logic
    # In fbcode however, we are using the same compiler for pytorch and for inductor codegen,
    # making the runtime check unnecessary.
    _avx_code = """
#if defined(CPU_CAPABILITY_AVX512) || defined(CPU_CAPABILITY_AVX2) || defined(CPU_CAPABILITY_ZVECTOR) || defined(CPU_CAPABILITY_NEON)
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#endif

__attribute__((aligned(64))) float in_out_ptr0[16] = {0.0};

extern "C" void __avx_chk_kernel() {
    auto tmp0 = at::vec::Vectorized<float>(1);
    auto tmp1 = tmp0.exp();
    tmp1.store(in_out_ptr0);
}
"""  # noqa: B950

    _avx_py_load = """
import torch
from ctypes import cdll
cdll.LoadLibrary("__lib_path__")
"""

    def bit_width(self) -> int:
        return self._bit_width

    def nelements(self, dtype: torch.dtype = torch.float) -> int:
        return self._dtype_nelements[dtype]

    def build_macro(self) -> str:
        return self._macro

    def build_arch_flags(self) -> str:
        return self._arch_flags

    def __hash__(self) -> int:
        return hash(str(self))

    @functools.lru_cache(None)
    def __bool__(self) -> bool:
        if config.cpp.vec_isa_ok is not None:
            return config.cpp.vec_isa_ok

        if config.is_fbcode():
            return True

        key, input_path = write(VecISA._avx_code, "cpp")
        from filelock import FileLock

        lock_dir = get_lock_dir()
        lock = FileLock(os.path.join(lock_dir, key + ".lock"), timeout=LOCK_TIMEOUT)
        with lock:
            output_path = input_path[:-3] + "so"
            build_cmd = shlex.split(
                cpp_compile_command(
                    input_path, output_path, warning_all=False, vec_isa=self
                )
            )
            try:
                # Check build result
                compile_file(input_path, output_path, build_cmd)
                subprocess.check_call(
                    [
                        sys.executable,
                        "-c",
                        VecISA._avx_py_load.replace("__lib_path__", output_path),
                    ],
                    stderr=subprocess.DEVNULL,
                    env={**os.environ, "PYTHONPATH": ":".join(sys.path)},
                )
            except Exception as e:
                return False

            return True


@dataclasses.dataclass
class VecNEON(VecISA):
    _bit_width = 256  # This is required to leverage the compute implemented in aten/src/ATen/cpu/vec/vec256/vec256_float_neon.h
    _macro = "-DCPU_CAPABILITY_NEON"
    _arch_flags = ""  # Unused
    _dtype_nelements = {torch.float: 8, torch.bfloat16: 16, torch.float16: 16}

    def __str__(self) -> str:
        return "neon"  # Unused

    __hash__: Callable[[VecISA], Any] = VecISA.__hash__


@dataclasses.dataclass
class VecAVX512(VecISA):
    _bit_width = 512
    _macro = "-DCPU_CAPABILITY_AVX512"
    _arch_flags = "-mavx512f -mavx512dq -mavx512vl -mavx512bw -mfma"
    _dtype_nelements = {torch.float: 16, torch.bfloat16: 32, torch.float16: 32}

    def __str__(self) -> str:
        return "avx512"

    __hash__: Callable[[VecISA], Any] = VecISA.__hash__


@dataclasses.dataclass
class VecAVX2(VecISA):
    _bit_width = 256
    _macro = "-DCPU_CAPABILITY_AVX2"
    _arch_flags = "-mavx2 -mfma"
    _dtype_nelements = {torch.float: 8, torch.bfloat16: 16, torch.float16: 16}

    def __str__(self) -> str:
        return "avx2"

    __hash__: Callable[[VecISA], Any] = VecISA.__hash__


@dataclasses.dataclass
class VecZVECTOR(VecISA):
    _bit_width = 256
    _macro = "-DCPU_CAPABILITY_ZVECTOR -DCPU_CAPABILITY=ZVECTOR -DHAVE_ZVECTOR_CPU_DEFINITION"
    _arch_flags = "-mvx -mzvector"
    _dtype_nelements = {torch.float: 8, torch.bfloat16: 16, torch.float16: 16}

    def __str__(self) -> str:
        return "zvector"

    __hash__: Callable[[VecISA], Any] = VecISA.__hash__


class InvalidVecISA(VecISA):
    _bit_width = 0
    _macro = ""
    _arch_flags = ""
    _dtype_nelements = {}

    def __str__(self) -> str:
        return "INVALID_VEC_ISA"

    def __bool__(self) -> bool:  # type: ignore[override]
        return False

    __hash__: Callable[[VecISA], Any] = VecISA.__hash__


invalid_vec_isa = InvalidVecISA()
supported_vec_isa_list = [VecAVX512(), VecAVX2()]


# Cache the cpuinfo to avoid I/O overhead. Meanwhile, the cpuinfo content
# might have too much redundant content that is useless for ISA check. Hence,
# we only cache some key isa information.
@functools.lru_cache(None)
def valid_vec_isa_list() -> List[VecISA]:
    if sys.platform == "darwin" and platform.processor() == "arm":
        return [VecNEON()]

    if sys.platform != "linux":
        return []

    if platform.machine() == "s390x":
        return [VecZVECTOR()]

    isa_list = []
    with open("/proc/cpuinfo") as _cpu_info:
        _cpu_info_content = _cpu_info.read()
        for isa in supported_vec_isa_list:
            if str(isa) in _cpu_info_content and isa:
                isa_list.append(isa)
        return isa_list


def pick_vec_isa() -> VecISA:
    if config.is_fbcode():
        return VecAVX2()

    _valid_vec_isa_list: List[VecISA] = valid_vec_isa_list()
    if not _valid_vec_isa_list:
        return invalid_vec_isa

    # If the simdlen is None, it indicates determin the vectorization length automatically
    if config.cpp.simdlen is None:
        assert _valid_vec_isa_list
        return _valid_vec_isa_list[0]

    for isa in _valid_vec_isa_list:
        if config.cpp.simdlen == isa.bit_width():
            return isa

    return invalid_vec_isa


def get_compile_only(compile_only: bool = True) -> str:
    return "-c" if compile_only else ""


def get_shared(shared: bool = True, compile_only: bool = False) -> str:
    if not shared:
        return ""
    if compile_only:
        return "-fPIC"
    if platform.system() == "Darwin" and "clang" in cpp_compiler():
        # This causes undefined symbols to behave the same as linux
        return "-shared -fPIC -undefined dynamic_lookup"
    else:
        return "-shared -fPIC"


def get_warning_all_flag(warning_all: bool = True) -> str:
    return "-Wall" if warning_all else ""


def get_glibcxx_abi_build_flags() -> str:
    return "-D_GLIBCXX_USE_CXX11_ABI=" + str(int(torch._C._GLIBCXX_USE_CXX11_ABI))


def cpp_flags() -> str:
    flags = ["-std=c++17", "-Wno-unused-variable", "-Wno-unknown-pragmas"]
    if is_clang():
        flags.append("-Werror=ignored-optimization-argument")
    return " ".join(flags)


def cpp_wrapper_flags() -> str:
    return "-DTORCH_INDUCTOR_CPP_WRAPPER"


def optimization_flags() -> str:
    base_flags = "-O0 -g" if config.aot_inductor.debug_compile else "-O3 -DNDEBUG"
    base_flags += " -ffast-math -fno-finite-math-only"
    if not config.cpp.enable_unsafe_math_opt_flag:
        base_flags += " -fno-unsafe-math-optimizations"
    if not config.cpp.enable_floating_point_contract_flag:
        base_flags += " -ffp-contract=off"

    if config.is_fbcode():
        # FIXME: passing `-fopenmp` adds libgomp.so to the generated shared library's dependencies.
        # This causes `ldopen` to fail in fbcode, because libgomp does not exist in the default paths.
        # We will fix it later by exposing the lib path.
        return base_flags

    if sys.platform == "darwin":
        # Per https://mac.r-project.org/openmp/ right way to pass `openmp` flags to MacOS is via `-Xclang`
        # Also, `-march=native` is unrecognized option on M1
        base_flags += " -Xclang"
    else:
        if platform.machine() == "ppc64le":
            base_flags += " -mcpu=native"
        else:
            base_flags += " -march=native"

    # Internal cannot find libgomp.so
    if not config.is_fbcode():
        base_flags += " -fopenmp"
    return base_flags


def use_custom_generated_macros() -> str:
    return "-D C10_USING_CUSTOM_GENERATED_MACROS"


def use_fb_internal_macros() -> str:
    if config.is_fbcode():
        openmp_lib = build_paths.openmp_lib()
        preprocessor_flags = " ".join(
            (
                "-D C10_USE_GLOG",
                "-D C10_USE_MINIMAL_GLOG",
                "-D C10_DISABLE_TENSORIMPL_EXTENSIBILITY",
            )
        )
        return f"-Wp,-fopenmp {openmp_lib} {preprocessor_flags}"
    else:
        return ""


def use_standard_sys_dir_headers() -> str:
    if config.is_fbcode():
        return "-nostdinc"
    else:
        return ""


@functools.lru_cache(None)
def is_conda_llvm_openmp_installed() -> bool:
    try:
        command = "conda list llvm-openmp --json"
        output = subprocess.check_output(command.split()).decode("utf8")
        return len(json.loads(output)) > 0
    except subprocess.SubprocessError:
        return False


@functools.lru_cache(None)
def homebrew_libomp() -> Tuple[bool, str]:
    try:
        # check if `brew` is installed
        subprocess.check_output(["which", "brew"])
        # get the location of `libomp` if it is installed
        # this is the location that `libomp` **would** be installed
        # see https://github.com/Homebrew/brew/issues/10261#issuecomment-756563567 for details
        libomp_path = (
            subprocess.check_output(["brew", "--prefix", "libomp"])
            .decode("utf8")
            .strip()
        )
        # check if `libomp` is installed
        omp_available = os.path.exists(libomp_path)
        return omp_available, libomp_path
    except subprocess.SubprocessError:
        return False, ""


def get_include_and_linking_paths(
    include_pytorch: bool = False,
    vec_isa: VecISA = invalid_vec_isa,
    cuda: bool = False,
    aot_mode: bool = False,
) -> Tuple[List[str], str, str, str, str]:
    if (
        config.is_fbcode()
        and "CUDA_HOME" not in os.environ
        and "CUDA_PATH" not in os.environ
    ):
        os.environ["CUDA_HOME"] = os.path.dirname(build_paths.cuda())
    from torch.utils import cpp_extension

    macros = vec_isa.build_macro() if vec_isa != invalid_vec_isa else ""
    build_arch_flags = ""
    if sys.platform == "linux" and (
        include_pytorch
        or vec_isa != invalid_vec_isa
        or cuda
        or config.cpp.enable_kernel_profile
    ):
        # Note - We include pytorch only on linux right now. There is more work
        # to do to enable OMP build on darwin where PyTorch is built with IOMP
        # and we need a way to link to what PyTorch links.
        ipaths = cpp_extension.include_paths(cuda) + [sysconfig.get_path("include")]
        lpaths = cpp_extension.library_paths(cuda) + [
            sysconfig.get_config_var("LIBDIR")
        ]

        libs = []

        # No need to manually specify libraries in fbcode.
        if not config.is_fbcode():
            libs += ["torch", "torch_cpu"]
            libs += ["gomp"]
            if not aot_mode:
                libs += ["torch_python"]
        else:
            # internal remote execution is able to find omp, but not gomp
            libs += ["omp"]
            if aot_mode:
                ipaths += [os.path.dirname(cpp_prefix_path())]
                if cuda:
                    # This is a special treatment for Meta internal cuda-12 where all libs
                    # are in lib/cuda-12 and lib/cuda-12/stubs
                    for i, path in enumerate(lpaths):
                        if path.startswith(
                            os.environ["CUDA_HOME"]
                        ) and not os.path.exists(f"{path}/libcudart_static.a"):
                            for root, dirs, files in os.walk(path):
                                if "libcudart_static.a" in files:
                                    lpaths[i] = os.path.join(path, root)
                                    lpaths.append(os.path.join(lpaths[i], "stubs"))
                                    break
        if macros:
            if config.is_fbcode() and vec_isa != invalid_vec_isa:
                cap = str(vec_isa).upper()
                macros = " ".join(
                    [
                        vec_isa.build_arch_flags(),
                        f"-D CPU_CAPABILITY={cap}",
                        f"-D CPU_CAPABILITY_{cap}",
                        f"-D HAVE_{cap}_CPU_DEFINITION",
                    ]
                )

        if cuda:
            if macros is None:
                macros = ""
            macros += " -D USE_ROCM" if torch.version.hip else " -D USE_CUDA"

        if cuda:
            if torch.version.hip is not None:
                libs += ["c10_hip", "torch_hip"]
                macros += " -D __HIP_PLATFORM_AMD__"
            else:
                if config.is_fbcode():
                    libs += ["cuda"]
                else:
                    libs += ["c10_cuda", "cuda", "torch_cuda"]
        build_arch_flags = vec_isa.build_arch_flags()
    else:
        # Note - this is effectively a header only inclusion. Usage of some header files may result in
        # symbol not found, if those header files require a library.
        # For those cases, include the lpath and libs command as we do for pytorch above.
        # This approach allows us to only pay for what we use.
        ipaths = cpp_extension.include_paths(cuda) + [sysconfig.get_path("include")]
        if aot_mode:
            ipaths += [os.path.dirname(cpp_prefix_path())]
        lpaths = []
        if sys.platform == "darwin":
            # only Apple builtin compilers (Apple Clang++) require openmp
            omp_available = not is_apple_clang()

            # check the `OMP_PREFIX` environment first
            if os.getenv("OMP_PREFIX") is not None:
                header_path = os.path.join(os.getenv("OMP_PREFIX"), "include", "omp.h")  # type: ignore[arg-type]
                valid_env = os.path.exists(header_path)
                if valid_env:
                    ipaths.append(os.path.join(os.getenv("OMP_PREFIX"), "include"))  # type: ignore[arg-type]
                    lpaths.append(os.path.join(os.getenv("OMP_PREFIX"), "lib"))  # type: ignore[arg-type]
                else:
                    warnings.warn("environment variable `OMP_PREFIX` is invalid.")
                omp_available = omp_available or valid_env

            libs = [] if omp_available else ["omp"]

            # prefer to use openmp from `conda install llvm-openmp`
            if not omp_available and os.getenv("CONDA_PREFIX") is not None:
                omp_available = is_conda_llvm_openmp_installed()
                if omp_available:
                    conda_lib_path = os.path.join(os.getenv("CONDA_PREFIX"), "lib")  # type: ignore[arg-type]
                    ipaths.append(os.path.join(os.getenv("CONDA_PREFIX"), "include"))  # type: ignore[arg-type]
                    lpaths.append(conda_lib_path)
                    # Prefer Intel OpenMP on x86 machine
                    if os.uname().machine == "x86_64" and os.path.exists(
                        os.path.join(conda_lib_path, "libiomp5.dylib")
                    ):
                        libs = ["iomp5"]

            # next, try to use openmp from `brew install libomp`
            if not omp_available:
                omp_available, libomp_path = homebrew_libomp()
                if omp_available:
                    ipaths.append(os.path.join(libomp_path, "include"))
                    lpaths.append(os.path.join(libomp_path, "lib"))

            # if openmp is still not available, we let the compiler to have a try,
            # and raise error together with instructions at compilation error later
        else:
            libs = ["omp"] if config.is_fbcode() else ["gomp"]

    # Unconditionally import c10 for non-abi-compatible mode to use TORCH_CHECK - See PyTorch #108690
    if not config.abi_compatible:
        libs += ["c10"]
        lpaths += [cpp_extension.TORCH_LIB_PATH]

    # third party libs
    if config.is_fbcode():
        ipaths.append(build_paths.sleef())
        ipaths.append(build_paths.openmp())
        ipaths.append(build_paths.python())
        ipaths.append(build_paths.cc_include())
        ipaths.append(build_paths.libgcc())
        ipaths.append(build_paths.libgcc_arch())
        ipaths.append(build_paths.libgcc_backward())
        ipaths.append(build_paths.glibc())
        ipaths.append(build_paths.linux_kernel())
        ipaths.append(build_paths.cuda())
        # We also need to bundle includes with absolute paths into a remote directory
        # (later on, we copy the include paths from cpp_extensions into our remote dir)
        ipaths.append("include")

    static_link_libs = []
    if aot_mode and cuda and config.is_fbcode():
        # For Meta internal cuda-12, it is recommended to static link cudart
        static_link_libs = ["-Wl,-Bstatic", "-lcudart_static", "-Wl,-Bdynamic"]

    lpaths_str = " ".join(["-L" + p for p in lpaths])
    libs_str = " ".join(static_link_libs + ["-l" + p for p in libs])
    return ipaths, lpaths_str, libs_str, macros, build_arch_flags


def cpp_compile_command(
    input: Union[str, List[str]],
    output: str,
    warning_all: bool = True,
    shared: bool = True,
    include_pytorch: bool = False,
    vec_isa: VecISA = invalid_vec_isa,
    cuda: bool = False,
    aot_mode: bool = False,
    compile_only: bool = False,
    use_absolute_path: bool = False,
) -> str:
    ipaths, lpaths, libs, macros, build_arch_flags = get_include_and_linking_paths(
        include_pytorch, vec_isa, cuda, aot_mode
    )
    if isinstance(input, str):
        input = [input]
    ipaths_str = " ".join(["-I" + p for p in ipaths])
    clang_flags = ""
    if config.is_fbcode():
        if aot_mode and not use_absolute_path:
            inp_name = input
            out_name = output
            linker_script = _LINKER_SCRIPT
        else:
            # We need to copy any absolute-path torch includes
            inp_name = [os.path.basename(i) for i in input]
            out_name = os.path.basename(output)
            linker_script = os.path.basename(_LINKER_SCRIPT)
        assert is_clang()
        # Use clang runtime instead of libgcc
        clang_flags += " --rtlib=compiler-rt"
        clang_flags += " -fuse-ld=lld"
        clang_flags += f" -Wl,--script={linker_script}"
        linker_paths = "-B" + build_paths.glibc_lib()
        linker_paths += " -L" + build_paths.glibc_lib()
    else:
        inp_name = input
        out_name = output
        linker_paths = ""  # let the compiler pick
    if compile_only:
        libs, lpaths = "", ""
    inp_name_str = " ".join(inp_name)
    return re.sub(
        r"[ \n]+",
        " ",
        f"""
            {cpp_compiler()} {inp_name_str} {get_shared(shared, compile_only)}
            {get_warning_all_flag(warning_all)} {cpp_flags()}
            {get_glibcxx_abi_build_flags()}
            {ipaths_str} {lpaths} {libs} {build_arch_flags}
            {macros} {linker_paths} {clang_flags}
            {optimization_flags()}
            {use_custom_generated_macros()}
            {use_fb_internal_macros()}
            {use_standard_sys_dir_headers()}
            {get_compile_only(compile_only)}
            -o {out_name}
        """,
    ).strip()


def run_command_and_check(cmd: str):
    cmd = shlex.split(cmd)
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        raise exc.CppCompileError(cmd, e.output) from e


@functools.lru_cache(None)
def split_aot_inductor_output_path(path: str) -> Tuple[str, str]:
    """Returns the path where the AOT Inductor compiled kernels are stored."""
    if path.endswith(".so"):
        return os.path.split(path)
    else:
        return path, ""


class CudaKernelParamCache:
    cache: Dict[str, Dict[str, str]] = dict()
    clear = staticmethod(cache.clear)

    @classmethod
    def set(cls, key: str, params: Dict[str, str], cubin: str) -> None:
        bin_type = "cubin" if torch.version.hip is None else "hsaco"
        _, path = write(
            cubin,
            bin_type,
            hash_type=bin_type,
            specified_dir=split_aot_inductor_output_path(
                config.aot_inductor.output_path
            )[0],
        )

        params[get_cpp_wrapper_cubin_path_name()] = path

        cls.cache[key] = params

    @classmethod
    def get(cls, key: str) -> Optional[Dict[str, str]]:
        return cls.cache.get(key, None)

    @classmethod
    def get_keys(cls):
        return cls.cache.keys()


class AotCodeCompiler:
    @classmethod
    def compile(
        cls,
        graph: GraphLowering,
        source_code: str,
        serialized_extern_kernel_nodes: Optional[str],
        cuda: bool,
    ) -> str:
        picked_vec_isa = pick_vec_isa()
        cpp_command = repr(
            cpp_compile_command(
                "i", "o", vec_isa=picked_vec_isa, cuda=cuda, aot_mode=graph.aot_mode
            )
        )
        fbcode_aot_cpu_re = False
        use_absolute_path = False
        if config.is_fbcode():
            ld_command = build_paths.ld()
            if not cuda and graph.aot_mode:  # Meta internal AOTInductor CPU
                objcopy_command = build_paths.objcopy_fallback()
                fbcode_aot_cpu_re = True
                use_absolute_path = True
            else:
                objcopy_command = build_paths.objcopy()
        else:
            ld_command = "ld"
            objcopy_command = "objcopy"

        (
            specified_output_path,
            specified_so_name,
        ) = split_aot_inductor_output_path(config.aot_inductor.output_path)
        key, input_path = write(
            source_code,
            "cpp",
            extra=cpp_command,
            specified_dir=specified_output_path,
        )
        output_code_log.info("Output code written to: %s", input_path)

        def _compile_consts_linux(consts: bytes) -> str:
            _, consts_path = write(
                consts,
                "bin",
                specified_dir=specified_output_path,
            )

            consts_o = os.path.splitext(consts_path)[0] + ".o"
            if fbcode_aot_cpu_re:
                cmd = f"{ld_command} -r -b binary -o {os.path.basename(consts_o)} {os.path.basename(consts_path)}"
                compile_file(consts_path, consts_o, cmd.split())
                os.chmod(consts_o, 0o644)
            else:
                cmd = f"{ld_command} -r -b binary -o {consts_o} {consts_path}"
                run_command_and_check(cmd)
            log.debug("aot constant binary command: %s", cmd)

            cmd = f"rm {consts_path}"
            log.debug("aot constant bin removal command: %s", cmd)
            run_command_and_check(cmd)

            if fbcode_aot_cpu_re:
                body = re.sub(r"[\W]", "_", os.path.basename(consts_path))
            else:
                body = re.sub(r"[\W]", "_", consts_path)

            symbol_list = []
            symbol_list.append(
                f"{objcopy_command} --redefine-sym _binary_{body}_start=_binary_constants_bin_start {consts_o}"
            )
            symbol_list.append(
                f"{objcopy_command} --redefine-sym _binary_{body}_size=_binary_constants_bin_size {consts_o}"
            )
            symbol_list.append(
                f"{objcopy_command} --redefine-sym _binary_{body}_end=_binary_constants_bin_end {consts_o}"
            )
            log.debug("aot constant binary redefine symbol: %s", " ".join(symbol_list))
            for cmd in symbol_list:
                run_command_and_check(cmd)
            return consts_o

        def _compile_consts_darwin(consts: bytes) -> str:
            is_large_consts = len(consts) > 1024
            consts_asm = "\t.section\t__DATA,__data\n"
            consts_asm += "\t.globl\t__binary_constants_bin_start\n"
            consts_asm += "__binary_constants_bin_start:\n"
            if not is_large_consts:
                for c in consts:
                    consts_asm += f"\t.byte {c}\n"
                # Add one element even if constants are empty
                # Otherwise assembler will not put them in data section
                if not consts:
                    consts_asm += "\t.space 1\n"
            else:
                consts_asm += "\t.quad 0x1234567899abcdef\n"
                consts_asm += f"\t.space {len(consts) - 8}\n"
            consts_asm += ".globl\t__binary_constants_bin_end\n"
            consts_asm += "__binary_constants_bin_end:\n"
            _, consts_path = write(
                consts_asm,
                "S",
                specified_dir=specified_output_path,
            )
            consts_o = os.path.splitext(consts_path)[0] + ".o"
            cmd = f"{cpp_compiler()} -c -o {consts_o} {consts_path}"
            run_command_and_check(cmd)
            if is_large_consts:
                with open(consts_o, "r+b") as f:
                    f.seek(0)
                    hdr = f.read(1024)
                    # Search for magic number and write the actual data over it
                    start_idx = hdr.find(b"\xef\xcd\xab\x99\x78\x56\x34\x12")
                    assert start_idx != -1
                    f.seek(start_idx)
                    pos = 0
                    while pos < len(consts):
                        rc = f.write(consts[pos:])
                        pos += rc
            return consts_o

        from filelock import FileLock

        lock_dir = get_lock_dir()
        lock = FileLock(os.path.join(lock_dir, key + ".lock"), timeout=LOCK_TIMEOUT)
        with lock:
            # Currently, this only support serializing extern nodes in fbcode
            # Eventually, we should also have a serializer for OSS.
            if config.is_fbcode() and serialized_extern_kernel_nodes:
                output_json = os.path.splitext(input_path)[0] + ".json"
                with open(output_json, "w") as f:
                    f.write(serialized_extern_kernel_nodes)

            output_so = (
                config.aot_inductor.output_path
                if specified_so_name
                else os.path.splitext(input_path)[0] + ".so"
            )

            output_o = os.path.splitext(input_path)[0] + ".o"
            compile_cmd = cpp_compile_command(
                input=input_path,
                output=output_o,
                vec_isa=picked_vec_isa,
                cuda=cuda,
                aot_mode=graph.aot_mode,
                compile_only=True,
                use_absolute_path=use_absolute_path,
            )
            log.debug("aot compilation command: %s", compile_cmd)
            if fbcode_aot_cpu_re:
                compile_file(input_path, output_o, compile_cmd.split())
                os.chmod(output_o, 0o644)
            else:
                run_command_and_check(compile_cmd)

            def _to_bytes(t: torch.Tensor) -> bytes:
                # This serializes the tensor's untyped_storage to bytes by accessing
                # the raw data of the underlying structure.
                import ctypes

                if t.numel() == 0:
                    return b""

                t_cpu = t.untyped_storage().cpu()
                raw_array = ctypes.cast(
                    t_cpu.data_ptr(),
                    ctypes.POINTER(ctypes.c_ubyte * t_cpu.nbytes()),
                )

                return bytes(raw_array.contents)

            aot_constants = b"".join(
                _to_bytes(graph.get_original_value_of_constant(name))
                for name in graph.constants.keys()
                if name not in graph.folded_constants
            )
            consts_o = {
                "linux": _compile_consts_linux,
                "darwin": _compile_consts_darwin,
            }[sys.platform](aot_constants)

            link_cmd = cpp_compile_command(
                input=[output_o, consts_o],
                output=output_so,
                vec_isa=picked_vec_isa,
                cuda=cuda,
                aot_mode=graph.aot_mode,
                use_absolute_path=use_absolute_path,
            )
            log.debug("aot linkage command: %s", link_cmd)
            if fbcode_aot_cpu_re:
                compile_file([output_o, consts_o], output_so, link_cmd.split())
                os.chmod(output_so, 0o755)
            else:
                run_command_and_check(link_cmd)

            # Append cmds to the end of codegen-ed wrapper file
            with open(input_path, "a") as f:
                f.write("\n")
                f.write(f"// Compile cmd\n// {compile_cmd}\n")
                f.write(f"// Link cmd\n// {link_cmd}\n")

        return output_so


# Putting this fn in cpp.py (unfortunately) causes a deadlock, which is why it's in codecache.py.
# Why? importing from cpp.py invokes codecache.pick_vec_isa(), which takes out a lock.
# Cycle goes:
# - CppCodeCache.load()
# - pick_vec_isa()
# - valid_vec_isa_list()
# - VecISA.__bool__() <-- takes out a lock
# - compile_file() <-- imports cpp_prefix_path from cpp, which causes us to try to take out the same lock.
@functools.lru_cache
def cpp_prefix_path() -> str:
    path = Path(__file__).parent / "codegen/cpp_prefix.h"
    with path.open() as f:
        content = f.read()
        _, filename = write(
            content,
            "h",
        )
    return filename


def cpp_prefix() -> str:
    filename = cpp_prefix_path()
    if config.is_fbcode():
        # We need relative paths, since we bundle up
        # everything that we compile into a folder for remote compilation.
        return f'#include "{os.path.basename(filename)}"'
    else:
        return f'#include "{filename}"'


# Given a path to an input cpp file and an output path,
# Attempts to compile the file, storing the output in "output_path"
@dynamo_timed
def compile_file(
    input_path: Union[str, List[str]], output_path: str, cmd: List[str]
) -> None:
    input_paths = [input_path] if isinstance(input_path, str) else input_path
    input_files = [
        os.path.basename(ip) if config.is_fbcode() else ip for ip in input_paths
    ]
    try:
        if config.is_fbcode():
            # Need to copy our header into the same folder as the sourcecode.
            header_path = cpp_prefix_path()
            header_name = os.path.basename(header_path)
            output_name = os.path.basename(output_path)
            # When we build remotely, we need to make sure to carefully copy any files
            # that are required during the compilation process into our build directly.
            # This is where all of the ATen/c10/Torch includes come from.
            torch_includes_path = os.path.join(_TORCH_PATH, "include")
            with tempfile.TemporaryDirectory() as tmp_dir:
                # Copy everything to tmp compilation folder
                shutil.copy(header_path, os.path.join(tmp_dir, header_name))
                shutil.copy(_LINKER_SCRIPT, os.path.join(tmp_dir, "script.ld"))
                for p, f in zip(input_paths, input_files):
                    shutil.copy(p, os.path.join(tmp_dir, f))
                dest_include_path = os.path.join(tmp_dir, "include")
                shutil.copytree(torch_includes_path, dest_include_path)
                # Run the build
                output_file_path = _run_build_command(cmd, tmp_dir, output_name)
                # Copy output from the build
                if os.path.exists(output_path):
                    os.remove(output_path)
                shutil.copy(output_file_path, output_path)
        else:
            subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        output = e.output.decode("utf-8")
        openmp_problem = "'omp.h' file not found" in output or "libomp" in output
        if openmp_problem and sys.platform == "darwin":
            instruction = (
                "\n\nOpenMP support not found. Please try one of the following solutions:\n"
                "(1) Set the `CXX` environment variable to a compiler other than Apple clang++/g++ "
                "that has builtin OpenMP support;\n"
                "(2) install OpenMP via conda: `conda install llvm-openmp`;\n"
                "(3) install libomp via brew: `brew install libomp`;\n"
                "(4) manually setup OpenMP and set the `OMP_PREFIX` environment variable to point to a path"
                " with `include/omp.h` under it."
            )
            output += instruction
        raise exc.CppCompileError(cmd, output) from e


_libgomp: Optional[CDLL] = None


def custom_op_wrapper(op: str, *args):
    # This function will be called from generated cpp wrapper code in the JIT mode.
    # Because tensors will be passed in as AtenTensorHandle, we need to explicitly convert them.
    def convert_arg(arg):
        if str(type(arg)) == "<class 'PyCapsule'>":
            # No easy way to do isinstance check on PyCapsule
            return torch._C._aoti.alloc_tensor_by_stealing_from_void_ptr(arg)
        elif isinstance(arg, (list, tuple)):
            return type(arg)(convert_arg(a) for a in arg)
        else:
            return arg

    converted_args = [convert_arg(arg) for arg in args]

    assert op.startswith("torch.ops."), (
        op + " can not be called through custom_op_wrapper"
    )
    func = None
    for i, s in enumerate(op.split(".")):
        if i == 0:
            func = importlib.import_module(s)
        func = getattr(func, s)

    assert callable(func), op + " can not be loaded through custom_op_wrapper"
    result = func(*converted_args)
    if isinstance(result, (list, tuple)):
        for r in result:
            assert isinstance(r, torch.Tensor), op + " returns a list of non-tensors"
        return torch._C._aoti.unsafe_alloc_void_ptrs_from_tensors(result)  # type: ignore[arg-type]
    else:
        assert isinstance(result, torch.Tensor), op + " returns a non-tensor"
        return torch._C._aoti.unsafe_alloc_void_ptr_from_tensor(result)


class CppCodeCache:
    cache: Dict[str, Union[CDLL, ModuleType]] = {}
    clear = staticmethod(cache.clear)
    cpp_compile_command_flags: Dict[str, Any] = {}

    @staticmethod
    def _load_library_inner(path: str, key: str) -> Union[CDLL, ModuleType]:
        return cdll.LoadLibrary(path)

    @classmethod
    def _load_library(cls, path: str, key: str) -> Union[CDLL, ModuleType]:
        try:
            return cls._load_library_inner(path, key)
        except (ImportError, OSError) as e:
            if "gomp" in str(e) and os.path.exists("/usr/lib64/libgomp.so.1"):
                # hacky workaround for fbcode/buck
                global _libgomp
                _libgomp = cdll.LoadLibrary("/usr/lib64/libgomp.so.1")
                return cls._load_library_inner(path, key)
            if "failed to map segment from shared object" in str(e):
                raise OSError(
                    f"{e}.  The most common reason this may occur is if the {tempfile.gettempdir()} folder "
                    "is mounted with noexec (e.g., by default Docker mounts tmp file systems "
                    f"as noexec).  Please remount {tempfile.gettempdir()} with exec enabled, or set another "
                    "temporary directory with TORCHINDUCTOR_CACHE_DIR environment variable."
                ) from e
            raise

    @classmethod
    def load(cls, source_code: str, cuda: bool = False) -> Union[CDLL, ModuleType]:
        cls.cpp_compile_command_flags.update({"cuda": cuda})
        picked_vec_isa = pick_vec_isa()
        cpp_command = repr(
            cpp_compile_command(
                "i", "o", vec_isa=picked_vec_isa, **cls.cpp_compile_command_flags
            )
        )
        key, input_path = write(source_code, "cpp", extra=cpp_command)
        if key not in cls.cache:
            from filelock import FileLock

            lock_dir = get_lock_dir()
            lock = FileLock(os.path.join(lock_dir, key + ".lock"), timeout=LOCK_TIMEOUT)
            with lock:
                output_path = input_path[:-3] + "so"
                if not os.path.exists(output_path):
                    cmd = shlex.split(
                        cpp_compile_command(
                            input=input_path,
                            output=output_path,
                            vec_isa=picked_vec_isa,
                            **cls.cpp_compile_command_flags,
                        )
                    )
                    compile_file(input_path, output_path, cmd)
                cls.cache[key] = cls._load_library(output_path, key)
                cls.cache[key].key = key  # type: ignore[union-attr]

        return cls.cache[key]


# Customized Python binding for cpp kernels
class CppPythonBindingsCodeCache(CppCodeCache):
    cache: Dict[str, Union[CDLL, ModuleType]] = {}
    clear = staticmethod(cache.clear)
    cpp_compile_command_flags = {
        # kernels have no dependency on libtorch
        "include_pytorch": False,
        "shared": True,
    }
    entry_function = "kernel"
    call_entry_function = "kernel(%s);Py_RETURN_NONE;"
    extra_parse_arg = ""
    suffix_template = textwrap.dedent(
        """
        // Python bindings to call %s():
        #define PY_SSIZE_T_CLEAN
        #include <Python.h>
        #include <sstream>
        #include <cstdlib>

        // This is defined in guards.cpp so we don't need to import PyTorch headers that are slooow.
        // We manually link it below to workaround issues with fbcode build.
        static void* (*_torchinductor_pyobject_tensor_data_ptr)(PyObject* obj);

        template <typename T> static inline T parse_arg(PyObject* args, size_t n) {
            static_assert(std::is_pointer<T>::value, "arg type must be pointer or long");
            return static_cast<T>(_torchinductor_pyobject_tensor_data_ptr(PyTuple_GET_ITEM(args, n)));
        }
        template <> inline long parse_arg<long>(PyObject* args, size_t n) {
            auto result = PyLong_AsSsize_t(PyTuple_GET_ITEM(args, n));
            if(result == -1 && PyErr_Occurred())
                [[unlikely]] throw std::runtime_error("expected int arg");
            return result;
        }

        %s

        static PyObject* %s_py(PyObject* self, PyObject* args) {
            try {
                if(!PyTuple_CheckExact(args))
                    [[unlikely]] throw std::runtime_error("tuple args required");
                if(PyTuple_GET_SIZE(args) != %s)
                    [[unlikely]] throw std::runtime_error("requires %s args");
                %s
            } catch(std::exception const& e) {
                PyErr_SetString(PyExc_RuntimeError, e.what());
                return nullptr;
            } catch(...) {
                PyErr_SetString(PyExc_RuntimeError, "unhandled error");
                return nullptr;
            }
        }

        static PyMethodDef py_methods[] = {
            {"%s", %s_py, METH_VARARGS, ""},
            {NULL, NULL, 0, NULL}};

        static struct PyModuleDef py_module =
            {PyModuleDef_HEAD_INIT, "%s", NULL, -1, py_methods};

        PyMODINIT_FUNC PyInit_%s(void) {
            const char* str_addr = std::getenv("_TORCHINDUCTOR_PYOBJECT_TENSOR_DATA_PTR");
            if(!str_addr) {
                PyErr_SetString(PyExc_RuntimeError, "_TORCHINDUCTOR_PYOBJECT_TENSOR_DATA_PTR must be set");
                return nullptr;
            }
            std::istringstream iss(str_addr);
            uintptr_t addr = 0;
            iss >> addr;
            _torchinductor_pyobject_tensor_data_ptr =
                reinterpret_cast<decltype(_torchinductor_pyobject_tensor_data_ptr)>(addr);
            return PyModule_Create(&py_module);
        }
        """
    )

    @classmethod
    def _load_library_inner(cls, path: str, key: str) -> ModuleType:
        os.environ["_TORCHINDUCTOR_PYOBJECT_TENSOR_DATA_PTR"] = str(
            torch._C._dynamo.guards._torchinductor_pyobject_tensor_data_ptr  # type: ignore[attr-defined]
        )
        module_name = f"{key}.{cls.entry_function}"
        try:
            return sys.modules[module_name]
        except KeyError:
            pass
        spec = importlib.util.spec_from_file_location(module_name, path)
        assert spec is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)  # type: ignore[union-attr]
        return module

    @classmethod
    def load_pybinding(
        cls,
        argtypes: List[str],
        source_code: str,
        cuda: bool = False,
        num_outputs: int = -1,
    ) -> Any:
        """
        Wrap a C++ function in fast Python bindings.

        Args:
            argtypes: The types of args to ENTRY_FUNCTION(), e.g. ["float*", "long"]
            source_code: C++ source code containing a ENTRY_FUNCTION() function

        Returns:
            A python version of ENTRY_FUNCTION()
        """
        parseargs = ", ".join(
            f"parse_arg<{argtype.replace('const ', '')}>(args, {n})"
            for n, argtype in enumerate(argtypes)
        )
        suffix = cls.suffix_template % (
            cls.entry_function,
            cls.extra_parse_arg % num_outputs if cls.extra_parse_arg else "",
            cls.entry_function,
            len(argtypes),
            len(argtypes),
            cls.call_entry_function % parseargs,
            cls.entry_function,
            cls.entry_function,
            cls.entry_function,
            cls.entry_function,
        )
        result = cls.load(source_code + suffix, cuda)
        assert isinstance(result, ModuleType)
        return getattr(result, cls.entry_function)


class CppWrapperCodeCache(CppPythonBindingsCodeCache):
    cache: Dict[str, Union[CDLL, ModuleType]] = {}
    clear = staticmethod(cache.clear)
    cpp_compile_command_flags = {
        "include_pytorch": not config.abi_compatible,
        "shared": True,
    }
    entry_function = "inductor_entry_cpp"
    call_entry_function = "return inductor_entry_cpp(%s);"
    extra_parse_arg = textwrap.dedent(
        """
        #include <torch/csrc/inductor/aoti_torch/c/shim.h>

        static inline std::vector<AtenTensorHandle> unpack_tensor_handle_list(PyObject* pyvec) {
            std::vector<AtenTensorHandle> result;
            size_t result_len = PyList_GET_SIZE(pyvec);
            result.reserve(result_len);
            for (size_t i = 0; i < result_len; i++) {
                // AtenTensorHandle is essentially a pointer
                void* elem = PyCapsule_GetPointer(PyList_GET_ITEM(pyvec, i), NULL);
                result.push_back(reinterpret_cast<AtenTensorHandle>(elem));
            }
            return result;
        }

        static inline PyObject* pack_tensor_handle_list(const std::vector<AtenTensorHandle>& cppvec) {
            size_t result_len = cppvec.size();
            PyObject* result = PyList_New(static_cast<Py_ssize_t>(result_len));
            for (size_t i = 0; i < result_len; i++) {
                PyObject *elem =
                    cppvec[i] == nullptr
                        ? Py_None
                        // Store AtenTensorHandle as PyCapsulate
                        : PyCapsule_New(reinterpret_cast<void*>(cppvec[i]), NULL, NULL);
                PyList_SET_ITEM(result, i, elem);
            }
            return result;
        }

        template <> inline std::vector<AtenTensorHandle> parse_arg<std::vector<AtenTensorHandle>>(PyObject* args, size_t n) {
            return unpack_tensor_handle_list(PyTuple_GET_ITEM(args, n));
        }

        PyObject* inductor_entry_cpp(std::vector<AtenTensorHandle>&& input_handles) {
            // For outputs, we only allocate a vector to hold returned tensor handles,
            // not allocating the actual output tensor storage here
            std::vector<AtenTensorHandle> output_handles(%s);
            try {
                inductor_entry_impl(input_handles.data(), output_handles.data());
                return pack_tensor_handle_list(output_handles);
            } catch(std::exception const& e) {
                PyErr_SetString(PyExc_RuntimeError, e.what());
                return {};
            } catch(...) {
                PyErr_SetString(PyExc_RuntimeError, "unhandled error");
                return {};
            }
        }
        """
    )


class PyCodeCache:
    cache: Dict[str, ModuleType] = dict()
    linemaps: Dict[str, List[Tuple[Any, ...]]] = dict()
    clear = staticmethod(cache.clear)

    @classmethod
    def write(cls, source_code: str, extra: str = "") -> Tuple[str, str]:
        return write(source_code, "py", extra=extra)

    @classmethod
    def load(
        cls,
        source_code: str,
        extra: str = "",
        linemap: Optional[List[Tuple[int, str]]] = None,
        attrs: Optional[Dict[str, Any]] = None,
    ) -> ModuleType:
        key, path = write(source_code, "py", extra=extra)
        return cls.load_by_key_path(key, path, linemap, attrs)

    @classmethod
    def load_by_key_path(
        cls,
        key: str,
        path: str,
        linemap: Optional[List[Tuple[int, str]]] = None,
        attrs: Optional[Dict[str, Any]] = None,
    ) -> ModuleType:
        if linemap is None:
            linemap = []
        if key not in cls.cache:
            with open(path) as f:
                try:
                    code = compile(f.read(), path, "exec")
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to import {path}\n{type(e).__name__}: {e}"
                    ) from None
                mod = ModuleType(f"{__name__}.{key}")
                mod.__file__ = path
                mod.key = key  # type: ignore[attr-defined]
                try:
                    exec(code, mod.__dict__, mod.__dict__)
                except Exception as e:
                    log.debug(f"problematic file path: {path}")
                    raise
                sys.modules[mod.__name__] = mod
                # another thread might set this first
                cls.cache.setdefault(key, mod)
                # unzip into separate lines/nodes lists
                cls.linemaps[path] = list(zip(*linemap))

                if attrs is not None:
                    for k, v in attrs.items():
                        setattr(mod, k, v)

        return cls.cache[key]

    @classmethod
    @functools.lru_cache(None)
    def stack_frames_for_code(
        cls, path: str, lineno: int
    ) -> Optional[List[Dict[str, Any]]]:
        if path not in cls.linemaps:
            return None
        # [(starting_line, <fx node>), ...]
        lines, nodes = cls.linemaps[path]
        p = bisect_right(lines, lineno)
        if p == 0:
            return None
        entry = nodes[p - 1]
        if not entry:
            return None

        def parse_stack_trace(stack_trace: str) -> List[Dict[str, Any]]:
            # ideally fx stores stack traces as data rather than a string
            # but this is not along a performance critical path
            regex = r'File "(.+)", line (\d+), in (.+)\n'
            matches = re.findall(regex, stack_trace)
            return [
                {"filename": f, "line": int(l), "name": n}
                for f, l, n in reversed(matches)
            ]

        return parse_stack_trace(entry)


class TritonCodeCache:
    @classmethod
    def load(cls, kernel_name: str, source_code: str) -> ModuleType:
        mod = PyCodeCache.load(source_code)
        return getattr(mod, kernel_name)


def _cuda_compiler() -> Optional[str]:
    if cuda_env.nvcc_exist(config.cuda.cuda_cxx):
        return config.cuda.cuda_cxx
    if cuda_env.nvcc_exist(os.getenv("CUDACXX")):
        return os.getenv("CUDACXX", "")
    if cuda_env.nvcc_exist(os.getenv("CUDA_HOME")):
        return os.path.realpath(os.path.join(os.getenv("CUDA_HOME", ""), "bin/nvcc"))
    return "nvcc"


def _cutlass_include_paths() -> List[str]:
    cutlass_path = config.cuda.cutlass_dir
    return [
        # Use realpath to get canonical absolute paths, in order not to mess up cache keys
        os.path.realpath(os.path.join(cutlass_path, "include")),
        os.path.realpath(os.path.join(cutlass_path, "tools/library/include")),
        os.path.realpath(os.path.join(cutlass_path, "tools/library/src")),
        os.path.realpath(os.path.join(cutlass_path, "tools/util/include")),
    ]


def _cuda_lib_options() -> List[str]:
    from torch.utils import cpp_extension

    extra_ldflags: List[str] = []
    if is_linux():
        extra_lib_dir = "lib64"
        if not os.path.exists(
            cpp_extension._join_cuda_home(extra_lib_dir)
        ) and os.path.exists(cpp_extension._join_cuda_home("lib")):
            # 64-bit CUDA may be installed in "lib"
            # Note that it's also possible both don't exist (see _find_cuda_home) - in that case we stay with "lib64"
            extra_lib_dir = "lib"
        extra_ldflags.append(f"-L{cpp_extension._join_cuda_home(extra_lib_dir)}")
        extra_ldflags.append(
            f'-L{cpp_extension._join_cuda_home(extra_lib_dir, "stubs")}'
        )
        extra_ldflags.append("-lcuda")
        extra_ldflags.append("-lcudart")
    else:
        raise NotImplementedError(
            "Unsupported env, failed to find cuda libs! Currently only Linux is supported."
        )
    return extra_ldflags


def _nvcc_host_compiler_options() -> List[str]:
    return [
        "-fPIC",
        "-fno-strict-aliasing",
        "-fvisibility=hidden",
        "-Wconversion",
    ]


def _nvcc_compiler_options() -> List[str]:
    arch = cuda_env.get_cuda_arch()
    if arch == "90":
        # Required by cutlass compilation.
        arch = "90a"
    code = [f"sm_{arch}", f"compute_{arch}"]
    if config.cuda.enable_cuda_lto:
        code += [f"lto_{arch}"]
    options = [
        "-t=0",
        "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
        "-w",
        f"-gencode=arch=compute_{arch},code=[{','.join(code)}]",
        config.cuda.compile_opt_level,
        "-std=c++17",
        "--expt-relaxed-constexpr",
        "-DNDEBUG",
    ]
    if config.cuda.enable_debug_info:
        options.extend(["-lineinfo", "-g", "-DCUTLASS_DEBUG_TRACE_LEVEL=1"])
    if config.cuda.enable_ptxas_info:
        options.extend(
            [
                "--keep",  # Keep the intermediate files for debugging (including ptx, sass, cubin etc.)
                "--ptxas-options=--warn-on-local-memory-usage",  # warn us if local memory is used in CUDA Kernels
                "--ptxas-options=--warn-on-spills",  # warn us if register spilling happens in CUDA Kernels
                "--resource-usage",  # Report on CUDA resource usage (shared mem, registers etc.)
                "--source-in-ptx",
            ]
        )  # Annotate the ptx file with source information
    if config.cuda.use_fast_math:
        options.extend(
            [
                "--use_fast_math",
                "-DCUTLASS_USE_TANH_FOR_SIGMOID=1",
            ]
        )
    return options


def cuda_compile_command(
    src_files: List[str],
    dst_file: str,
    dst_file_ext: str,
    extra_args: Optional[List[str]] = None,
) -> str:
    if extra_args is None:
        extra_args = []
    include_paths = _cutlass_include_paths()
    cuda_lib_options = _cuda_lib_options()
    nvcc_host_compiler_options = _nvcc_host_compiler_options()
    nvcc_compiler_options = _nvcc_compiler_options()
    options = (
        nvcc_compiler_options
        + extra_args
        + [
            f"-Xcompiler {opt}" if "=" in opt else f"-Xcompiler={opt}"
            for opt in nvcc_host_compiler_options
        ]
        + ["-I" + path for path in include_paths]
        + cuda_lib_options
    )
    src_file = " ".join(src_files)
    res = ""
    if dst_file_ext == "o":
        res = f"{_cuda_compiler()} {' '.join(options)} -c -o {dst_file} {src_file}"
    elif dst_file_ext == "so":
        options.append("-shared")
        res = f"{_cuda_compiler()} {' '.join(options)} -o {dst_file} {src_file}"
    elif dst_file_ext == "exe":
        res = f"{_cuda_compiler()} {' '.join(options)} -o {dst_file} {src_file}"
    else:
        raise NotImplementedError(f"Unsupported output file suffix {dst_file_ext}!")
    log.debug("CUDA command: %s", res)
    return res


class DLLWrapper:
    """A wrapper for a dynamic library."""

    def __init__(
        self,
        lib_path: str,
    ):
        self.lib_path = lib_path
        self.DLL = cdll.LoadLibrary(lib_path)
        self.is_open = True

    def close(self):
        if self.is_open:
            self._dlclose()
            self.is_open = False

    def _dlclose(self):
        f_dlclose = None

        if is_linux():
            syms = CDLL(None)
            if not hasattr(syms, "dlclose"):
                # Apline Linux
                syms = CDLL("libc.so")

            if hasattr(syms, "dlclose"):
                f_dlclose = syms.dlclose
        else:
            raise NotImplementedError("Unsupported env, failed to do dlclose!")

        if f_dlclose is not None:
            f_dlclose.argtypes = [c_void_p]
            f_dlclose(self.DLL._handle)
        else:
            log.warning(
                "dll unloading function was not found, library may not be unloaded properly!"
            )

    def __getattr__(self, name):
        if not self.is_open:
            raise RuntimeError(f"Cannot use closed DLL library: {self.lib_path}")

        method = getattr(self.DLL, name)

        def _wrapped_func(*args):
            err = method(*args)
            if err:
                raise RuntimeError(f"Error in function: {method.__name__}")

        return _wrapped_func

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()


class CUDACodeCache:
    @dataclasses.dataclass
    class CacheEntry:
        input_path: str
        output_path: str

    cache: Dict[str, CacheEntry] = dict()
    clear = staticmethod(cache.clear)
    _SOURCE_CODE_SUFFIX = "cu"

    @classmethod
    def write(cls, source_code, dst_file_ext) -> Tuple[str, str]:
        """
        Writes source code into a file with dst_file_ext as the file extension.
        Returns the hash key of source code, and the path to the file.
        """

        cuda_command = repr(
            cuda_compile_command(["dummy_input"], "dummy_output", dst_file_ext)
        )
        key, input_path = write(
            source_code, cls._SOURCE_CODE_SUFFIX, extra=cuda_command
        )
        return key, input_path

    @classmethod
    def compile(
        cls, source_code, dst_file_ext, extra_args: Optional[List[str]] = None
    ) -> Tuple[str, str, str]:
        """
        Compiles CUDA source_code into a file with dst_file_ext extension.
        Returns a tuple of dst_file_path, hash_key, source_code_path
        """
        key, input_path = cls.write(source_code, dst_file_ext)
        if key not in cls.cache:
            from filelock import FileLock

            lock_dir = get_lock_dir()
            lock = FileLock(os.path.join(lock_dir, key + ".lock"), timeout=LOCK_TIMEOUT)
            with lock:
                output_path = input_path[: -len(cls._SOURCE_CODE_SUFFIX)] + dst_file_ext
                if not os.path.exists(output_path):
                    cmd = cuda_compile_command(
                        [input_path], output_path, dst_file_ext, extra_args
                    )
                    start_time = time()
                    log.debug("CUDA Compilation: %s", cmd)
                    cmd_parts = cmd.split(" ")
                    try:
                        subprocess.check_output(
                            cmd_parts, stderr=subprocess.STDOUT, env=os.environ
                        )
                    except subprocess.CalledProcessError as error:
                        raise exc.CUDACompileError(cmd_parts, error.output) from error
                    end_time = time()
                    log_duration_msg = f"CUDA Compilation took {end_time-start_time} seconds. Compile command: {cmd}"
                    log.info(log_duration_msg)
                else:
                    log.debug(
                        "CUDA Compilation skipped: %s since output already exists",
                        input_path,
                    )
                cls.cache[key] = CUDACodeCache.CacheEntry(input_path, output_path)

        return (cls.cache[key].output_path, key, input_path)

    @classmethod
    def load(cls, source_code, dst_file_ext) -> Tuple[DLLWrapper, str, str]:
        """
        Compiles source code and loads the generated .so file.
        Returns a tuple of DLLWrapper, hash_key, source_code_path
        """

        if dst_file_ext != "so":
            raise RuntimeError(
                f"Only support loading a .so file for now. "
                f"Requested file extension: {dst_file_ext}. Source code: {source_code}"
            )
        dst_file_path, hash_key, source_code_path = cls.compile(
            source_code, dst_file_ext
        )
        return (DLLWrapper(dst_file_path), hash_key, source_code_path)


def caching_device_properties():
    for _, device_interface in get_registered_device_interfaces():
        if device_interface.is_available():
            device_interface.Worker.get_device_properties()


def _set_triton_ptxas_path() -> None:
    if os.environ.get("TRITON_PTXAS_PATH") is not None:
        return
    ptxas_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "bin", "ptxas")
    )
    if not os.path.exists(ptxas_path):
        return
    if os.path.isfile(ptxas_path) and os.access(ptxas_path, os.X_OK):
        os.environ["TRITON_PTXAS_PATH"] = ptxas_path
    else:
        warnings.warn(f"{ptxas_path} exists but is not an executable")


def _worker_compile(
    kernel_name: str, source_code: str, cc: int, device: torch.device
) -> None:
    device_interface = get_interface_for_device(device.type)
    device_interface.Worker.set_device(device.index)
    kernel = TritonCodeCache.load(kernel_name, source_code)
    kernel.precompile(warm_cache_only_with_cc=cc)


def _load_kernel(kernel_name: str, source_code: str) -> ModuleType:
    _set_triton_ptxas_path()
    kernel = TritonCodeCache.load(kernel_name, source_code)
    kernel.precompile()
    return kernel


class TritonFuture:
    kernel: ModuleType

    def __init__(
        self,
        kernel_name: str,
        source_code: str,
        future: Future[Any],
    ) -> None:
        self.kernel_name = kernel_name
        self.source_code = source_code
        self.future = future

    # @dynamo_utils.dynamo_timed
    def result(self) -> ModuleType:
        t0 = time()
        if hasattr(self, "kernel"):
            return self.kernel
        # If the worker failed this will throw an exception.
        self.future.result()
        kernel = self.kernel = _load_kernel(self.kernel_name, self.source_code)
        latency = time() - t0
        if latency > 50:
            developer_warning(
                f"Detected long compilation time of {latency} seconds for kernel name {self.kernel_name}"
            )
            developer_warning(self.source_code)
        del self.kernel_name, self.source_code, self.future
        return kernel


# If this process dies abnormally (e.g. segfault)
# it will not shut down the workers. Instead
# the workers will have their parent reassigned to the
# init process. This launches a separate thread to
# watch for the worker getting reassigned,
# and cleans it up in this case.
#
# This function cannot be an inner function since otherwise mp_context="spawn" would
# not work for ProcessPoolExecutor since inner functions cannot be pickled.
def _async_compile_initializer(orig_ppid) -> None:
    def run() -> None:
        while True:
            sleep(1)
            if orig_ppid != os.getppid():
                os.kill(os.getpid(), signal.SIGKILL)

    global _watchdog_thread
    _watchdog_thread = Thread(target=run, daemon=True)
    _watchdog_thread.start()
    # Ignore Ctrl-C (i.e. SIGINT) sent to pool workers to avoid meaningless log spam.
    signal.signal(signal.SIGINT, signal.SIG_IGN)


_watchdog_thread: Optional[Thread] = None

# Used to keep track of all process pools invoked so far.
_pool_set: Set[ProcessPoolExecutor] = set()


def shutdown_compile_workers() -> None:
    """Shut down all outstanding compile-worker pools."""
    global _pool_set
    for pool in _pool_set:
        pool.shutdown()
    _pool_set.clear()


class AsyncCompile:
    def __init__(self) -> None:
        pass

    @staticmethod
    @functools.lru_cache(1)
    def pool() -> ThreadPoolExecutor:
        assert config.compile_threads > 1
        return ThreadPoolExecutor(config.compile_threads)

    @staticmethod
    @functools.lru_cache(1)
    def process_pool() -> ProcessPoolExecutor:
        # ensure properties have been calculated before processes
        # are forked
        caching_device_properties()
        assert config.compile_threads > 1
        orig_ppid = os.getpid()

        ctx = multiprocessing.get_context(config.worker_start_method)
        pool = ProcessPoolExecutor(
            config.compile_threads,
            mp_context=ctx,
            initializer=partial(_async_compile_initializer, orig_ppid),
        )

        global _pool_set
        _pool_set.add(pool)

        # when this pool is created in a subprocess object, the normal exit handler
        # doesn't run, and we need to register our own handler.
        # exitpriority has to be high, because another one of the finalizers will
        # kill the worker thread that sends the shutdown message to the workers...
        multiprocessing.util.Finalize(None, pool.shutdown, exitpriority=sys.maxsize)
        return pool

    @classmethod
    def warm_pool(cls) -> None:
        if config.compile_threads <= 1:
            return
        _compile_start()
        pool = cls.process_pool()

        # We have to fork processes for compiler workers, but the more memory and other resources that are loaded, the
        # slower the os.fork time is, quite drastically. It also holds the GIL so we can't put it on another thread.

        # Examples:
        # A simple x + x + x script: 10ms seconds in the middle of the program, 2ms at startup
        # tf_efficientnet_b0 benchmark: 50ms! in the middle of the program , 3ms at startup

        # So we want to start the workers early when it is still cheap, and also to allow the workers to get
        # ready before we have work for them.

        # ProcessPoolExecutor also does not launch the workers until it finds a point when all the workers are idle.
        # But if we waited until then fork time will be long and we will be waiting for the processes to initialize.

        # We force them to start here with some YOLOing of the internal methods.
        if hasattr(pool, "_start_queue_management_thread"):
            pool._start_queue_management_thread()
        else:
            for _ in range(config.compile_threads):
                pool._adjust_process_count()
            if hasattr(pool, "_start_executor_manager_thread"):
                pool._start_executor_manager_thread()
        _compile_end()

    @classmethod
    def submit(cls, task: Callable[..., Any]) -> Any:
        if config.compile_threads <= 1:
            return task()
        return cls.pool().submit(task)

    def triton(
        self, kernel_name: str, source_code: str, device_str: str = "cuda"
    ) -> Union[TritonFuture, ModuleType]:
        _compile_start()

        if config.compile_threads > 1:
            device_interface = get_interface_for_device(device_str)
            device = torch.device(device_str, device_interface.current_device())
            cc = device_interface.get_compute_capability(device)
            future = self.process_pool().submit(
                _worker_compile, kernel_name, source_code, cc, device
            )
            return TritonFuture(kernel_name, source_code, future)
        else:
            return _load_kernel(kernel_name, source_code)

    def multi_kernel(self, *args, **kwargs) -> Any:
        from torch._inductor.codegen.multi_kernel import MultiKernelCall

        # no need to call this in parallel since the sub-kernels are already parallel tasks
        return MultiKernelCall(*args, **kwargs)

    def cpp(self, source_code: str) -> ModuleType:
        def task():
            return CppCodeCache.load(source_code).kernel

        return self.submit(task)

    def cpp_pybinding(self, argtypes: List[str], source_code: str) -> ModuleType:
        return self.submit(
            functools.partial(
                CppPythonBindingsCodeCache.load_pybinding, argtypes, source_code
            )
        )

    def cuda(self, source_code, dst_file_ext):
        def task():
            return CUDACodeCache.load(source_code, dst_file_ext)[0]

        return self.submit(task)

    def wait(self, scope: Dict[str, Any]) -> None:
        num_kernels = len(
            [
                value
                for key, value in scope.items()
                if isinstance(value, (Future, TritonFuture))
            ]
        )
        pbar = tqdm(
            total=num_kernels,
            desc="Inductor Compilation",
            disable=config.disable_progress,
            delay=0,
        )
        if config.compile_threads > 1:
            for key, result in scope.items():
                if config.verbose_progress and not isinstance(pbar, _Faketqdm):
                    pbar.set_postfix_str(key)
                if isinstance(result, (Future, TritonFuture)):
                    scope[key] = result.result()
                    pbar.update(1)

        _compile_end()


if os.environ.get("TORCH_TNT_IN_USE", "0") == "1":
    # When TorchTNT is used, calling warm_pool() here will cause the
    # compile workers created not being able to be shut down inside
    # shutdown_compile_workers(). This may cause significant QPS drop.
    log.info("Do not call AsyncCompile.warm_pool() because TorchTNT is in use.")
elif sys.version_info >= (3, 12):
    log.info("AsyncCompile.warm_pool() is broken on 3.12+.")
else:
    AsyncCompile.warm_pool()
