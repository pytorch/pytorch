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
import os
import pickle
import pkgutil
import platform
import re
import shlex
import shutil
import struct
import subprocess
import sys
import sysconfig
import tempfile
import textwrap
import threading
import warnings
from bisect import bisect_right
from copy import copy
from ctypes import c_void_p, CDLL, cdll
from functools import partial
from pathlib import Path
from time import time, time_ns
from types import ModuleType
from typing import (
    Any,
    Callable,
    cast,
    Counter,
    Dict,
    Generator,
    List,
    NoReturn,
    Optional,
    Sequence,
    Set,
    Tuple,
    TYPE_CHECKING,
    TypeVar,
    Union,
)
from typing_extensions import TypeAlias

import torch
from torch import SymInt, Tensor
from torch._dynamo.utils import counters, dynamo_timed
from torch._inductor import config, exc, metrics
from torch._inductor.codegen.cuda import cuda_env
from torch._inductor.codegen.rocm.compile_command import (
    rocm_compile_command,
    rocm_compiler,
)

T = TypeVar("T")

from _collections_abc import dict_keys  # noqa: TCH003

"""
codecache.py, cpp_builder.py and cpu_vec_isa.py import rule:
https://github.com/pytorch/pytorch/issues/124245#issuecomment-2197778902
"""
from torch._inductor.cpp_builder import (
    _get_python_include_dirs,
    _set_gpu_runtime_env,
    _transform_cuda_paths,
    CppBuilder,
    CppOptions,
    CppTorchCudaOptions,
    get_compiler_version_info,
    get_cpp_compiler,
    get_name_and_dir_from_output_file_path,
    homebrew_libomp,
    is_apple_clang,
    is_clang,
    is_conda_llvm_openmp_installed,
    normalize_path_separator,
)
from torch._inductor.cpu_vec_isa import invalid_vec_isa, pick_vec_isa, VecISA
from torch._inductor.runtime.compile_tasks import (
    _module_to_triton_kernel,
    _reload_python_module,
    _reload_python_module_in_subproc,
)
from torch._inductor.runtime.runtime_utils import cache_dir, default_cache_dir
from torch._inductor.utils import ALIGN_BYTES, clear_on_fresh_inductor_cache, is_linux
from torch._logging import trace_structured
from torch._subclasses.fake_tensor import (
    extract_tensor_metadata,
    FakeTensor,
    TensorMetadata,
)
from torch.fx.experimental.symbolic_shapes import has_hint, hint_int, ShapeEnv


if TYPE_CHECKING:
    from concurrent.futures import Future

    from torch._inductor.graph import GraphLowering
    from torch._inductor.ir import ChoiceCaller
    from torch._inductor.runtime.hints import HalideInputSpec, HalideMeta


_HERE = os.path.abspath(__file__)
_TORCH_PATH = os.path.dirname(os.path.dirname(_HERE))
_LINKER_SCRIPT = os.path.join(_TORCH_PATH, "_inductor/script.ld")

_IS_WINDOWS = sys.platform == "win32"

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

    def log_global_cache_errors(*args: Any, **kwargs: Any) -> None:
        pass

    def log_global_cache_stats(*args: Any, **kwargs: Any) -> None:
        pass

    def log_global_cache_vals(*args: Any, **kwargs: Any) -> None:
        pass

    def use_global_cache() -> bool:
        return False


output_code_log = torch._logging.getArtifactLogger(__name__, "output_code")

LOCK_TIMEOUT = 600

_IS_WINDOWS = sys.platform == "win32"


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


def get_cpp_wrapper_cubin_path_name() -> str:
    return "cubin_path" if torch.version.hip is None else "hsaco_path"


class CacheBase:
    @staticmethod
    @functools.lru_cache(None)
    def get_system() -> Dict[str, Any]:
        try:
            from triton.compiler.compiler import triton_key

            # Use triton_key instead of triton.__version__ as the version
            # is not updated with each code change
            triton_version = triton_key()
        except ModuleNotFoundError:
            triton_version = None

        try:
            system: Dict[str, Any] = {
                "device": {"name": None},
                "version": {
                    "triton": triton_version,
                },
            }
            device_properties = torch.cuda.get_device_properties(
                torch.cuda.current_device()
            )
            if torch.version.cuda is not None:
                system["device"]["name"] = device_properties.name
                system["version"]["cuda"] = torch.version.cuda
            else:
                system["device"]["name"] = device_properties.gcnArchName
                system["version"]["hip"] = torch.version.hip
        except (AssertionError, RuntimeError):
            # If cuda is not installed, none of the above config is relevant.
            system = {}

        system["hash"] = hashlib.sha256(
            json.dumps(system, sort_keys=True).encode("utf-8")
        ).hexdigest()

        return system

    @staticmethod
    @clear_on_fresh_inductor_cache
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
        self.system = CacheBase.get_system()

    def get_local_cache(self) -> Dict[str, Any]:
        local_cache_path = self.get_local_cache_path()
        if not local_cache_path.is_file():
            return {}
        with open(local_cache_path) as local_cache_fp:
            local_cache = json.load(local_cache_fp)
        return local_cache["cache"]

    def update_local_cache(self, local_cache: Dict[str, Any]) -> None:
        local_cache_path = self.get_local_cache_path()
        write_atomic(
            str(local_cache_path),
            json.dumps({"system": self.system, "cache": local_cache}, indent=4),
            make_dirs=True,
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
    @functools.lru_cache(None)  # noqa: B019
    def get_global_cache(self) -> Dict[str, Any]:
        global_cache_path = self.get_global_cache_path()
        if global_cache_path is None or not global_cache_path.is_file():
            return {}
        with open(global_cache_path) as global_cache_fp:
            global_cache = json.load(global_cache_fp)
        return global_cache["cache"]

    def lookup(
        self,
        choices: List[ChoiceCaller],
        op: str,
        inputs: str,
        benchmark: Optional[Callable[[Any], Dict[ChoiceCaller, float]]],
    ) -> Dict[ChoiceCaller, float]:
        """
        Check to see if we have benchmarked the given choice callers. For each
        choice caller:

            1. Check global_cache[op][inputs][choice][precision], return benchmark if cached.
            2. Check local_cache[op][inputs][choice][precision], return benchmark if cached.
            3. If benchmark is not None:
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

        def check_cache(cache: Dict[str, Any], callback: Any = None) -> bool:
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
            local_cache = self.get_local_cache() if config.autotune_local_cache else {}
            # check local cache first since it is data specific to the current machine
            if (
                not check_cache(local_cache)
                and not (
                    use_global_cache()
                    and check_cache(self.get_global_cache(), callback=log_stats)
                )
                and benchmark is not None
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


def code_hash(code: Union[str, bytes], extra: str = "") -> str:
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


def get_hash(
    content: Union[str, bytes], extra: str = "", hash_type: str = "code"
) -> str:
    if hash_type == "code":
        return code_hash(content, extra)
    if hash_type in ["cubin", "hsaco", "spv"]:
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
    # hashes just because the content begins/ends with different number of
    # spaces.
    key: str = get_hash(content.strip(), extra, hash_type)
    basename, subdir, path = get_path(key, extension, specified_dir)
    if not os.path.exists(path):
        write_atomic(path, content, make_dirs=True)
    return basename, path


def write_text(text: str) -> str:
    """
    Write the `text` to a file and return the path computed based on the hash.
    """
    return write(text, "txt")[1]


def write_atomic(
    path: str, content: Union[str, bytes], make_dirs: bool = False
) -> None:
    # Write into temporary file first to avoid conflicts between threads
    # Avoid using a named temporary file, as those have restricted permissions
    assert isinstance(
        content, (str, bytes)
    ), "Only strings and byte arrays can be saved in the cache"
    path = Path(path)
    if make_dirs:
        path.parent.mkdir(parents=True, exist_ok=True)
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


def _ident(x: T) -> T:
    return x


def extract_tensor_metadata_for_cache_key(
    device_map: Dict[torch.device, torch.device], t: Tensor
) -> TensorMetadata:
    """
    Extracts the tensor metadata and removes fields of the TensorMetadata
    that are not needed for caching
    """
    meta = extract_tensor_metadata(t)
    if not hasattr(t, "_is_inductor_static"):
        meta = dataclasses.replace(meta, storage_offset=0, storage_bytes=None)

    # The pickle implementation avoids serializing the same object more than once.
    # That behavior means the byte stream we create to hash will vary if, for example,
    # we see two tensor objects with the same device, but the torch.device object is
    # actually the same object vs. merely equivalent. We want to produce the same hash
    # value in either situation, so we memoize the device objects and always reference
    # the same object for a given device. It's possible other metadata fields deserve
    # the same treatment, but so far we've only observed this issue with the device.
    if meta.device not in device_map:
        device_map[meta.device] = meta.device
    meta = dataclasses.replace(meta, device=device_map[meta.device])

    return meta


def _reduce_fake_tensor(
    device_map: Dict[torch.device, torch.device], t: Tensor
) -> Tuple[Callable[[T], T], Tuple[TensorMetadata]]:
    """
    See FxGraphCachePickler. Custom reducer to pickle FakeTensors.
    """
    metadata = extract_tensor_metadata_for_cache_key(device_map, t)
    return (_ident, (metadata,))


def _reduce_tensor(
    device_map: Dict[torch.device, torch.device], t: Tensor
) -> Tuple[Callable[[T], T], Tuple[TensorMetadataAndValues]]:
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
        raise BypassFxGraphCache

    # Very large tensors could be expensive to copy to cpu and hash. Let's
    # at least report if we find slowness.
    start = time()
    values = t.tolist()
    elapsed = time() - start
    if elapsed > 1.0:
        warnings.warn(
            f"FX graph cache handling of a large constant took {elapsed:.1}s. Please file an issue."
        )

    metadata = extract_tensor_metadata_for_cache_key(device_map, t)
    return (_ident, (TensorMetadataAndValues(metadata, values),))


def _reduce_symint(s: SymInt) -> Tuple[Callable[[T], T], Tuple[str]]:
    """
    See FxGraphCachePickler. Custom reducer to pickle SymInts.
    """
    # For hashing purposes, we only care about the name of the symbol and
    # not the backed value. We evaluate guards stored with a cached graph
    # to ensure a cached entity with SymInt args is safe to reuse.
    return (_ident, (str(s),))


def _reduce_unsupported(s: Any) -> NoReturn:
    """
    See FxGraphCachePickler. Custom reducer to handle any objects that we don't
    support and therefore raise to bypass caching.
    """
    raise BypassFxGraphCache


class FxGraphCachePickler(pickle.Pickler):
    """
    Custom pickler to customize the pickling of some objects (Tensors), only for the
    purpose of computing a hash for keying into the FxGraphCache. Tensors contain
    objects that don't pickle and/or vary between runs, and we want to capture the
    data that allow us to compute a stable, but safe hash.
    """

    # See extract_tensor_metadata_for_cache_key. Whenever we extract metadata during
    # pickling, we make sure devices always reference the same torch.device object.
    _device_map: Dict[torch.device, torch.device] = {}

    dispatch_table = copyreg.dispatch_table.copy()
    dispatch_table[FakeTensor] = functools.partial(_reduce_fake_tensor, _device_map)
    dispatch_table[torch.Tensor] = functools.partial(_reduce_tensor, _device_map)
    dispatch_table[torch.SymInt] = _reduce_symint
    dispatch_table[
        torch.fx.experimental._backward_state.BackwardState
    ] = _reduce_unsupported

    @classmethod
    def dumps(cls, obj: Any) -> bytes:
        """
        Pickle an object using the FxGraphCachePickler.
        """
        with io.BytesIO() as stream:
            pickler = cls(stream)
            # TODO: pickler.fast is technically deprecated. Will this work on new python versions?
            pickler.fast = True  # Run with pickler.fast so it doesn't intern strings, making the hash result more predictable
            try:
                pickler.dump(obj)
            except (TypeError, AttributeError) as e:
                # Some configs options are callables, e.g., post_grad_custom_pre_pass,
                # and may not pickle.
                log.warning("Can't pickle", exc_info=True)
                raise BypassFxGraphCache from e
            return stream.getvalue()

    @classmethod
    def get_hash(cls, obj: Any) -> str:
        """
        Serialize an object using the FxGraphCachePickler and return a hash
        of the pickled object.
        """
        serialized_data = cls.dumps(obj)
        return sha256_hash(serialized_data)

    @classmethod
    def debug_lines(cls, inp: FxGraphHashDetails) -> List[str]:
        """
        Get a printable string describing in more detail all the attributes
        comprising an object. Useful for debugging when one graph hashes
        to a different value than another.
        """

        def get_str(obj: Any) -> str:
            if isinstance(obj, torch.Tensor):
                return str(extract_tensor_metadata_for_cache_key(cls._device_map, obj))
            elif isinstance(obj, bytes):
                return "<bytes>"
            elif type(obj) in cls.dispatch_table:
                # Run the reducer on the object
                return str(cls.dispatch_table[type(obj)](obj)[1])
            else:
                return str(obj)

        lines = []
        for attr, obj in vars(inp).items():
            if isinstance(obj, list):
                for ii in range(len(obj)):
                    h = cls.get_hash(obj[ii])
                    lines.append(f"[{h}] {attr}[{ii}]: {get_str(obj[ii])}")
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    h = cls.get_hash(v)
                    lines.append(f"[{h}] {attr}[{k}]: {get_str(v)}")
            else:
                h = cls.get_hash(obj)
                lines.append(f"[{h}] {attr}: {get_str(obj)}")
        return lines


def build_code_hash(
    roots: List[str] | None, prefix: str, hasher: hashlib._Hash
) -> None:
    for lib in sorted(pkgutil.iter_modules(roots, prefix), key=lambda x: x.name):
        spec = lib.module_finder.find_spec(lib.name, None)
        assert spec is not None
        module = spec.origin
        assert module is not None
        with open(module, "rb") as f:
            hasher.update(spec.name.encode("utf-8"))
            hasher.update(f.read())
        if lib.ispkg:
            # need to also hash submodules
            build_code_hash(spec.submodule_search_locations, f"{spec.name}.", hasher)


@functools.lru_cache(None)
def torch_key() -> bytes:
    """
    Compute a key that contains relevant information about torch source files
    """
    if not config.is_fbcode():

        def get_code_hash(root: str) -> bytes:
            # This function isn't meant to be used outside of torch_key, just a
            # helper for clarity. Instead, use torch_key() directly when you need
            # a hash representing the state of the source code.
            extra_files = (
                "codegen/aoti_runtime/interface.cpp",
                "codegen/aoti_runtime/implementation.cpp",
                "codegen/cpp_prefix.h",
                "script.ld",
            )
            inductor_root = os.path.dirname(__file__)
            extra_files = [os.path.join(inductor_root, x) for x in extra_files]
            hasher = hashlib.sha256()
            hasher.update(torch.__version__.encode("utf-8"))
            build_code_hash([root], "", hasher)
            for path in extra_files:
                if os.path.exists(path):
                    with open(path, "rb") as f:
                        hasher.update(f.read())
            return hasher.digest()

        return get_code_hash(_TORCH_PATH)

    from libfb.py import parutil

    return parutil.get_file_contents("torch/src_hash.txt").rstrip()


def get_inductor_root() -> str:
    return os.path.dirname(__file__)


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
        inputs_to_check: Sequence[int],
    ) -> None:
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

        # Alignment checks
        self.inputs_to_check = inputs_to_check

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
        self.torch_version = torch_key()
        self.system_info = CacheBase.get_system()
        self.inductor_config = config.save_config_portable()

    def debug_lines(self) -> List[str]:
        """
        Get a printable string describing in more detail all the attributes
        comprising this object. Useful for debugging when one graph hashes
        to a different value than another.
        """
        return FxGraphCachePickler.debug_lines(self)


def compiled_fx_graph_hash(
    gm: torch.fx.GraphModule,
    example_inputs: List[torch.Tensor],
    fx_kwargs: Dict[str, Any],
    inputs_to_check: Sequence[int],
) -> Tuple[str, List[str]]:
    """
    Generate a unique hash of the FX graph for caching.
    """
    details = FxGraphHashDetails(gm, example_inputs, fx_kwargs, inputs_to_check)
    # The prefix distinguishes among the other kinds of objects we
    # cache in this module.
    key = "f" + FxGraphCachePickler.get_hash(details)
    debug_lines = details.debug_lines()
    debug_str = "\n".join(debug_lines)
    log.debug(f"FX graph cache hash details for key {key}:\n{debug_str}")  # noqa: G004
    return key, debug_lines


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
    def _filter_backed_symints(inputs: List[Any]) -> List[torch.SymInt]:
        """
        Get the backed SymInt objects from the input list. Note that we can never
        have guards that depend on unbacked symint.
        """
        return [s for s in inputs if isinstance(s, torch.SymInt) and has_hint(s)]

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
        local: bool,
        remote_cache: Optional[Any],
    ) -> Optional[CompiledFxGraph]:
        """
        Lookup a compiled graph in the cache by key. On a hit, return the
        deserialized CompiledFxGraph object. On a miss, return None.
        """
        shape_env = FxGraphCache._get_shape_env()
        assert shape_env is not None

        symints = FxGraphCache._filter_backed_symints(example_inputs)
        hints = [hint_int(s) for s in symints]

        def iterate_over_candidates() -> Generator[CompiledFxGraph, None, None]:
            if local:
                subdir = FxGraphCache._get_tmp_dir_for_key(key)
                if os.path.exists(subdir):
                    for path in sorted(os.listdir(subdir)):
                        try:
                            with open(os.path.join(subdir, path), "rb") as f:
                                yield pickle.load(f)
                        except Exception:
                            log.warning(
                                "fx graph cache unable to load compiled graph",
                                exc_info=True,
                            )

            if remote_cache:
                try:
                    if (data := remote_cache.get(key)) is not None:
                        yield pickle.loads(data)
                except Exception:
                    log.warning(
                        "fx graph cache unable to load compiled graph", exc_info=True
                    )

        # Iterate over any entries in the subdir for this key and evaluate
        # their guards to determine whether there's a hit.
        graph = None

        for candidate in iterate_over_candidates():
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
        artifact_path = get_path(graph.cache_key, "py")[2]
        code = graph.source_code
        if not os.path.exists(artifact_path):
            counters["inductor"]["fxgraph_lookup_write_file"] += 1
            Path(os.path.dirname(artifact_path)).mkdir(parents=True, exist_ok=True)
            cpp_pp = cpp_prefix_path()
            if os.path.basename(cpp_pp) in code:
                if cpp_pp in code:
                    # Great the name is correct
                    pass
                else:
                    # Old dir name is included, replace it
                    pattern = rf'#include\s*"[^"]+{os.path.basename(cpp_pp)}"'
                    code = re.sub(pattern, f'#include "{cpp_pp}"', code)

            write_atomic(artifact_path, code, make_dirs=True)

        try:
            graph.current_callable = PyCodeCache.load_by_key_path(
                graph.cache_key,
                artifact_path,
                graph.cache_linemap,
                graph.constants,
            ).call
        except OSError:
            # Not expected, but in case the PyCodeCache entry is removed from
            # underneath us, treat it as a cache miss and recompile.
            log.error("Failed to load cached artifact: %s", artifact_path)
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

        # Increment the cached metrics/counters by the amounts recorded when the FX
        # graph was compiled for this cache entry. Pretending these counters
        # were incremented normally is useful for testing with the cache enabled.
        metrics.CachedMetricsHelper.apply_deltas(graph.metrics_deltas)
        counters["inductor"] += graph.counter_deltas

        from .graph import GraphLowering

        GraphLowering.save_output_code(code)
        output_code_log.debug("Output code: \n%s", code)
        # On cache hit, use artifact path as filename
        trace_structured(
            "inductor_output_code",
            lambda: {"filename": artifact_path},
            payload_fn=lambda: code,
        )

        return graph

    @staticmethod
    def _save_graph(
        key: str,
        compiled_graph: CompiledFxGraph,
        example_inputs: List[torch.Tensor],
        time_taken_ns: int,
        local: bool,
        remote_cache: None,
    ) -> None:
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
        symints = FxGraphCache._filter_backed_symints(example_inputs)
        guards = shape_env.get_pruned_guards(symints)
        disk_compiled_graph.guards_expr = shape_env.produce_guards_expression(
            placeholders=symints, guards=guards
        )

        try:
            content = pickle.dumps(disk_compiled_graph)
        except Exception:
            log.warning(
                "fx graph cache unable to serialize compiled graph", exc_info=True
            )
            counters["inductor"]["fxgraph_cache_pickle_error"] += 1
            return

        try:
            if local:
                subdir = FxGraphCache._get_tmp_dir_for_key(key)
                if not os.path.exists(subdir):
                    os.makedirs(subdir, exist_ok=True)

                # Use a hash of the serialized CompiledFxGraph to get a unique file
                # name. The specific name doesn't matter since a lookup involves
                # iterating over all entries in the parent subdir.
                path = os.path.join(subdir, sha256_hash(content))
                write_atomic(path, content, make_dirs=True)

            if remote_cache:
                cache_data = (
                    {
                        "data": content,
                        "time_taken_ms": time_taken_ns
                        // 1000000,  # Convert from NS to MS
                    }
                    if config.is_fbcode()
                    else content
                )
                remote_cache.put(key, cache_data)
        except Exception:
            log.warning("fx graph unable to write to cache", exc_info=True)
            counters["inductor"]["fxgraph_cache_write_error"] += 1

    @staticmethod
    def _check_can_cache(gm: torch.fx.GraphModule) -> None:
        """
        Check some conditions that would preclude caching and raise BypassFxGraphCache
        to bypass in case caching is not possible.
        """
        # Freezing can embed constants that wouldn't be static across runs.
        if config.freezing or config.aot_inductor.use_runtime_constant_folding:
            raise BypassFxGraphCache

        # The treatment of guards in the caching implementation requires that
        # we have a shape env.
        if FxGraphCache._get_shape_env() is None:
            log.debug("fx graph cache no shape env")
            raise BypassFxGraphCache

        # HigherOrderOperators should be handled on a case-by-case basis.
        # Currently, we just skip caching if we have any.
        # We also skip if there are any torchbind objects.
        for node in gm.graph.nodes:
            if isinstance(node.target, torch._ops.HigherOrderOperator):
                raise BypassFxGraphCache
            if node.op == "getattr" and isinstance(
                getattr(gm, node.target), torch._C.ScriptObject
            ):
                raise BypassFxGraphCache

    @staticmethod
    def load(  # type: ignore[no-untyped-def]
        compile_fx_fn: Callable[..., Any],
        gm: torch.fx.GraphModule,
        example_inputs: List[torch.Tensor],
        fx_kwargs: Dict[str, Any],
        inputs_to_check: Sequence[int],
        local: bool,
        remote: bool,
    ):
        """
        Load a compiled graph from the cache. If a cached entry does not exist,
        compile the graph and save it to the cache.
        """
        assert local or remote, "at least one of them needs to be enabled"
        compiled_graph = None
        cache_state = None
        key = None
        debug_lines = None
        try:
            FxGraphCache._check_can_cache(gm)
            key, debug_lines = compiled_fx_graph_hash(
                gm, example_inputs, fx_kwargs, inputs_to_check
            )

            remote_cache = None
            if remote:
                cache_id = "fx-graph-v1"
                try:
                    if config.is_fbcode():
                        from torch._inductor.fb.remote_cache import (
                            FbRemoteFxGraphCacheBackend,
                        )

                        remote_cache = FbRemoteFxGraphCacheBackend(cache_id)
                    else:
                        from torch._inductor.remote_cache import RedisRemoteCacheBackend

                        remote_cache = RedisRemoteCacheBackend(cache_id)
                except Exception:
                    remote_cache = None
                    log.warning("Unable to create a remote cache", exc_info=True)

            compiled_graph = FxGraphCache._lookup_graph(
                key, example_inputs, local, remote_cache
            )

            if compiled_graph is None:
                log.debug("fx graph cache miss for key %s", key)
                counters["inductor"]["fxgraph_cache_miss"] += 1
                cache_state = "miss"
                start_time = time_ns()
                compiled_graph = compile_fx_fn(gm, example_inputs, **fx_kwargs)
                time_taken_ns = time_ns() - start_time
                FxGraphCache._save_graph(
                    key,
                    compiled_graph,
                    example_inputs,
                    time_taken_ns,
                    local,
                    remote_cache,
                )
            else:
                log.debug("fx graph cache hit for key %s", key)
                counters["inductor"]["fxgraph_cache_hit"] += 1
                cache_state = "hit"
            compiled_graph._fx_graph_cache_key = key
        except BypassFxGraphCache:
            counters["inductor"]["fxgraph_cache_bypass"] += 1
            cache_state = "bypass"
            if not compiled_graph:
                compiled_graph = compile_fx_fn(gm, example_inputs, **fx_kwargs)

        torch._logging.trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "fx_graph_cache_hash",
                "encoding": "json",
            },
            payload_fn=lambda: json.dumps(
                {"key": key, "cache_state": cache_state, "components": debug_lines}
            ),
        )

        return compiled_graph

    @staticmethod
    def clear() -> None:
        """
        Clear out the on-disk cache.
        """
        try:
            shutil.rmtree(FxGraphCache._get_tmp_dir())
        except FileNotFoundError:
            pass


_StrideExprStr: TypeAlias = str


@dataclasses.dataclass
class CompiledFxGraph:
    """
    Class holding a compiled FX graph. This is the object serialized on disk
    to support FxGraph caching.
    """

    current_callable: Optional[Callable[..., Any]]
    cache_key: str
    source_code: str = dataclasses.field(repr=False)  # Do not display source_code
    cache_linemap: Optional[List[Tuple[int, str]]]
    device_types: Set[str]
    device_idxs: Set[int]
    mutated_inputs: Set[str]
    mutated_input_idxs: Set[int]
    constants: Dict[str, torch.Tensor]
    torchbind_constants: Dict[str, torch._C.ScriptObject]
    output_strides: Optional[List[Optional[Tuple[_StrideExprStr, ...]]]]
    disabled_cudagraphs_reason: Optional[str]
    metrics_deltas: metrics.CachedMetricsDeltas
    counter_deltas: Counter[str]
    # This is a string representation of an expression we serialize
    # with the object so the guards can be evaluated in a different
    # context in order to verify the validity of serving a cached
    # fx graph. The expression must be generated by:
    # ShapeEnv.produce_guards_expression()
    guards_expr: Optional[str]

    _boxed_call: Optional[bool] = None
    _fx_graph_cache_key: Optional[str] = None

    def __init__(
        self,
        current_callable: Optional[Callable[..., Any]],
        graph: GraphLowering,
        output_strides: List[Optional[Tuple[_StrideExprStr, ...]]],
        disabled_cudagraphs_reason: Optional[str],
        metrics_deltas: metrics.CachedMetricsDeltas,
        counter_deltas: Counter[str],
    ) -> None:
        self.current_callable = current_callable
        self.cache_key = graph.cache_key
        if graph.cache_path:
            with open(graph.cache_path) as f:
                self.source_code = f.read()
        self.cache_linemap = graph.cache_linemap
        self.device_types = graph.device_types
        self.device_idxs = graph.device_idxs
        self.mutated_inputs = graph.mutated_inputs
        self.mutated_input_idxs = set(graph.mutated_input_idxs)
        self.constants = graph.constants
        self.torchbind_constants = graph.torchbind_constants
        self.output_strides = output_strides
        self.disabled_cudagraphs_reason = disabled_cudagraphs_reason
        self.metrics_deltas = metrics_deltas
        self.counter_deltas = counter_deltas
        self.guards_expr = None

    def __call__(self, inputs: List[Any]) -> Any:
        assert self.current_callable is not None
        return self.current_callable(inputs)


"""
TODO: will remove old cpp builder when we switch to the new one.
"""


def get_compile_only(compile_only: bool = True) -> str:
    return "-c" if compile_only else ""


def get_shared(shared: bool = True, compile_only: bool = False) -> str:
    if not shared:
        return ""
    if compile_only:
        return "-fPIC"
    if platform.system() == "Darwin" and "clang" in get_cpp_compiler():
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
    return "-D TORCH_INDUCTOR_CPP_WRAPPER"


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
        # TODO: this is to avoid FC breakage for fbcode. When using newly
        # generated model.so on an older verion of PyTorch, need to use
        # the v1 version for aoti_torch_create_tensor_from_blob
        create_tensor_from_blob_v1 = "-D AOTI_USE_CREATE_TENSOR_FROM_BLOB_V1"
        openmp_lib = build_paths.openmp_lib()
        preprocessor_flags = " ".join(
            (
                "-D C10_USE_GLOG",
                "-D C10_USE_MINIMAL_GLOG",
                "-D C10_DISABLE_TENSORIMPL_EXTENSIBILITY",
            )
        )
        return f"-Wp,-fopenmp {openmp_lib} {preprocessor_flags} {create_tensor_from_blob_v1}"
    else:
        return ""


def use_standard_sys_dir_headers() -> str:
    if config.is_fbcode():
        return "-nostdinc"
    else:
        return ""


def get_include_and_linking_paths(
    include_pytorch: bool = False,
    vec_isa: VecISA = invalid_vec_isa,
    cuda: bool = False,
    aot_mode: bool = False,
) -> Tuple[List[str], str, str, str, str]:
    _set_gpu_runtime_env()
    from torch.utils import cpp_extension

    # Remove below in the further
    # macros = "-D {}".format(vec_isa.build_macro()) if vec_isa != invalid_vec_isa else ""
    macros = ""
    if vec_isa != invalid_vec_isa:
        for x in vec_isa.build_macro():
            macros_def = f"-D {x} "
            macros += macros_def

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
        ipaths = cpp_extension.include_paths(cuda) + _get_python_include_dirs()
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
                if cuda and torch.version.hip is None:
                    _transform_cuda_paths(lpaths)
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
                if config.is_fbcode():
                    libs += ["amdhip64"]
                else:
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
        ipaths = cpp_extension.include_paths(cuda) + _get_python_include_dirs()
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

        # For AOT mode, the produced library relies on torch cpu to set grad mode
        # like aoti_torch_grad_mode_set_enabled
        if aot_mode and sys.platform == "linux" and not config.is_fbcode():
            libs += ["torch", "torch_cpu"]
            lpaths += [cpp_extension.TORCH_LIB_PATH]

    # Unconditionally import c10 for non-abi-compatible mode to use TORCH_CHECK - See PyTorch #108690
    if not config.abi_compatible:
        libs += ["c10"]
        lpaths += [cpp_extension.TORCH_LIB_PATH]

    # third party libs
    if config.is_fbcode():
        # Note that the order of include paths do matter, as a result
        # we need to have several branches interleaved here
        if torch.version.hip is None:
            ipaths.append(build_paths.sleef())
        ipaths.append(build_paths.openmp())
        ipaths.append(build_paths.python())
        if torch.version.hip is not None:
            ipaths.append(build_paths.clang_include())
            ipaths.append(build_paths.gcc_include())
            ipaths.append(build_paths.gcc_install_tools_include())
        else:
            ipaths.append(build_paths.cc_include())
            ipaths.append(build_paths.libgcc())
            ipaths.append(build_paths.libgcc_arch())
        ipaths.append(build_paths.libgcc_backward())
        ipaths.append(build_paths.glibc())
        ipaths.append(build_paths.linux_kernel())
        if torch.version.hip is not None:
            ipaths.append(build_paths.rocm())
        else:
            ipaths.append(os.path.join(build_paths.cuda(), "include"))
        # We also need to bundle includes with absolute paths into a remote directory
        # (later on, we copy the include paths from cpp_extensions into our remote dir)
        ipaths.append("include")

    static_link_libs = []
    if aot_mode and cuda and config.is_fbcode():
        # For Meta internal cuda-12, it is recommended to static link cudart
        if torch.version.hip is None:
            static_link_libs = ["-Wl,-Bstatic", "-lcudart_static", "-Wl,-Bdynamic"]

    lpaths_str = " ".join(["-L" + p for p in lpaths])
    libs_str = " ".join(static_link_libs + ["-l" + p for p in libs])
    return ipaths, lpaths_str, libs_str, macros, build_arch_flags


def deprecated_cpp_compile_command(
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
    use_mmap_weights: bool = False,
    extra_flags: Sequence[str] = (),
) -> str:
    """
    Please don't use this function in new development code.
    It was planed to remove after we switched to new cpp_builder, but I can't access to Meta
    internal environment to fix AotCodeCompiler fb_code.
    TODO: need some Meta employee help on fix AotCodeCompiler fb_code, and then delete this
    deprecated function.
    """
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
    if use_mmap_weights:
        macros += " -D USE_MMAP_SELF"

    return re.sub(
        r"[ \n]+",
        " ",
        f"""
            {get_cpp_compiler()} {inp_name_str} {get_shared(shared, compile_only)}
            {get_warning_all_flag(warning_all)} {cpp_flags()}
            {get_glibcxx_abi_build_flags()}
            {ipaths_str} {lpaths} {libs} {build_arch_flags}
            {macros} {linker_paths} {clang_flags}
            {optimization_flags()} {cpp_wrapper_flags()}
            {use_custom_generated_macros()}
            {use_fb_internal_macros()}
            {use_standard_sys_dir_headers()}
            {get_compile_only(compile_only)}
            {' '.join(extra_flags)}
            -o {out_name}
        """,
    ).strip()


def _temp_validate_new_and_old_command(new_cmd: List[str], old_cmd: List[str]) -> None:
    """
    TODO: Will remove the temp code after switch to new cpp_builder
    """
    new_diff: List[str] = [x for x in new_cmd if x not in old_cmd]
    old_diff: List[str] = [y for y in old_cmd if y not in new_cmd]
    if new_diff or old_diff:
        print("!!! new_cmd: ", new_cmd)
        print("!!! old_cmd: ", old_cmd)
        print("!!! new_diff: ", new_diff)
        print("!!! old_diff: ", old_diff)
        raise RuntimeError("Error in new and old command different.")


def run_command_and_check(cmd: str) -> None:
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
    elif path.endswith(".pt2"):
        return os.path.split(path)
    else:
        return path, ""


@clear_on_fresh_inductor_cache
class CudaKernelParamCache:
    cache: Dict[str, Dict[str, str]] = {}
    cache_clear = staticmethod(cache.clear)

    @classmethod
    def set(cls, key: str, params: Dict[str, str], cubin: str, bin_type: str) -> None:
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
    def get_keys(cls) -> dict_keys[str, Dict[str, str]]:
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
        if sys.platform == "win32":
            raise RuntimeError("AotCodeCompiler not yet supported for inductor")

        _set_gpu_runtime_env()  # cpp_extension consults the env

        picked_vec_isa = pick_vec_isa()
        vec_isa_cmd_gen = CppBuilder(
            name="o",
            sources="i",
            BuildOption=CppTorchCudaOptions(
                vec_isa=picked_vec_isa,
                cuda=cuda,
                aot_mode=graph.aot_mode,
            ),
        )
        # write function will calc source_code hash, the same source code with different
        # ISA level should be generate different hash.
        # So we need get a command_line which contains isa related parameter as a part of hash key.
        # And then pass the command_line to below write function as extra parameter to
        # guarantee the source code hash contains ISA difference.
        cpp_command = repr(vec_isa_cmd_gen.get_command_line())

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
        trace_structured(
            "graph_dump",
            lambda: {
                "name": "inductor_aot_code",
                "type": "cpp",
                "filename": input_path,
            },
            payload_fn=lambda: source_code,
        )

        def _compile_consts_linux(consts: bytes) -> str:
            _, consts_path = write(
                consts,
                "bin",
                specified_dir=os.path.split(input_path)[0],
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

            if graph.mutated_buffers & set(graph.constants.keys()):
                # .data section is between .text and .bss. When the size of .data is large,
                # during the linking, the relocation of .text against .bss may overflow.
                # Rename it to .ldata so that it won't be in between the .text and .bss section
                if len(consts) > 2_000_000_000:
                    raise ValueError(
                        "Models with buffer mutation included doesn't support constants greater than 2GB!"
                    )
                rename_data = " .data=.ldata"
            else:
                # if no buffer mutation is needed, we could instead set the data region
                # as read-only (i.e. .lrodata) which could accomodate larger size of data
                # to be linked.
                rename_data = " .data=.lrodata,alloc,load,readonly,data,contents"

            assert (
                ALIGN_BYTES & (ALIGN_BYTES - 1)
            ) == 0 and ALIGN_BYTES >= 64, "must be power of 2 and >= 64"
            cmd = (
                f"{objcopy_command} --rename-section"
                f"{rename_data}"
                f" --set-section-alignment .data={ALIGN_BYTES}"  # following the gAlignment of CPU in c10/core/alignment.h
                f" {consts_o} {consts_o}"
            )
            log.debug("aot constant rename section command: %s", cmd)
            run_command_and_check(cmd)

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
            if config.aot_inductor.debug_dump_consts_bin:
                _, _binary_constants_path = write(
                    consts,
                    "bin",
                    specified_dir=os.path.split(input_path)[0],
                )
                log.debug("binary constants path: %s", _binary_constants_path)

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
                specified_dir=os.path.split(input_path)[0],
            )
            consts_o = os.path.splitext(consts_path)[0] + ".o"
            cmd = f"{get_cpp_compiler()} -c -o {consts_o} {consts_path}"
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
            if serialized_extern_kernel_nodes:
                output_json = os.path.splitext(input_path)[0] + ".json"
                with open(output_json, "w") as f:
                    f.write(serialized_extern_kernel_nodes)

            output_so = (
                config.aot_inductor.output_path
                if specified_so_name
                else os.path.splitext(input_path)[0] + ".so"
            )
            output_o = os.path.splitext(input_path)[0] + ".o"

            consts_size = sum(
                torch.ops.mkldnn._nbytes(tensor)
                if tensor.is_mkldnn
                else tensor.untyped_storage().nbytes()
                for (name, tensor) in graph.constants.items()
                if name not in graph.folded_constants
            )
            # TODO: Fix mmap weights with cuda
            use_mmap_weights = not config.is_fbcode() and consts_size > 2_000_000_000
            if config.aot_inductor.force_mmap_weights:
                use_mmap_weights = True

            if config.aot_inductor.package:
                (
                    object_output_name,
                    object_output_dir,
                ) = get_name_and_dir_from_output_file_path(input_path)
                object_build_options = CppTorchCudaOptions(
                    vec_isa=picked_vec_isa,
                    cuda=cuda,
                    aot_mode=graph.aot_mode,
                    compile_only=True,
                    use_absolute_path=use_absolute_path,
                    use_mmap_weights=use_mmap_weights,
                )
                object_builder = CppBuilder(
                    name=object_output_name,
                    sources=input_path,
                    output_dir=object_output_dir,
                    BuildOption=object_build_options,
                )
                compile_cmd = object_builder.get_command_line()
                output_o = object_builder.get_target_file_path()

                compile_flags = os.path.splitext(input_path)[0] + "_compile_flags.json"
                object_build_options.save_flags_to_file(compile_flags)

            else:
                (
                    object_output_name,
                    object_output_dir,
                ) = get_name_and_dir_from_output_file_path(input_path)
                object_build_options = CppTorchCudaOptions(
                    vec_isa=picked_vec_isa,
                    cuda=cuda,
                    aot_mode=graph.aot_mode,
                    compile_only=True,
                    use_absolute_path=use_absolute_path,
                    use_mmap_weights=use_mmap_weights,
                )
                object_builder = CppBuilder(
                    name=object_output_name,
                    sources=input_path,
                    output_dir=object_output_dir,
                    BuildOption=object_build_options,
                )
                compile_cmd = object_builder.get_command_line()
                output_o = object_builder.get_target_file_path()

                # TODO: replace this with using the CppBuilder above
                compile_cmd_old = deprecated_cpp_compile_command(
                    input=input_path,
                    output=output_o,
                    vec_isa=picked_vec_isa,
                    cuda=cuda,
                    aot_mode=graph.aot_mode,
                    compile_only=True,
                    use_absolute_path=use_absolute_path,
                    use_mmap_weights=use_mmap_weights,
                )
                # TODO: Enable below code to debug in fb_code.
                """
                _temp_validate_new_and_old_command(
                    compile_cmd.split(" "), compile_cmd_old.split(" ")
                )
                """
                compile_cmd = compile_cmd_old

                log.debug("aot compilation command: %s", compile_cmd)
                if fbcode_aot_cpu_re:
                    compile_file(input_path, output_o, compile_cmd.split())
                    os.chmod(output_o, 0o644)
                else:
                    run_command_and_check(compile_cmd)

            def _to_bytes(t: torch.Tensor, all_cuda: bool) -> bytes:
                def _pad_to_alignment(raw_bytes: bytes) -> bytes:
                    padded_bytes = raw_bytes.ljust(
                        (len(raw_bytes) + ALIGN_BYTES - 1) // ALIGN_BYTES * ALIGN_BYTES,
                        b"\x00",
                    )
                    return padded_bytes

                # This serializes the tensor's untyped_storage to bytes by accessing
                # the raw data of the underlying structure.
                import ctypes

                if t.numel() == 0:
                    return b""

                if t.is_mkldnn:
                    data_ptr = torch.ops.mkldnn.data_ptr(t)
                    nbytes = torch.ops.mkldnn._nbytes(t)
                else:
                    t_cpu = t.untyped_storage().cpu()
                    data_ptr = t_cpu.data_ptr()
                    nbytes = t_cpu.nbytes()

                raw_array = ctypes.cast(
                    data_ptr,
                    ctypes.POINTER(ctypes.c_ubyte * nbytes),
                )
                raw_bytes = bytes(raw_array.contents)
                return raw_bytes if all_cuda else _pad_to_alignment(raw_bytes)

            all_cuda = all(
                graph.get_original_value_of_constant(name).is_cuda
                for name in graph.constants.keys()
                if name not in graph.folded_constants
            )
            serialized_weights = b"".join(
                _to_bytes(graph.get_original_value_of_constant(name), all_cuda)
                for name in graph.constants.keys()
                if name not in graph.folded_constants
            )
            if not use_mmap_weights:
                aot_constants = serialized_weights
                magic_number = 0
            else:
                magic_number = cast(
                    int, torch.randint(0, torch.iinfo(torch.int64).max, (1,)).item()
                )
                aot_constants = struct.pack("qq", consts_size + 8, magic_number)

            consts_o = {
                "linux": _compile_consts_linux,
                "darwin": _compile_consts_darwin,
            }[sys.platform](aot_constants)

            if config.aot_inductor.package:
                output_name, output_dir = get_name_and_dir_from_output_file_path(
                    output_so
                )
                so_build_options = CppTorchCudaOptions(
                    vec_isa=picked_vec_isa,
                    cuda=cuda,
                    aot_mode=graph.aot_mode,
                    use_absolute_path=use_absolute_path,
                )
                so_builder = CppBuilder(
                    name=output_name,
                    sources=[output_o, consts_o],
                    output_dir=output_dir,
                    BuildOption=so_build_options,
                )
                link_cmd = so_builder.get_command_line()
                output_so = so_builder.get_target_file_path()

                linker_flags = os.path.splitext(input_path)[0] + "_linker_flags.json"
                so_build_options.save_flags_to_file(linker_flags)

                from torch._inductor.package import package_aoti

                if use_mmap_weights:
                    weight_file = (
                        os.path.splitext(input_path)[0] + "_serialized_weights.bin"
                    )
                    with open(weight_file, "wb") as f_weights:
                        f_weights.write(serialized_weights)
                        f_weights.write(struct.pack("q", magic_number))

                archive_path = package_aoti(os.path.split(input_path)[0])
                return archive_path

            output_name, output_dir = get_name_and_dir_from_output_file_path(output_so)
            so_build_options = CppTorchCudaOptions(
                vec_isa=picked_vec_isa,
                cuda=cuda,
                aot_mode=graph.aot_mode,
                use_absolute_path=use_absolute_path,
            )
            so_builder = CppBuilder(
                name=output_name,
                sources=[output_o, consts_o],
                output_dir=output_dir,
                BuildOption=so_build_options,
            )
            link_cmd = so_builder.get_command_line()
            output_so = so_builder.get_target_file_path()

            # TODO: replace this with using the CppBuilder above
            link_cmd_old = deprecated_cpp_compile_command(
                input=[output_o, consts_o],
                output=output_so,
                vec_isa=picked_vec_isa,
                cuda=cuda,
                aot_mode=graph.aot_mode,
                use_absolute_path=use_absolute_path,
            )
            # TODO: Enable below code to debug in fb_code.
            """
            _temp_validate_new_and_old_command(
                link_cmd.split(" "), link_cmd_old.split(" ")
            )
            """
            link_cmd = link_cmd_old

            log.debug("aot linkage command: %s", link_cmd)
            if fbcode_aot_cpu_re:
                compile_file([output_o, consts_o], output_so, link_cmd.split())
                os.chmod(output_so, 0o755)
            else:
                run_command_and_check(link_cmd)

            if use_mmap_weights:
                with open(output_so, "a+b") as f_so:
                    so_size = f_so.tell()
                    # Page align the weights
                    f_so.write(b" " * (16384 - so_size % 16384))
                    f_so.write(serialized_weights)
                    f_so.write(struct.pack("q", magic_number))

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
@clear_on_fresh_inductor_cache
@functools.lru_cache
def cpp_prefix_path() -> str:
    path = Path(__file__).parent / "codegen/cpp_prefix.h"
    with path.open() as f:
        content = f.read()
        _, filename = write(
            content,
            "h",
        )
    return normalize_path_separator(filename)


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


def custom_op_wrapper(op: str, *args: Any) -> Union[list[c_void_p], c_void_p]:
    # This function will be called from generated cpp wrapper code in the JIT mode.
    # Because tensors will be passed in as AtenTensorHandle, we need to explicitly convert them.
    def convert_arg(arg: Any) -> Any:
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


@clear_on_fresh_inductor_cache
class CppCodeCache:
    cache: Dict[str, Callable[[], Union[CDLL, ModuleType]]] = {}
    cache_clear = staticmethod(cache.clear)
    cpp_compile_command_flags: Dict[str, Any] = {}

    @staticmethod
    def _load_library_inner(path: str, key: str) -> Union[CDLL, ModuleType]:
        return cdll.LoadLibrary(path)

    @classmethod
    def _load_library(cls, path: str, key: str) -> Union[CDLL, ModuleType]:
        try:
            result = cls._load_library_inner(path, key)
            result.key = key  # type: ignore[union-attr]
            return result
        except (ImportError, OSError) as e:
            if "gomp" in str(e) and os.path.exists("/usr/lib64/libgomp.so.1"):
                # hacky workaround for fbcode/buck
                global _libgomp
                _libgomp = cdll.LoadLibrary("/usr/lib64/libgomp.so.1")
                result = cls._load_library_inner(path, key)
                result.key = key  # type: ignore[union-attr]
                return result
            if "failed to map segment from shared object" in str(e):
                raise OSError(
                    f"{e}.  The most common reason this may occur is if the {tempfile.gettempdir()} folder "
                    "is mounted with noexec (e.g., by default Docker mounts tmp file systems "
                    f"as noexec).  Please remount {tempfile.gettempdir()} with exec enabled, or set another "
                    "temporary directory with TORCHINDUCTOR_CACHE_DIR environment variable."
                ) from e
            raise

    @classmethod
    def load_async(
        cls,
        source_code: str,
        cuda: bool = False,
        submit_fn: Any = None,
        extra_flags: Sequence[str] = (),
    ) -> Any:
        compile_command = {
            **cls.cpp_compile_command_flags,
            "cuda": cuda,
            "vec_isa": pick_vec_isa(),
            "extra_flags": extra_flags,
        }

        _set_gpu_runtime_env()  # cpp_extension consults the env

        command_gen = CppBuilder(
            name="o", sources="i", BuildOption=CppTorchCudaOptions(**compile_command)
        )
        # write function will calc source_code hash, the same source code with different
        # ISA level should be generate different hash.
        # So we need get a command_line which contains isa related parameter as a part of hash key.
        # And then pass the command_line to below write function as extra parameter to
        # guarantee the source code hash contains ISA difference.
        vec_isa_cmd = repr(command_gen.get_command_line())
        key, input_path = write(source_code, "cpp", extra=vec_isa_cmd)

        if key not in cls.cache:
            from filelock import FileLock

            lock_path = os.path.join(get_lock_dir(), key + ".lock")
            output_name, output_dir = get_name_and_dir_from_output_file_path(input_path)
            """
            If `fb_code` env, it need to be dispatched to original `compile_file` function.
            So, we still need to prepare parameters for the function: `input_path` and `fb_output_path`.
            """
            fb_output_path = input_path[:-3] + "so"
            future: Optional[Future[Any]] = None
            lib = None

            cpp_build_option = CppTorchCudaOptions(**compile_command)
            cpp_builder = CppBuilder(
                name=output_name,
                sources=input_path,
                output_dir=output_dir,
                BuildOption=cpp_build_option,
            )

            worker_fn = functools.partial(
                _worker_compile_cpp,
                lock_path,
                cpp_builder,
                input_path,
                fb_output_path,
            )

            binary_path = normalize_path_separator(
                fb_output_path
                if config.is_fbcode()
                else cpp_builder.get_target_file_path()
            )

            def load_fn() -> Any:
                nonlocal lib
                if lib is None:
                    if future is not None:
                        future.result()
                    result = worker_fn()
                    assert result is None
                    lib = cls._load_library(binary_path, key)
                    assert lib is not None
                return lib

            if submit_fn is not None:
                with FileLock(lock_path, timeout=LOCK_TIMEOUT):
                    if not os.path.exists(binary_path):
                        future = submit_fn(worker_fn)

            cls.cache[key] = load_fn

        return cls.cache[key]

    @classmethod
    def load(cls, source_code: str, cuda: bool = False) -> Any:
        return cls.load_async(source_code, cuda)()


def _worker_compile_cpp(
    lock_path: str,
    cpp_builder: CppBuilder,
    fb_input_path: str,
    fb_output_path: str,
) -> None:
    from filelock import FileLock

    with FileLock(lock_path, timeout=LOCK_TIMEOUT):
        binary_path = (
            fb_output_path if config.is_fbcode() else cpp_builder.get_target_file_path()
        )
        if not os.path.exists(binary_path):
            if config.is_fbcode():
                compile_file(
                    fb_input_path,
                    fb_output_path,
                    shlex.split(cpp_builder.get_command_line()),
                )
            else:
                cpp_builder.build()


# Customized Python binding for cpp kernels
@clear_on_fresh_inductor_cache
class CppPythonBindingsCodeCache(CppCodeCache):
    cache: Dict[str, Callable[[], Union[CDLL, ModuleType]]] = {}
    cache_clear = staticmethod(cache.clear)
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

        #ifndef _MSC_VER
        #if __cplusplus < 202002L
        // C++20 (earlier) code
        // https://en.cppreference.com/w/cpp/language/attributes/likely
        #define likely(x)       __builtin_expect(!!(x), 1)
        #define unlikely(x)     __builtin_expect(!!(x), 0)
        #endif
        #else
        #define likely(x) (x)
        #define unlikely(x) (x)
        #endif

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
        template <> inline uintptr_t parse_arg<uintptr_t>(PyObject* args, size_t n) {
            auto result = PyLong_AsVoidPtr(PyTuple_GET_ITEM(args, n));
            if(result == reinterpret_cast<void*>(-1) && PyErr_Occurred())
                [[unlikely]] throw std::runtime_error("expected int arg");
            return reinterpret_cast<uintptr_t>(result);
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
    def load_pybinding_async(
        cls,
        argtypes: List[str],
        source_code: str,
        cuda: bool = False,
        num_outputs: int = -1,
        submit_fn: Any = None,
        extra_flags: Sequence[str] = (),
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
        get_result = cls.load_async(
            source_code + suffix, cuda, submit_fn=submit_fn, extra_flags=extra_flags
        )
        result = None

        def future() -> Any:
            nonlocal result
            if result is None:
                result = get_result()
                assert isinstance(result, ModuleType)
            return getattr(result, cls.entry_function)

        return future

    @classmethod
    def load_pybinding(cls, *args: Any, **kwargs: Any) -> Any:
        return cls.load_pybinding_async(*args, **kwargs)()


@clear_on_fresh_inductor_cache
class CppWrapperCodeCache(CppPythonBindingsCodeCache):
    cache: Dict[str, Callable[[], Union[CDLL, ModuleType]]] = {}
    cache_clear = staticmethod(cache.clear)
    cpp_compile_command_flags = {
        "include_pytorch": True,
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


@clear_on_fresh_inductor_cache
class HalideCodeCache(CppPythonBindingsCodeCache):
    cache: Dict[str, Callable[[], Union[ModuleType, CDLL]]] = {}
    cache_clear = staticmethod(cache.clear)
    _standalone_runtime_path: Optional[str] = None
    prefix = textwrap.dedent(
        """
        #include "{halideruntime_h}"
        #include "{headerfile}"
        #include <stdexcept>
        #include <cmath>

        namespace c10 {{
            inline long div_floor_integer(long a, long b) {{
                if ((a<0) != (b<0)) {{
                    const auto quot = a / b;
                    const auto rem = a % b;
                    return rem ? quot - 1 : quot;
                }}
                return a / b;
            }}
        }}
        """
    )
    glue_template_cpp = prefix + textwrap.dedent(
        """
        void kernel({argdefs}) {{
            {buffers}
            int err = halide_kernel({buffer_names});
            if(err != 0) throw std::runtime_error("halide_kernel failed");
        }}
        """
    )
    glue_template_cuda = prefix + textwrap.dedent(
        """
        #include <cuda.h>
        static const halide_device_interface_t* cuda_interface = halide_cuda_device_interface();

        void kernel({argdefs}, uintptr_t stream) {{
            {buffers}
            int err = halide_kernel(reinterpret_cast<void*>(stream), {buffer_names});
            if(err != 0) throw std::runtime_error("halide_kernel failed");
        }}
        """
    )
    standalone_runtime_cuda_init = textwrap.dedent(
        """
        #include "{}"
        #include <cuda.h>

        static int acquire_context(void* user_context,
                                   void** cuda_context_out,
                                   bool create) {{
            return cuCtxGetCurrent(reinterpret_cast<CUcontext*>(cuda_context_out));
        }}

        static int release_context(void* user_context) {{
            return 0;
        }}

        static int get_stream(void* user_context,
                              void* cuda_context,
                              void** stream_out) {{
            *stream_out = user_context;
            return 0;
        }}

        static int register_halide_hooks() {{
            halide_set_cuda_acquire_context(&acquire_context);
            halide_set_cuda_release_context(&release_context);
            halide_set_cuda_get_stream(&get_stream);
            return 0;
        }}

        int inductor_register_halide_hooks_result = register_halide_hooks();
        """
    )

    @classmethod
    def _codegen_buffer(cls, name: str, arg: HalideInputSpec, cuda: bool) -> List[str]:
        assert arg.shape is not None
        assert arg.stride is not None and len(arg.shape) == len(arg.stride)
        assert arg.offset is not None
        data_ptr = f"{arg.alias_of or arg.name} + {arg.offset}"
        if cuda:
            device = f"reinterpret_cast<uint64_t>({data_ptr})"
            device_interface = "cuda_interface"
            host = "nullptr"
            flags = "halide_buffer_flag_device_dirty"
        else:
            device = "0"
            device_interface = "nullptr"
            host = f"reinterpret_cast<uint8_t*>({data_ptr})"
            flags = "halide_buffer_flag_host_dirty"

        dims = []
        for size, stride in zip(arg.shape, arg.stride):
            dims.append(f"halide_dimension_t(0, {size}, {stride})")

        return [
            f"halide_buffer_t {name};",
            f"halide_dimension_t {name}_dims[] = {{{', '.join(dims)}}};",
            f"{name}.device = {device};",
            f"{name}.device_interface = {device_interface};",
            f"{name}.host = {host};",
            f"{name}.flags = {flags};",
            f"{name}.type = {arg.halide_type()};",
            f"{name}.dimensions = {len(dims)};",
            f"{name}.dim = {name}_dims;",
            f"{name}.padding = nullptr;",
        ]

    @classmethod
    def _codegen_glue(cls, meta: HalideMeta, headerfile: object) -> str:
        is_cuda = meta.is_cuda()
        assert is_cuda is ("user_context" in meta.target)
        assert "no_runtime" in meta.target
        buffers = []
        buffer_names = []
        for i, arg in enumerate(meta.argtypes):
            if arg.is_buffer():
                buffer_names.append(f"&hl_buf_{i}")
                buffers.extend(cls._codegen_buffer(f"hl_buf_{i}", arg, is_cuda))
            else:
                assert "*" not in arg.ctype
                buffer_names.append(arg.name)
        buffers = "\n".join([f"    {line}" for line in buffers]).lstrip()

        glue_template = cls.glue_template_cuda if is_cuda else cls.glue_template_cpp
        glue_code = glue_template.format(
            halideruntime_h=cls.find_header(
                "HalideRuntimeCuda.h" if is_cuda else "HalideRuntime.h"
            ),
            headerfile=headerfile,
            argdefs=", ".join(
                f"{a.bindings_type()} {a.name}"
                for a in meta.argtypes
                if a.alias_of is None
            ),
            buffers=buffers,
            buffer_names=", ".join(buffer_names),
        )
        return glue_code

    @classmethod
    @functools.lru_cache(None)
    def config_hash(cls) -> str:
        command_gen = CppBuilder(
            name="O",
            sources="I",
            BuildOption=CppOptions(),
        )
        command_line = command_gen.get_command_line()
        return sha256_hash(
            "\n".join(
                [
                    cls.glue_template_cpp,
                    cls.glue_template_cuda,
                    cls.standalone_runtime_cuda_init,
                    command_line,
                ]
            ).encode("utf-8")
        )

    @staticmethod
    def _search_for_file(suffix: str, errmsg: str) -> str:
        spec = importlib.machinery.PathFinder.find_spec("halide")
        if spec is None or not spec.submodule_search_locations:
            raise RuntimeError("halide python bindings not installed")
        try:
            search = spec.submodule_search_locations[0]
            for file in os.listdir(search):
                if file.endswith(".so"):
                    try:
                        out = subprocess.check_output(
                            ["ldd", os.path.join(search, file)]
                        )
                    except subprocess.SubprocessError:
                        continue
                    m = re.search(r"(/.*)/libHalide.so", out.decode("utf-8"))
                    if m:
                        path = os.path.join(os.path.abspath(m.group(1)), suffix)
                        if os.path.exists(path):
                            return os.path.abspath(path)
        except Exception as e:
            raise RuntimeError(errmsg) from e
        raise RuntimeError(errmsg)

    @staticmethod
    @functools.lru_cache(None)
    def find_libautoschedule(name: str) -> str:
        sofile = f"libautoschedule_{name.lower()}.so"
        if "HALIDE_LIB" in os.environ:
            path = os.path.join(os.environ["HALIDE_LIB"], sofile)
            if os.path.exists(path):
                return path
        errmsg = (
            f"Can't find {sofile}, set env HALIDE_LIB to the directory containing it"
        )
        return HalideCodeCache._search_for_file(sofile, errmsg)

    @staticmethod
    @functools.lru_cache(None)
    def find_header(name: str) -> str:
        if "HALIDE_INCLUDE" in os.environ:
            path = os.path.join(os.environ["HALIDE_INCLUDE"], name)
            if os.path.exists(path):
                return path
        if "HALIDE_LIB" in os.environ:
            path = os.path.abspath(
                os.path.join(os.environ["HALIDE_LIB"], f"../include/{name}")
            )
            if os.path.exists(path):
                return path
        errmsg = (
            f"Can't find {name}, set env HALIDE_INCLUDE to the directory containing it"
        )
        return HalideCodeCache._search_for_file(f"../include/{name}", errmsg)

    @classmethod
    def generate_halide_async(
        cls, meta: HalideMeta, source_code: str, submit_fn: Any = None
    ) -> Callable[[], Any]:
        dirpath = Path(
            get_path(
                code_hash(
                    source_code,
                    extra=repr((cls.config_hash(), meta)),
                ),
                "halide",
            )[2]
        )
        os.makedirs(dirpath, exist_ok=True)
        wait_for_compile = None
        genfile = str(dirpath / "generate_kernel.py")
        libfile = str(dirpath / "halide_kernel.a")
        headerfile = str(dirpath / "halide_kernel.h")
        donefile = str(dirpath / "done")
        lockfile = str(dirpath / "lock")
        need_compile = not os.path.exists(donefile)
        jobs = []
        if need_compile:
            write_atomic(genfile, source_code)
            cmd = [
                sys.executable,
                genfile,
                "-g",
                "kernel",
                "-o",
                f"{dirpath}",
                "-f",
                "halide_kernel",
                "-e",
                "static_library,h,schedule",
            ]
            if meta.scheduler:
                cmd.extend(["-p", cls.find_libautoschedule(meta.scheduler)])
            cmd.extend(meta.args())
            jobs.append(functools.partial(subprocess.check_call, cmd))

        binding_types = [
            arg.bindings_type() for arg in meta.argtypes if arg.alias_of is None
        ]
        if meta.is_cuda():
            binding_types.append("uintptr_t")  # stream
        bindings_future = cls.load_pybinding_async(
            binding_types,
            cls._codegen_glue(meta, headerfile),
            extra_flags=(libfile, cls.build_standalone_runtime()),
            submit_fn=jobs.append if need_compile else None,
            cuda=meta.is_cuda(),
        )

        if need_compile:
            jobs.append(functools.partial(touch, donefile))
            task = functools.partial(_worker_task_halide, lockfile, jobs)
            if submit_fn:
                wait_for_compile = submit_fn(task).result
            else:
                task()

        def load() -> Callable[[], Any]:
            if wait_for_compile:
                wait_for_compile()
            return bindings_future()

        return load

    @classmethod
    def generate_halide(cls, *args: Any, **kwargs: Any) -> Callable[[], Any]:
        return cls.generate_halide_async(*args, **kwargs)()

    @classmethod
    def build_standalone_runtime(cls) -> str:
        if cls._standalone_runtime_path and os.path.exists(
            cls._standalone_runtime_path
        ):
            return cls._standalone_runtime_path
        is_cuda = torch.cuda.is_available()
        libname = "libStandaloneHalideRuntime.so"
        target = "host-cuda" if is_cuda else "host"
        if cls._standalone_runtime_path:
            assert not os.path.exists(cls._standalone_runtime_path)
            # We hit this case in unittests when we run with fresh_inductor_cache()
            # Generating a fresh runtime over and over causes errors because we initialize
            # cuda hundreds of times in the same process and run out of file descriptors.
            # Workaround by jail breaking the current fresh_inductor_cache().
            base = default_cache_dir()
        else:
            base = cache_dir()
        dirpath = Path(base) / f"halide-runtime-{target}-{cls.config_hash()}"
        os.makedirs(dirpath, exist_ok=True)
        donefile = str(dirpath / "done")
        lockfile = str(dirpath / "lock")
        hookfile = str(dirpath / "hooks.cpp")
        afile = str(dirpath / "standalone_halide_runtime.a")
        sofile = str(dirpath / libname)
        if not os.path.exists(donefile):
            import filelock
            import halide as hl  # type: ignore[import-untyped,import-not-found]

            with filelock.FileLock(lockfile, LOCK_TIMEOUT):
                if not os.path.exists(donefile):
                    with open(hookfile, "w") as f:
                        if is_cuda:
                            f.write(
                                cls.standalone_runtime_cuda_init.format(
                                    cls.find_header("HalideRuntimeCuda.h")
                                )
                            )
                    hl.compile_standalone_runtime(afile, hl.Target(target))

                    name, output_dir = get_name_and_dir_from_output_file_path(sofile)
                    halide_cmd_gen = CppBuilder(
                        name=name,
                        sources=[hookfile, afile],
                        output_dir=output_dir,
                        BuildOption=CppTorchCudaOptions(
                            cuda=is_cuda,
                        ),
                    )

                    subprocess.check_call(
                        shlex.split(halide_cmd_gen.get_command_line())
                    )
                    touch(donefile)
        assert os.path.exists(sofile)
        cls._standalone_runtime_path = sofile
        return sofile


def _worker_task_halide(lockfile: str, jobs: List[partial[Any]]) -> None:
    from filelock import FileLock

    try:
        with FileLock(lockfile, LOCK_TIMEOUT):
            for job in jobs:
                job()
    except subprocess.SubprocessError as e:
        if os.environ.get("HALIDE_REPRO") == "1":
            python, script, *cmd = getattr(e, "cmd", ("", "", ""))
            if os.path.basename(python).startswith("python"):
                code = open(script).read()
                main = "    hl.main()"
                assert code.count(main) == 1

                class Out:
                    def __repr__(self) -> str:
                        return "out"

                cmd[cmd.index("-o") + 1] = Out()  # type: ignore[call-overload]
                repl = textwrap.indent(
                    textwrap.dedent(
                        f"""\
                        import sys, tempfile
                        with tempfile.TemporaryDirectory() as out:
                            sys.argv = {["repro.py", *cmd]!r}
                            hl.main()
                        """
                    ),
                    "    ",
                )
                code = code.replace(main, repl)
                with open("repro.py", "w") as fd:
                    fd.write(code.lstrip())
                raise RuntimeError(f"wrote repro.py: {e}") from e
        raise


def touch(filename: str):  # type: ignore[no-untyped-def]
    open(filename, "a").close()


@clear_on_fresh_inductor_cache
class PyCodeCache:
    cache: Dict[str, ModuleType] = {}
    linemaps: Dict[str, List[Tuple[Any, ...]]] = {}
    cache_clear = staticmethod(cache.clear)

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
            mod = _reload_python_module(key, path)

            # another thread might set this first
            cls.cache.setdefault(key, mod)
            # unzip into separate lines/nodes lists
            cls.linemaps[path] = list(zip(*linemap))

            if attrs is not None:
                for k, v in attrs.items():
                    setattr(mod, k, v)

            if not (linemap or attrs):
                mod._reload_in_subproc = functools.partial(  # type: ignore[attr-defined]
                    _reload_python_module_in_subproc, key, path
                )

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
        return _module_to_triton_kernel(PyCodeCache.load(source_code), kernel_name)


def _cuda_compiler() -> Optional[str]:
    if cuda_env.nvcc_exist(config.cuda.cuda_cxx):
        return config.cuda.cuda_cxx
    if config.is_fbcode():
        return os.path.join(build_paths.cuda(), "bin", "nvcc")
    if cuda_env.nvcc_exist(os.getenv("CUDACXX")):
        return os.getenv("CUDACXX", "")
    if cuda_env.nvcc_exist(os.getenv("CUDA_HOME")):
        return os.path.realpath(os.path.join(os.getenv("CUDA_HOME", ""), "bin/nvcc"))
    return "nvcc"


def _cutlass_include_paths() -> List[str]:
    if config.is_fbcode():
        from libfb.py import parutil

        cutlass_path = parutil.get_dir_path("cutlass-3-headers")
    else:
        cutlass_path = config.cuda.cutlass_dir
    return [
        # Use realpath to get canonical absolute paths, in order not to mess up cache keys
        os.path.realpath(os.path.join(cutlass_path, "include")),
        os.path.realpath(os.path.join(cutlass_path, "tools/library/include")),
        os.path.realpath(os.path.join(cutlass_path, "tools/library/src")),
        os.path.realpath(os.path.join(cutlass_path, "tools/util/include")),
    ]


def _cuda_lib_options() -> List[str]:
    _set_gpu_runtime_env()  # cpp_extension consults the env
    from torch.utils import cpp_extension

    lpaths = cpp_extension.library_paths(cuda=True) + [
        sysconfig.get_config_var("LIBDIR")
    ]
    extra_ldflags: List[str] = []
    if is_linux():
        _transform_cuda_paths(lpaths)
        for path in lpaths:
            # -rpath ensures the DLL can find its dependencies when loaded, even
            # if the library path is non-standard.
            extra_ldflags.extend([f"-L{path}", "-Xlinker", f"-rpath={path}"])
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
    if config.is_fbcode():
        options.extend(["-ccbin", os.path.dirname(build_paths.gcc())])
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
        self.is_open = False
        self.DLL = cdll.LoadLibrary(lib_path)
        self.is_open = True

    def close(self) -> None:
        if self.is_open:
            self._dlclose()
            self.is_open = False

    def _dlclose(self) -> None:
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

    def __getattr__(self, name: str) -> Callable[..., None]:
        if not self.is_open:
            raise RuntimeError(f"Cannot use closed DLL library: {self.lib_path}")

        method = getattr(self.DLL, name)

        def _wrapped_func(*args: Any) -> None:
            err = method(*args)
            if err:
                raise RuntimeError(f"Error in function: {method.__name__}")

        return _wrapped_func

    def __enter__(self) -> DLLWrapper:  # noqa: PYI034
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


@clear_on_fresh_inductor_cache
class CUDACodeCache:
    @dataclasses.dataclass
    class CacheEntry:
        input_path: str
        output_path: str

    cache: Dict[str, CacheEntry] = {}
    cache_clear = staticmethod(cache.clear)
    _SOURCE_CODE_SUFFIX = "cu"

    @classmethod
    def write(cls, source_code: str, dst_file_ext: str) -> Tuple[str, str]:
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
        cls, source_code: str, dst_file_ext: str, extra_args: Optional[List[str]] = None
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
                    log_duration_msg = f"CUDA Compilation took {end_time - start_time} seconds. Compile command: {cmd}"
                    log.info(log_duration_msg)
                else:
                    log.debug(
                        "CUDA Compilation skipped: %s since output already exists",
                        input_path,
                    )
                cls.cache[key] = CUDACodeCache.CacheEntry(input_path, output_path)

        return (cls.cache[key].output_path, key, input_path)

    @classmethod
    def load(cls, source_code: str, dst_file_ext: str) -> Tuple[DLLWrapper, str, str]:
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


@clear_on_fresh_inductor_cache
class ROCmCodeCache:
    @dataclasses.dataclass
    class CacheEntry:
        input_path: str
        output_path: str

    cache: Dict[str, CacheEntry] = {}
    cache_clear = staticmethod(cache.clear)
    _SOURCE_CODE_SUFFIX = "cpp"
    _logged_compiler_version = False

    @classmethod
    def write(cls, source_code: str, dst_file_ext: str) -> Tuple[str, str]:
        """
        Writes source code into a file with dst_file_ext as the file extension.
        Returns the hash key of source code, and the path to the file.
        """

        cuda_command = repr(
            rocm_compile_command(["dummy_input"], "dummy_output", dst_file_ext)
        )
        key, input_path = write(
            source_code, cls._SOURCE_CODE_SUFFIX, extra=cuda_command
        )
        return key, input_path

    @classmethod
    def compile(
        cls, source_code: str, dst_file_ext: str, extra_args: Optional[List[str]] = None
    ) -> Tuple[str, str, str]:
        """
        Compiles source_code into a file with dst_file_ext extension,
        using the compile command specific for the ROCm platform.
        Returns a tuple of dst_file_path, hash_key, source_code_path
        """
        if not cls._logged_compiler_version:
            cls._logged_compiler_version = True
            log.debug(get_compiler_version_info(str(rocm_compiler())))

        key, input_path = cls.write(source_code, dst_file_ext)
        if key not in cls.cache:
            from filelock import FileLock

            lock_dir = get_lock_dir()
            lock = FileLock(os.path.join(lock_dir, key + ".lock"), timeout=LOCK_TIMEOUT)
            with lock:
                output_path = input_path[: -len(cls._SOURCE_CODE_SUFFIX)] + dst_file_ext
                if not os.path.exists(output_path):
                    cmd = rocm_compile_command(
                        [input_path], output_path, dst_file_ext, extra_args
                    )
                    start_time = time()
                    cmd_parts = cmd.split(" ")
                    try:
                        output = subprocess.check_output(
                            cmd_parts,
                            stderr=subprocess.STDOUT,
                            text=True,
                            env=os.environ,
                        )
                        log.debug("Compilation output: %s", output)
                    except subprocess.CalledProcessError as error:
                        raise exc.CUDACompileError(cmd_parts, error.output) from error
                    end_time = time()
                    log_duration_msg = f"Compilation took {end_time - start_time} seconds. Compile command: {cmd}"
                    log.info(log_duration_msg)
                else:
                    log.debug(
                        "Compilation skipped: %s since output already exists",
                        input_path,
                    )
                cls.cache[key] = ROCmCodeCache.CacheEntry(input_path, output_path)

        return (cls.cache[key].output_path, key, input_path)

    @classmethod
    def load(cls, source_code: str, dst_file_ext: str) -> Tuple[DLLWrapper, str, str]:
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


class CodeCacheFuture:
    def result(self) -> None:
        raise NotImplementedError


class TritonFuture(CodeCacheFuture):
    kernel: ModuleType

    def __init__(
        self,
        kernel: Any,
        future: Optional[Future[Any]],
    ):
        self.kernel = kernel
        self.future = future

    # @dynamo_utils.dynamo_timed
    def result(self) -> ModuleType:  # type: ignore[override]
        if self.future is not None:
            # If the worker failed this will throw an exception.
            result = self.future.result()
            assert result is None
            self.future = None
            self.kernel.precompile()
        return self.kernel


class LambdaFuture(CodeCacheFuture):
    def __init__(self, result_fn: Callable[..., Any]) -> None:
        self.result_fn = result_fn

    def result(self) -> Callable[..., Any]:  # type: ignore[override]
        return self.result_fn()
