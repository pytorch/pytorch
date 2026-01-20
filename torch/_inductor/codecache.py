from __future__ import annotations

import base64
import copyreg
import dataclasses
import functools
import hashlib
import importlib
import importlib.resources
import io
import itertools
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
import tempfile
import textwrap
import threading
import warnings
from bisect import bisect_right
from copy import copy
from ctypes import c_void_p, CDLL, cdll
from datetime import timedelta
from functools import lru_cache, partial
from pathlib import Path
from tempfile import _TemporaryFileWrapper
from time import time, time_ns
from types import ModuleType
from typing import Any, cast, Generic, NoReturn, TYPE_CHECKING, TypeVar, Union
from typing_extensions import override, Self

import torch
import torch.distributed as dist
from torch import SymInt, Tensor
from torch._dynamo.device_interface import get_interface_for_device
from torch._dynamo.exc import SkipFrame
from torch._dynamo.utils import (
    CompileEventLogger,
    counters,
    dynamo_timed,
    get_metrics_context,
)
from torch._inductor import config, exc, metrics
from torch._inductor.codegen.common import (
    custom_backend_codegen_configs,
    custom_backend_passes,
    init_backend_registration,
)
from torch._inductor.codegen.cuda import cuda_env
from torch._inductor.codegen.rocm.compile_command import (
    rocm_compile_command,
    rocm_compiler,
)
from torch._inductor.compile_worker.utils import in_toplevel_process
from torch._inductor.cpp_builder import (
    _LINKER_SCRIPT,
    _set_gpu_runtime_env,
    _TORCH_PATH,
    _transform_cuda_paths,
    convert_cubin_to_obj,
    CppBuilder,
    CppOptions,
    CppTorchDeviceOptions,
    get_compiler_version_info,
    get_ld_and_objcopy,
    get_name_and_dir_from_output_file_path,
    normalize_path_separator,
    run_asm_build_object,
)
from torch._inductor.cpu_vec_isa import pick_vec_isa
from torch._inductor.custom_graph_pass import (
    CustomGraphModulePass,
    CustomGraphPass,
    CustomGraphPassType,
    CustomPartitionerFn,
    CustomPartitionerFnType,
)
from torch._inductor.freezing_utils import has_frozen_params, is_frozen_param
from torch._inductor.runtime.compile_tasks import _reload_python_module
from torch._inductor.runtime.runtime_utils import cache_dir, default_cache_dir
from torch._inductor.utils import (
    ALIGN_BYTES,
    clear_on_fresh_cache,
    determine_aoti_mmap_flags,
    is_linux,
    is_windows,
    XPU_KERNEL_FORMAT,
)
from torch._logging import trace_structured
from torch._subclasses.fake_tensor import (
    extract_tensor_metadata,
    FakeTensor,
    TensorMetadata,
)
from torch._utils_internal import log_cache_bypass
from torch.compiler import config as cconfig
from torch.compiler._cache import (
    CacheArtifact,
    CacheArtifactFactory,
    CacheArtifactManager,
)
from torch.export.pt2_archive._package_weights import TensorProperties, Weights
from torch.export.pt2_archive.constants import CUSTOM_OBJ_FILENAME_PREFIX
from torch.fx.experimental.symbolic_shapes import has_hint, ShapeEnv, size_hint
from torch.utils._ordered_set import OrderedSet

from .output_code import CompiledFxGraph
from .remote_cache import create_cache
from .runtime import autotune_cache
from .runtime.autotune_cache import AutotuneCacheBundler
from .triton_bundler import TritonBundler
from .virtualized import V


if config.is_fbcode():
    from triton.fb.build import build_paths


T = TypeVar("T")

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, KeysView, Sequence
    from concurrent.futures import Future

    from .compile_fx import _CompileFxKwargs
    from .cpp_builder import BuildOptionsBase
    from .graph import GraphLowering
    from .ir import ChoiceCaller
    from .output_code import CompiledFxGraphConstants, OutputCode
    from .remote_cache import JsonDataTy, RemoteCache
    from .runtime.hints import HalideInputSpec, HalideMeta
    from .runtime.triton_heuristics import CachingAutotuner
    from .utils import InputType


_IS_WINDOWS = sys.platform == "win32"
LOCK_TIMEOUT = config.file_lock_timeout

output_code_log = torch._logging.getArtifactLogger(__name__, "output_code")
autotuning_log = torch._logging.getArtifactLogger(__name__, "autotuning")
log = logging.getLogger(__name__)


def use_re_build() -> bool:
    """
    Use for CUTLASS compilation only right now.
    """
    if config.is_fbcode() and not cuda_env.nvcc_exist(_cuda_compiler()):
        from triton.fb.re_build_helper import should_build_locally

        return not should_build_locally()
    return False


def get_cpp_wrapper_cubin_path_name() -> str:
    return "cubin_path" if torch.version.hip is None else "hsaco_path"


def get_kernel_bin_format(device: str) -> str:
    if device == "cuda":
        return "cubin" if torch.version.hip is None else "hsaco"
    elif device == "xpu":
        return XPU_KERNEL_FORMAT
    else:
        return ""


def get_device_information(device_type: str) -> dict[str, str]:
    """
    Gets all the current device information used to compile the .so.
    """
    metadata: dict[str, str] = {
        "AOTI_PLATFORM": sys.platform,
        "AOTI_MACHINE": platform.machine(),
        "AOTI_CPU_ISA": str(torch._inductor.cpu_vec_isa.pick_vec_isa()).upper(),
        "AOTI_COMPUTE_CAPABILITY": str(
            get_interface_for_device(device_type).get_compute_capability()
        ),
    }
    return metadata


class CacheBase:
    @staticmethod
    @functools.cache
    def get_system() -> dict[str, Any]:
        from torch._inductor.runtime.triton_compat import HAS_TRITON, triton_key

        if HAS_TRITON:
            # Use triton_key instead of triton.__version__ as the version
            # is not updated with each code change
            triton_version = triton_key()
        else:
            triton_version = None

        try:
            system: dict[str, Any] = {
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
    @clear_on_fresh_cache
    @functools.cache
    def get_local_cache_path() -> Path:
        return Path(os.path.join(cache_dir(), "cache", CacheBase.get_system()["hash"]))

    def __init__(self) -> None:
        self.system = CacheBase.get_system()

    def get_local_cache(self) -> dict[str, Any]:
        local_cache_path = self.get_local_cache_path()
        if not local_cache_path.is_file():
            return {}
        with open(local_cache_path) as local_cache_fp:
            local_cache = json.load(local_cache_fp)
        return local_cache["cache"]

    def update_local_cache(self, local_cache: dict[str, Any]) -> None:
        local_cache_path = self.get_local_cache_path()
        write_atomic(
            str(local_cache_path),
            json.dumps({"system": self.system, "cache": local_cache}, indent=4),
            make_dirs=True,
        )


class LocalCache(CacheBase):
    def lookup(self, *keys: str) -> dict[str, Any] | None:
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
    def lookup(
        self,
        choices: list[ChoiceCaller],
        op: str,
        inputs: str,
        benchmark: Callable[[Any], dict[ChoiceCaller, float]] | None,
        hint_override: int | None = None,
    ) -> dict[ChoiceCaller, float]:
        """
        Check to see if we have benchmarked the given choice callers. For each
        choice caller:

            1. Check local_cache[op][inputs][choice][precision], return benchmark if cached.
            2. If benchmark is not None:
                a. `max_autotune_gemm=True`: benchmark the choice, update
                    local_cache[op][inputs][choice], and return the benchmark.
                b. `max_autotune_gemm=False`: don't benchmark the choice, return nothing.
        """
        precision = torch.get_float32_matmul_precision()
        cache_key = f"{inputs}_{hint_override}" if hint_override is not None else inputs

        timings = {}

        def check_cache(cache: dict[str, Any]) -> bool:
            """Check if `cache` contains data for all the choices"""
            hit = True
            for choice in choices:
                choice_hash = choice.hash_key()
                if choice_hash in cache.get(op, {}).get(cache_key, {}).get(
                    precision, {}
                ):
                    # cache hit
                    timings[choice] = cache[op][cache_key][precision][choice_hash]
                else:
                    # cache miss
                    hit = False
                    break
            return hit

        local_cache = self.get_local_cache() if config.autotune_local_cache else {}
        if (not check_cache(local_cache)) and (benchmark is not None):
            # re-benchmark everything to try to get consistent numbers from the same machine
            timings = benchmark(choices)
            assert all(choice in timings for choice in choices)
            local_cache.setdefault(op, {})
            local_cache[op].setdefault(cache_key, {}).setdefault(precision, {})
            for choice, timing in timings.items():
                local_cache[op][cache_key][precision][choice.hash_key()] = timing

            self.update_local_cache(local_cache)

        return timings


def get_lock_dir() -> str:
    lock_dir = os.path.join(cache_dir(), "locks")
    if not os.path.exists(lock_dir):
        os.makedirs(lock_dir, exist_ok=True)
    return lock_dir


def sha256_hash(data: bytes) -> str:
    # [:51] to strip off the "Q====" suffix common to every hash value.
    return base64.b32encode(hashlib.sha256(data).digest())[:51].decode("utf-8").lower()


def code_hash(code: str | bytes, extra: str | bytes = "") -> str:
    hashing_str = code if isinstance(code, bytes) else code.encode("utf-8")
    if extra:
        extra_b = extra if isinstance(extra, bytes) else extra.encode("utf-8")
        hashing_str = hashing_str + b"||" + extra_b
    return "c" + sha256_hash(hashing_str)


def get_path(
    basename: str, extension: str, specified_dir: str = ""
) -> tuple[str, str, str]:
    if specified_dir:
        if os.path.isabs(specified_dir):
            subdir = specified_dir
        else:
            subdir = os.path.join(cache_dir(), specified_dir)
    else:
        subdir = os.path.join(cache_dir(), basename[1:3])
    path = os.path.join(subdir, f"{basename}.{extension}")
    return basename, subdir, path


def get_hash(content: str | bytes, extra: str = "", hash_type: str = "code") -> str:
    if hash_type in {"amdgcn", "code", "ptx", "spv"}:
        return code_hash(content, extra)
    if hash_type in {"cubin", "hsaco", XPU_KERNEL_FORMAT}:
        return code_hash(repr(content))
    raise AssertionError(f"Unknown hash type {hash_type}")


class WritableTempFile:
    """
    Avoid "Permission denied error" on Windows:
      with tempfile.NamedTemporaryFile("w", suffix=".gv") as temp_file:
        # Not writable on Windows:
        # https://docs.python.org/3/library/tempfile.html#tempfile.NamedTemporaryFile

    Example:
        with WritableTempFile("w", suffix=".gv") as temp_file:
            tree.to_dotfile(temp_file.name)
    """

    def __init__(
        self, mode: str = "w", *, encoding: Any = None, suffix: Any = None
    ) -> None:
        self.mode = mode
        self.encoding = encoding
        self.suffix = suffix

    def __enter__(self) -> _TemporaryFileWrapper[Any]:
        self.temp_file = tempfile.NamedTemporaryFile(
            self.mode, encoding=self.encoding, suffix=self.suffix, delete=False
        )
        return self.temp_file

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.temp_file.close()
        try:
            os.unlink(self.temp_file.name)
        except OSError as e:
            if _IS_WINDOWS:
                # On Windows, some case temp file is opened and fail to unlink. Need to ignore it.
                pass
            else:
                raise e


def write(
    content: str | bytes,
    extension: str,
    extra: str = "",
    hash_type: str = "code",
    specified_dir: str = "",
    key: str | None = None,
) -> tuple[str, str]:
    if key is None:
        # use striped content to compute hash so we don't end up with different
        # hashes just because the content begins/ends with different number of
        # spaces.
        key = get_hash(content.strip(), extra, hash_type)
    basename, _subdir, path = get_path(key, extension, specified_dir)
    if not os.path.exists(path):
        write_atomic(path, content, make_dirs=True)
    return basename, path


def write_text(text: str) -> str:
    """
    Write the `text` to a file and return the path computed based on the hash.
    """
    return write(text, "txt")[1]


def write_atomic(
    path_: str,
    content: str | bytes,
    make_dirs: bool = False,
    encode_utf_8: bool = False,
) -> None:
    # Write into temporary file first to avoid conflicts between threads
    # Avoid using a named temporary file, as those have restricted permissions
    assert isinstance(content, (str, bytes)), (
        "Only strings and byte arrays can be saved in the cache"
    )
    path = Path(path_)
    if make_dirs:
        path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".{os.getpid()}.{threading.get_ident()}.tmp"
    write_mode = "w" if isinstance(content, str) else "wb"
    with tmp_path.open(write_mode, encoding="utf-8" if encode_utf_8 else None) as f:
        f.write(content)
    try:
        tmp_path.rename(target=path)
    except FileExistsError:
        if not _IS_WINDOWS:
            raise
        # On Windows file exist is expected: https://docs.python.org/3/library/pathlib.html#pathlib.Path.rename
        # Below two lines code is equal to `tmp_path.rename(path)` on non-Windows OS.
        # 1. Copy tmp_file to Target(Dst) file.
        shutil.copy2(src=tmp_path, dst=path)
        # 2. Delete tmp_file.
        os.remove(tmp_path)


@dataclasses.dataclass
class TensorMetadataAndValues:
    """
    TensorMetadata plus the elements as a list of raw values.
    Used for hashing inlined constants.
    """

    tensor_metadata: TensorMetadata
    values: list[Any]


def _ident(x: T) -> T:
    return x


def extract_tensor_metadata_for_cache_key(t: Tensor) -> TensorMetadata:
    """
    Extracts the tensor metadata and removes fields of the TensorMetadata
    that are not needed for caching
    """
    meta = extract_tensor_metadata(t)
    if not hasattr(t, "_is_inductor_static"):
        meta = dataclasses.replace(meta, storage_offset=0, storage_bytes=None)

    return meta


class FxGraphCachePickler(pickle.Pickler):
    """
    Custom pickler to customize the pickling of some objects (Tensors), only for the
    purpose of computing a hash for keying into the FxGraphCache. Tensors contain
    objects that don't pickle and/or vary between runs, and we want to capture the
    data that allow us to compute a stable, but safe hash.
    """

    def __init__(
        self,
        gm: torch.fx.GraphModule,
        has_user_defined_triton_kernels: bool = False,
    ) -> None:
        """
        Create an FX graph pickler. If include_non_inlined=True, then pickling will
        include the _values_ for all Tensors. (Note that any tensors are constants
        attached as attributes to the GraphModule). Otherwise, pickling will include
        only the metadata for these tensors.
        """
        self._stream = io.BytesIO()
        super().__init__(self._stream)

        self.dispatch_table = copyreg.dispatch_table.copy()
        self.dispatch_table.update(
            {
                FakeTensor: functools.partial(self._reduce_fake_tensor),
                torch.Tensor: functools.partial(self._reduce_tensor),
                torch.nn.parameter.Parameter: functools.partial(self._reduce_tensor),
                torch.SymInt: functools.partial(self._reduce_symint),
                torch.fx.experimental._backward_state.BackwardState: functools.partial(
                    self._reduce_unsupported
                ),
            }
        )
        if has_user_defined_triton_kernels:
            # Need to use runtime type as GraphModule generates a singleton in __new__ function
            self.dispatch_table[gm.__class__] = functools.partial(
                self._reduce_graph_module
            )

        # Run with pickler.fast so it doesn't intern strings, making the hash result more predictable
        # TODO: pickler.fast is technically deprecated. Will this work on new python versions?
        self.fast = True

    def _reduce_fake_tensor(
        self, t: Tensor
    ) -> tuple[Callable[[T], T], tuple[TensorMetadata]]:
        """
        Custom reducer to pickle FakeTensors.
        """
        metadata = extract_tensor_metadata_for_cache_key(t)
        return (_ident, (metadata,))

    def _reduce_tensor(
        self, t: Tensor
    ) -> tuple[Callable[[T], T], tuple[TensorMetadata | TensorMetadataAndValues]]:
        """
        Custom reducer to pickle Tensors.  If we see tensors, we know they're constants
        stored as attributes on the GraphModule.
        """
        from .graph import GraphLowering

        if t.is_mkldnn:
            # TODO: These tensors don't currently pickle, so we can't cache a compiled
            # graph containing them. Just fail now. If mkldnn tensors get pickling
            # support, we can remove this.
            raise BypassFxGraphCache("mkldnn tensors unpickleable")

        metadata = extract_tensor_metadata_for_cache_key(t)

        # If this is a non-inlined frozen parameter, we consider the metadata only.
        if is_frozen_param(t) and not GraphLowering.can_inline_constant(t):
            return (_ident, (metadata,))

        # Very large tensors will be expensive to copy to cpu and hash. Let's at least
        # report any slowness.
        start = time()
        values = t.tolist()
        elapsed = time() - start
        if elapsed > 1.0:
            warnings.warn(
                f"FX graph cache copying of a large constant took {elapsed:.1}s. "
                "Please file an issue."
            )

        return (_ident, (TensorMetadataAndValues(metadata, values),))

    def _reduce_symint(self, s: SymInt) -> tuple[Callable[[T], T], tuple[str]]:
        """
        Custom reducer to pickle SymInts.
        """
        # For hashing purposes, we only care about the name of the symbol and not the
        # backed value. We evaluate guards stored with a cached graph to ensure a cached
        # entity with SymInt args is safe to reuse.
        return (_ident, (str(s),))

    def _reduce_unsupported(self, s: Any) -> NoReturn:
        """
        Custom reducer to handle any objects that we don't support and therefore
        raise to bypass caching.
        """
        raise BypassFxGraphCache("Reduce unsupported")

    def _reduce_graph_module(
        self, gm: torch.fx.GraphModule
    ) -> tuple[Any, tuple[dict[str, Any], str]]:
        """
        Custom reducer for graph module to handle irrelevant data for user
        defined triton kernels
        Essentially what we are doing here is a huge hack where user defined
        triton kernel contain a dynamo time side table and the arguments to the
        call_function are indices into this side table. These arguments are not
        for hashing purposes since we included the source code into the cache
        key and the numbers are prone to give false negatives due to ordering.
        """
        fn, (data, imports) = gm.__reduce__()
        code = data["_code"]
        code = re.sub(r"kernel_idx = \d+", "", code)
        code = re.sub(r"constant_args_idx = \d+", "", code)
        data["_code"] = code
        return fn, (data, imports)

    def dumps(self, obj: Any) -> bytes:
        """
        Pickle an object and return a byte string.
        """
        try:
            self.dump(obj)
            return self._stream.getvalue()
        except (TypeError, AttributeError, pickle.PicklingError) as e:
            # Some configs options may not pickle.
            log.warning("Failed to pickle cache key", exc_info=True)
            raise BypassFxGraphCache("Failed to pickle cache key") from e
        finally:
            # Reset our stream for the next dump.
            self._stream.seek(0)
            self._stream.truncate(0)

    def get_hash(self, obj: Any) -> str:
        """
        Serialize an object and return a hash of the bytes.
        """
        serialized_data = self.dumps(obj)
        return sha256_hash(serialized_data)

    def debug_lines(self, inp: FxGraphHashDetails) -> list[str]:
        """
        Get a printable string describing in more detail all the attributes
        comprising an object. Useful for debugging when one graph hashes
        to a different value than another.
        """

        def get_str(obj: Any) -> str:
            if isinstance(obj, torch.Tensor):
                return str(extract_tensor_metadata_for_cache_key(obj))
            elif isinstance(obj, bytes):
                val = obj.decode("utf-8", errors="replace")
                return val if len(val) <= 1024 else val[:1024] + "..."
            elif type(obj) in self.dispatch_table:
                # Run the reducer on the object
                return str(self.dispatch_table[type(obj)](obj)[1])
            else:
                return str(obj)

        lines = []
        for attr, obj in vars(inp).items():
            if isinstance(obj, list):
                for ii in range(len(obj)):
                    h = self.get_hash(obj[ii])
                    lines.append(f"[{h}] {attr}[{ii}]: {get_str(obj[ii])}")
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    h = self.get_hash(v)
                    lines.append(f"[{h}] {attr}[{k}]: {get_str(v)}")
            else:
                h = self.get_hash(obj)
                lines.append(f"[{h}] {attr}: {get_str(obj)}")
        return lines


def build_code_hash(
    roots: list[str] | None, prefix: str, hasher: hashlib._Hash
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


def torch_key_cache(func: Callable[[], bytes]) -> Callable[[], bytes]:
    """
    This function is a reimplementation of functools.lru_cache with a
    set function that allows prepopulating the cache.
    """
    # Use list for reference semantics
    _cache: list[bytes] = []

    def wrapper() -> bytes:
        if len(_cache) == 0:
            _cache.append(func())
        return _cache[0]

    def set_val(val: bytes) -> None:
        assert len(_cache) == 0
        _cache.append(val)

    def clear() -> None:
        _cache.clear()

    wrapper.set = set_val  # type: ignore[attr-defined]
    wrapper.clear = clear  # type: ignore[attr-defined]
    return wrapper


@torch_key_cache
def torch_key() -> bytes:
    """
    Compute a key that contains relevant information about torch source files
    """
    with dynamo_timed("inductor_codecache_torch_key", log_pt2_compile_event=False):
        if not config.is_fbcode():

            def get_code_hash(root: str) -> bytes:
                # This function isn't meant to be used outside of torch_key, just a
                # helper for clarity. Instead, use torch_key() directly when you need
                # a hash representing the state of the source code.
                extra_files = (
                    "codegen/aoti_runtime/interface.cpp",
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

        return parutil.get_file_contents("torch/src_hash.txt").rstrip().encode("ascii")


def get_inductor_root() -> str:
    return os.path.dirname(__file__)


@dataclasses.dataclass
class OrderedSetHolder:
    """
    See FxGraphHashDetails. Holds a sorted list to support stable hashing
    of set kwargs.
    """

    items: list[Any]


class BypassFxGraphCache(Exception):
    """
    Exception to indicate that the FxGraphCache should be bypassed.
    """


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
        example_inputs: Sequence[InputType],
        fx_kwargs: _CompileFxKwargs,
        inputs_to_check: Sequence[int],
    ) -> None:
        self.gm = gm
        self.example_inputs = example_inputs
        self.cache_key_tag = cconfig.cache_key_tag

        # Order kwargs so hashing is stable to changes in kwarg order. Although
        # it's technically a _CompileFxKwargs we don't actually need it typed as
        # such since we're just using it to generate a hash.
        self.fx_kwargs: dict[str, object] = {}
        for k, v in sorted(fx_kwargs.items()):
            if k not in self.EXCLUDED_KWARGS:
                if type(v) in (set, OrderedSet):  # noqa: set_linter
                    # Special case to handle set params. Python sets can't be
                    # ordered, so sort the elements and store them in a proxy.
                    self.fx_kwargs[k] = OrderedSetHolder(sorted(v))  # type: ignore[call-overload]
                else:
                    self.fx_kwargs[k] = v

        from torch._higher_order_ops.triton_kernel_wrap import (
            kernel_side_table,
            triton_kernel_wrapper_functional,
            triton_kernel_wrapper_mutation,
        )
        from torch._inductor.codegen.wrapper import (
            user_defined_triton_kernel_transitive_closure_source_code,
        )

        # Node meta will not be part of gm's reduce function, so lets remember
        # the kernel source code separately
        self.user_defined_triton_source: list[Any] = []
        if gm is not None:
            for module in gm.modules():
                if not isinstance(module, torch.fx.GraphModule):
                    continue
                for node in itertools.chain(
                    module.graph.find_nodes(
                        op="call_function", target=triton_kernel_wrapper_functional
                    ),
                    module.graph.find_nodes(
                        op="call_function", target=triton_kernel_wrapper_mutation
                    ),
                ):
                    from triton.runtime.autotuner import Autotuner

                    kernel = kernel_side_table.get_kernel(node.kwargs["kernel_idx"])
                    configs = None
                    if isinstance(kernel, Autotuner):
                        if kernel.configs:
                            configs = str(
                                sorted(
                                    sorted(str(kv) for kv in c.all_kwargs().items())
                                    for c in kernel.configs
                                )
                            )
                        kernel = kernel.fn

                    kernel_source = (
                        user_defined_triton_kernel_transitive_closure_source_code(
                            kernel
                        )
                    )
                    constant_args = kernel_side_table.get_constant_args(
                        node.kwargs["constant_args_idx"]
                    )
                    self.user_defined_triton_source.append(
                        (kernel_source, constant_args, configs)
                    )

        # Alignment checks
        self.inputs_to_check = inputs_to_check

        no_tensor_inputs = not any(isinstance(x, torch.Tensor) for x in example_inputs)
        # This device index is usually already encoded by the device of the inputs
        # but fx graphs don't necessarily have tensor inputs. If there aren't any,
        # we need to guard on the device index in case we allocate cuda tensors
        if no_tensor_inputs and torch.accelerator.is_available():
            self.default_cuda_device_index = torch.accelerator.current_device_index()

        # 'Deterministic algorithms' can affect codegen via lowering to cuda kernels.
        self.deterministic_algorithms_settings = (
            torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled(),
            torch.utils.deterministic.fill_uninitialized_memory,  # type: ignore[attr-defined]
        )

        # Global settings affecting matmul codegen.
        self.cuda_matmul_settings = (
            torch.backends.cuda.matmul.fp32_precision,
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        )

        # Also hash on various system info (including the triton compiler version).
        self.torch_version = torch_key()
        self.system_info = CacheBase.get_system()
        self.inductor_config = config.save_config_portable(ignore_private_configs=False)
        # Custom post grad passes should provide an ID to hash.
        self.post_grad_custom_pre_pass = self._get_custom_pass_detail(
            config.post_grad_custom_pre_pass
        )
        # TODO: change to more holistic config rather than bundled_autograd_cache
        self.precompile_enabled = torch._functorch.config.bundled_autograd_cache
        self.post_grad_custom_post_pass = self._get_custom_pass_detail(
            config.post_grad_custom_post_pass
        )
        self.joint_custom_pre_pass = self._get_custom_pass_detail(
            config.joint_custom_pre_pass
        )
        self.joint_custom_post_pass = self._get_custom_pass_detail(
            config.joint_custom_post_pass
        )
        self._pre_fusion_custom_pass = self._get_custom_pass_detail_unsafe(
            config._pre_fusion_custom_pass
        )
        self._fuse_ddp_communication_passes = self._get_custom_pass_detail_unsafe(
            config._fuse_ddp_communication_passes
        )

        # Register indcutor backends and custom passes and get their UUIDs.
        init_backend_registration()
        self.custom_backend_passes = tuple(
            map(self._get_custom_pass_detail, custom_backend_passes.values())
        )

        # Save custom inductor codegen configs
        self.custom_backend_codegen_configs = {
            device: custom_config.save_config_portable(ignore_private_configs=False)
            for device, custom_config in custom_backend_codegen_configs.items()
            if custom_config is not None
        }

        # Register the custom partitioner function
        self._custom_partitioner_fn = self._get_custom_partitioner_fn_detail(
            config.custom_partitioner_fn
        )

    # This is mainly added to handle these two inductor configs, which are (unfortunately)
    # sometimes cache safe:
    # - _pre_fusion_custom_pass
    # - _fuse_ddp_communication_passes
    # Their types can be found in `torch/_inductor/config.py`, but:
    # - if they are string names, we can cache them safely (one is by default)
    # - if any of them are set to custom callables, we will need to cache miss
    # Future work is for someone to find any places where these functions are used
    # and force them to be of type CustomGraphPass, so we can guarantee serialization.
    def _get_custom_pass_detail_unsafe(self, custom_pass: Any) -> Any | None:
        if not custom_pass:
            return None
        if isinstance(custom_pass, list):
            return [self._get_custom_pass_detail_unsafe(x) for x in custom_pass]
        if isinstance(custom_pass, str):
            return custom_pass
        if isinstance(custom_pass, CustomGraphPass):
            return custom_pass.uuid()
        if callable(custom_pass):
            # Returning None is safe here because we raise an explicit bypass error
            # later if we detect these passes are set to callables
            return None
        raise AssertionError(f"unknown config type: {str(type(custom_pass))}")

    def _get_custom_pass_detail(
        self, custom_pass: CustomGraphPassType | CustomGraphModulePass
    ) -> Any | None:
        if not custom_pass:
            return None
        assert isinstance(custom_pass, (CustomGraphPass, CustomGraphModulePass))
        return custom_pass.uuid()

    def _get_custom_partitioner_fn_detail(
        self, custom_partitioner_fn: CustomPartitionerFnType
    ) -> Any | None:
        if not custom_partitioner_fn:
            return None
        assert isinstance(custom_partitioner_fn, CustomPartitionerFn)
        return custom_partitioner_fn.uuid()


def compiled_fx_graph_hash(
    gm: torch.fx.GraphModule,
    example_inputs: Sequence[InputType],
    fx_kwargs: _CompileFxKwargs,
    inputs_to_check: Sequence[int],
) -> tuple[str, list[str]]:
    """
    Generate a unique hash of the FX graph for caching.
    """
    details = FxGraphHashDetails(gm, example_inputs, fx_kwargs, inputs_to_check)
    has_user_defined_triton_kernels = len(details.user_defined_triton_source) != 0
    pickler = FxGraphCachePickler(gm, has_user_defined_triton_kernels)

    # The prefix distinguishes among the other kinds of objects we
    # cache in this module.
    key = "f" + pickler.get_hash(details)
    debug_lines = pickler.debug_lines(details)
    debug_str = "\n".join(debug_lines)
    log.debug(f"FX graph cache hash details for key {key}:\n{debug_str}")  # noqa: G004
    return key, debug_lines


def add_ephemeral_timeout_increase_for_distributed(time_saved_ns: int) -> int:
    """
    Ephemerally increases the NCCL timeout when compiling for a distributed job
    Returns amount of seconds increased
    """
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return 0

    increased_timeout_sec = int(time_saved_ns // 1e9)  # convert to seconds

    if config.is_fbcode():
        fudge_factor = torch._utils_internal.justknobs_getval_int(
            "pytorch/remote_cache:ephemeral_timeout_fudge_factor_percentage"
        )
        log.info(
            "Ephemeral NCCL timeout increase fudge factor %d and original increase value %d",
            fudge_factor,
            increased_timeout_sec,
        )
        increased_timeout_sec += int(increased_timeout_sec * fudge_factor / 100)

    log.info("Increasing NCCL timeout by %d", increased_timeout_sec)
    dist.distributed_c10d._add_ephemeral_timeout_for_all_pgs(
        timedelta(seconds=increased_timeout_sec)
    )
    return increased_timeout_sec


class GuardedCache(Generic[T]):
    """
    Mixin for caches that have guards associated with their entries.
    """

    @classmethod
    def _get_tmp_dir_for_key(cls: type[GuardedCache[T]], _key: str) -> str:
        raise NotImplementedError("Implement _get_tmp_dir_for_key on parent class")

    @classmethod
    def _record_result(
        cls: type[GuardedCache[T]],
        key: str,
        local_hit: bool,
        local_miss: bool,
        remote_hit: bool,
        remote_miss: bool,
    ) -> None:
        raise NotImplementedError("Implement _record_result on parent class")

    @classmethod
    def iterate_over_candidates(
        cls: type[GuardedCache[T]],
        local: bool,
        remote_cache: RemoteCache[JsonDataTy] | None,
        key: str,
    ) -> Generator[tuple[T, bytes, bool], None, None]:
        if local:
            subdir = cls._get_tmp_dir_for_key(key)
            if os.path.exists(subdir):
                for path in sorted(os.listdir(subdir)):
                    try:
                        with open(os.path.join(subdir, path), "rb") as f:
                            content = f.read()
                            yield pickle.loads(content), content, True
                    except Exception:
                        log.warning(
                            "fx graph cache unable to load compiled graph",
                            exc_info=True,
                        )

        if remote_cache:
            try:
                if (cache_data := remote_cache.get(key)) is not None:
                    assert isinstance(cache_data, dict)
                    data = cache_data["data"]
                    assert isinstance(data, (str, bytes))
                    content = base64.b64decode(data)
                    yield pickle.loads(content), content, False
            except Exception:
                log.warning(
                    "%s unable to load compiled graph", cls.__name__, exc_info=True
                )

    @classmethod
    def find_guarded_entry(
        cls: type[GuardedCache[T]],
        key: str,
        local: bool,
        remote_cache: RemoteCache[JsonDataTy] | None,
        evaluate_guards: Callable[[str, list[int] | list[torch.SymInt]], bool],
        hints: list[int],
    ) -> tuple[T | None, bytes | None, dict[str, str]]:
        """
        Find the first cache entry in iterate_over_candidates that passes `evaluate_guards`.

        Args:
            key: The cache key to look up
            local: Whether to check the local cache
            remote_cache: The remote cache to check, if any
            evaluate_guards: Function that evaluates whether a guard passes the check,
                given a list of hint values and the guard expression.
            hints: List of symint hints paired with evaluate_guards

        Returns:
            A tuple of (graph, pickled_content) if found, or (None, None) if not found
        """
        graph = None
        pickled_content = None
        result_status = "full_miss"
        sample_guards_expr = None
        in_local = False

        # Iterate over any entries in the subdir for this key and evaluate
        # guards to determine whether there's a hit.

        for candidate, content, in_local in cls.iterate_over_candidates(
            local, remote_cache, key
        ):
            assert hasattr(candidate, "guards_expr")
            if not candidate.guards_expr:  # type: ignore[attr-defined]
                # No guards to evaluate, so this is a hit.
                graph = candidate
                pickled_content = content
                result_status = "hit"
                break

            # Evaluate the guard expression in the current context.
            # If there's not a cache hit, we don't want the evaluation to
            # affect the current env, e.g., cause the creation of new guards,
            # so we evaluate with the hints instead of the symbols.
            hit = bool(evaluate_guards(candidate.guards_expr, hints))  # type: ignore[attr-defined]
            if hit:
                graph = candidate
                pickled_content = content
                result_status = "hit"
                sample_guards_expr = candidate.guards_expr
                break
            else:
                # At least one guard missed, log this
                result_status = "guard_miss"
                sample_guards_expr = candidate.guards_expr

        info = {"cache_status_detailed": result_status}
        if sample_guards_expr is not None:
            info["cache_status_guard_expr"] = sample_guards_expr

        # Record hits/misses for compilation event logging. The tricky part is that a
        # remote hit would imply a local miss (if local caching is enabled).
        local_hit = graph is not None and in_local
        remote_hit = graph is not None and not in_local
        local_miss = (graph is None or remote_hit) and local
        remote_miss = graph is None and remote_cache is not None
        cls._record_result(
            key,
            local_hit=local_hit,
            local_miss=local_miss,
            remote_hit=remote_hit,
            remote_miss=remote_miss,
        )

        return graph, pickled_content, info

    @classmethod
    def _filter_backed_symints(
        cls: type[GuardedCache[T]], inputs: Sequence[InputType]
    ) -> list[torch.SymInt]:
        """
        Get the backed SymInt objects from the input list. Note that we can never
        have guards that depend on unbacked symint.
        """
        return [s for s in inputs if isinstance(s, torch.SymInt) and has_hint(s)]

    @classmethod
    def _get_shape_env(cls: type[GuardedCache[T]]) -> ShapeEnv | None:
        """
        Helper to get the shape env from the tracing context.
        """
        ctx = torch._guards.TracingContext.try_get()
        if not ctx or not ctx.fake_mode:
            return None
        return ctx.fake_mode.shape_env


@CacheArtifactFactory.register
class InductorCacheArtifact(CacheArtifact):
    @override
    def populate_cache(self) -> None:
        FxGraphCache._write_to_local_cache(self.key, self.content)

    @override
    @staticmethod
    def type() -> str:
        return "inductor"


class FxGraphCache(GuardedCache[CompiledFxGraph]):
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
          <temp dir>/<fx graph hash>/<serialized metadata>
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

    @classmethod
    def _get_tmp_dir_for_key(cls: type[FxGraphCache], key: str) -> str:
        """
        Return the disk location for a given cache key.
        """
        return os.path.join(FxGraphCache._get_tmp_dir(), key[1:3], key)

    @classmethod
    def _record_result(
        cls: type[FxGraphCache],
        key: str,
        local_hit: bool,
        local_miss: bool,
        remote_hit: bool,
        remote_miss: bool,
    ) -> None:
        """
        Called by GuardedCache to record hit/miss statistics.
        """
        if local_hit:
            CompileEventLogger.try_(
                CompileEventLogger.increment_toplevel,
                "inductor_fx_local_cache_hit_count",
            )
        if remote_hit:
            CompileEventLogger.try_(
                CompileEventLogger.increment_toplevel,
                "inductor_fx_remote_cache_hit_count",
            )
            CompileEventLogger.try_(
                CompileEventLogger.add_to_set_toplevel,
                "inductor_fx_remote_cache_hit_keys",
                key,
            )
        if local_miss:
            CompileEventLogger.try_(
                CompileEventLogger.increment_toplevel,
                "inductor_fx_local_cache_miss_count",
            )
        if remote_miss:
            CompileEventLogger.try_(
                CompileEventLogger.increment_toplevel,
                "inductor_fx_remote_cache_miss_count",
            )
            CompileEventLogger.try_(
                CompileEventLogger.add_to_set_toplevel,
                "inductor_fx_remote_cache_miss_keys",
                key,
            )

    @staticmethod
    def cache_hit_post_compile(
        graph: CompiledFxGraph,
        cache_info: dict[str, Any],
        constants: CompiledFxGraphConstants,
    ) -> tuple[CompiledFxGraph | None, dict[str, Any]]:
        """
        Cache specific post compile steps that need to run if we find a graph in the cache
        This includes putting bundled triton artifacts in the right place,
        reloading the PyCodeCache artifact, etc.

        These don't always happen (i.e. on a cache miss, so they are in a separate function from
        CompiledFxGraph.post_compile)
        """
        if bundle := graph._triton_bundle:
            triton_bundler_meta = TritonBundler.read_and_emit(bundle)
            if (meta := triton_bundler_meta) is not None:
                cache_info["triton_bundler_meta"] = str(meta)
                CompileEventLogger.try_add_pt2_compile(
                    "inductor_compile", cached_kernel_names=meta.cached_kernel_names
                )
                CompileEventLogger.try_add_pt2_compile(
                    "AOTAutogradCache.inductor_load",
                    cached_kernel_names=meta.cached_kernel_names,
                )
                if len(meta.cached_kernel_names) > 0:
                    CompileEventLogger.try_(
                        CompileEventLogger.increment_toplevel, "num_triton_bundles"
                    )

        try:
            artifact_path = graph.after_deserialization(constants)

            from .graph import GraphLowering

            # This is used by tests to check the output for specific details.
            if GraphLowering.save_output_code is not None:
                GraphLowering.save_output_code(graph.source_code)

        except OSError:
            # Not expected, but in case the PyCodeCache entry is removed from
            # underneath us, treat it as a cache miss and recompile.
            return None, cache_info

        inductor_meta = autotune_cache.inductor_meta_from_config()
        code = graph.source_code
        AutotuneCacheBundler.begin_compile(inductor_meta, code=code)

        # Increment the cached metrics/counters by the amounts recorded when the FX
        # graph was compiled for this cache entry. Pretending these counters
        # were incremented normally is useful for testing with the cache enabled.
        metrics.CachedMetricsHelper.apply_deltas(graph.metrics_deltas)
        counters["inductor"] += graph.counter_deltas

        output_code_log.debug("Output code: \n%s", code)
        output_code_log.debug("Output code written to: %s", artifact_path)
        # On cache hit, use artifact path as filename
        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "fx_graph_runnable",
                "encoding": "string",
            },
            payload_fn=lambda: graph.runnable_graph_str,
        )
        trace_structured(
            "inductor_post_grad_graph",
            payload_fn=lambda: graph.inductor_post_grad_graph_str,
        )
        trace_structured(
            "inductor_output_code",
            lambda: {
                "filename": artifact_path,
                "file_path": os.path.abspath(artifact_path),
            },
            payload_fn=lambda: code,
        )
        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "inductor_provenance_tracking_node_mappings",
                "encoding": "json",
            },
            payload_fn=lambda: graph.inductor_provenance_mapping_str,
        )
        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "inductor_provenance_tracking_kernel_stack_traces",
                "encoding": "json",
            },
            payload_fn=lambda: graph.inductor_provenance_stack_traces_str,
        )
        if (
            get_metrics_context().in_progress()
            and graph.inductor_provenance_stack_traces_str
        ):
            get_metrics_context().add_to_set(
                "inductor_provenance", graph.inductor_provenance_stack_traces_str
            )
        return graph, cache_info

    @staticmethod
    def _lookup_graph(
        key: str,
        example_inputs: Sequence[InputType],
        local: bool,
        remote_cache: RemoteCache[JsonDataTy] | None,
        constants: CompiledFxGraphConstants,
        evaluate_guards: Callable[[str, list[int] | list[torch.SymInt]], bool]
        | None = None,
    ) -> tuple[CompiledFxGraph | None, dict[str, Any]]:
        """
        Lookup a compiled graph in the cache by key. On a hit, return the
        deserialized CompiledFxGraph object. On a miss, return None.
        `constants` tracks a list of constants, or a way to obtain the list of constants
        associated with a given cache entry
        `evaluate_guards` allows AOTAutogradCache and other callers to customize
        what constitutes a guard success. Normally, a guard hit happens if
        `shape_env.evaluate_guards_expression` returns True.
        """
        shape_env = FxGraphCache._get_shape_env()
        assert shape_env is not None

        symints = FxGraphCache._filter_backed_symints(example_inputs)
        hints = [size_hint(s) for s in symints]

        # If this config is turned on, everything is a guard hit and we check nothing
        if config.unsafe_skip_cache_dynamic_shape_guards:
            # This also makes it so we don't add anything to the dynamic
            # shape environment
            evaluate_guards = lambda x, y: True  # noqa: E731

        if evaluate_guards is None:
            evaluate_guards = shape_env.evaluate_guards_expression

        cache_info: dict[str, Any] = dict()

        # Use the find_graph_for_key method to find a graph for the given key
        graph, pickled_content, guard_info = FxGraphCache.find_guarded_entry(
            key, local, remote_cache, evaluate_guards, hints
        )
        cache_info.update(guard_info)
        if graph is None:
            return None, cache_info

        if pickled_content is not None:
            CacheArtifactManager.record_artifact(
                InductorCacheArtifact.type(), key, pickled_content
            )

        # Now re-evaluate with the symints to add any guards to the current env.
        if graph.guards_expr:
            check = bool(evaluate_guards(graph.guards_expr, symints))
            assert check is True
            log.debug(
                "fx graph cache key %s post-load guards: %s", key, shape_env.guards
            )

        return FxGraphCache.cache_hit_post_compile(graph, cache_info, constants)

    @staticmethod
    def _write_to_local_cache(key: str, content: bytes) -> None:
        subdir = FxGraphCache._get_tmp_dir_for_key(key)
        if not os.path.exists(subdir):
            os.makedirs(subdir, exist_ok=True)

        # Use a hash of the serialized CompiledFxGraph to get a unique file
        # name. The specific name doesn't matter since a lookup involves
        # iterating over all entries in the parent subdir.
        path = os.path.join(subdir, sha256_hash(content))
        write_atomic(path, content, make_dirs=True)

    @staticmethod
    def _save_graph(
        key: str,
        compiled_graph: OutputCode,
        example_inputs: Sequence[InputType],
        local: bool,
        remote_cache: RemoteCache[JsonDataTy] | None,
    ) -> None:
        """
        Store a serialized CompiledFxGraph on disk.
        """
        from .compile_fx import CompiledFxGraph

        assert isinstance(compiled_graph, CompiledFxGraph), (
            f"serialization for {type(compiled_graph)} NYI"
        )

        # Before serializing, compute the guard expression that will be used to
        # ensure that a CompiledFxGraph is valid when loaded from the cache. It's
        # sufficient to consider only the SymInt args to the fx graph since the
        # Tensor shapes are already captured in the hash for the cache key. Any
        # Tensor arg with a symbolic shape will have a SymInt arg for the graph.
        shape_env = FxGraphCache._get_shape_env()
        assert shape_env is not None
        symints = FxGraphCache._filter_backed_symints(example_inputs)
        guards = shape_env.get_pruned_guards(symints)
        compiled_graph.guards_expr = shape_env.produce_guards_expression(
            placeholders=symints, guards=guards
        )
        disk_compiled_graph = copy(compiled_graph)
        disk_compiled_graph.prepare_for_serialization()

        try:
            content = pickle.dumps(disk_compiled_graph)
        except Exception:
            log.warning(
                "fx graph cache unable to serialize compiled graph", exc_info=True
            )
            counters["inductor"]["fxgraph_cache_pickle_error"] += 1
            return

        try:
            CacheArtifactManager.record_artifact(
                InductorCacheArtifact.type(), key, content
            )
            if local:
                FxGraphCache._write_to_local_cache(key, content)

            if remote_cache:
                time_taken_ms = int((disk_compiled_graph._time_taken_ns or 0) // 1e6)
                cache_data: JsonDataTy = {
                    "data": base64.b64encode(content).decode("ascii"),
                    "time_taken_ms": time_taken_ms,
                }
                remote_cache.put(key, cache_data)
        except Exception:
            log.warning("fx graph unable to write to cache", exc_info=True)
            counters["inductor"]["fxgraph_cache_write_error"] += 1

    @staticmethod
    def _check_for_hop(gm: torch.fx.GraphModule) -> None:
        for module in gm.modules():
            if not isinstance(module, torch.fx.GraphModule):
                continue
            for node in module.graph.nodes:
                if (
                    isinstance(node.target, torch._ops.HigherOrderOperator)
                    and not node.target.cacheable()
                ):
                    raise BypassFxGraphCache(
                        f"Can't cache HigherOrderOperator: {node.target.name()}"
                    )
                if node.op == "getattr" and isinstance(
                    getattr(gm, node.target), torch._C.ScriptObject
                ):
                    raise BypassFxGraphCache("Can't cache torchbind objects")

    @staticmethod
    def _check_can_cache(gm: torch.fx.GraphModule) -> None:
        """
        Check some conditions that would preclude caching and raise BypassFxGraphCache
        to bypass in case caching is not possible.
        """
        # Post grad custom passes must implement the CustomGraphPass or we don't
        # know how to include them in the cache key calculation.
        for p in (config.post_grad_custom_pre_pass, config.post_grad_custom_post_pass):
            if p and (not isinstance(p, CustomGraphPass) or not p.uuid()):
                raise BypassFxGraphCache("Unsupported post grad custom pass")
        # Same with the joint custom passes
        for p in (config.joint_custom_pre_pass, config.joint_custom_post_pass):
            if p and (not isinstance(p, CustomGraphPass) or not p.uuid()):
                raise BypassFxGraphCache("Unsupported joint custom pass")
        # We should find any users of _pre_fusion_custom_pass and _fuse_ddp_communication_passes
        # and ensure they are not passing us raw callables
        if config._pre_fusion_custom_pass is not None:
            if not isinstance(config._pre_fusion_custom_pass, CustomGraphPass):
                raise BypassFxGraphCache("Unsupported _pre_fusion_custom_pass")
        for p in config._fuse_ddp_communication_passes:
            if callable(p) and not isinstance(p, CustomGraphPass):
                raise BypassFxGraphCache("Unsupported _fuse_ddp_communication_pass")

        # Freezing can embed constants that wouldn't be static across runs.
        if has_frozen_params(gm) and not torch._utils_internal.justknobs_check(
            "pytorch/inductor:allow_freezing_with_caching"
        ):
            raise BypassFxGraphCache("Skipping graph with frozen constants")

        if config.aot_inductor.use_runtime_constant_folding:
            raise BypassFxGraphCache(
                "Runtime constant folding can introduce constants that aren't "
                "static across runs"
            )

        from torch._inductor.compiler_bisector import CompilerBisector

        if CompilerBisector.bisection_enabled:
            log.debug("dont cache graph when bisect enabled")
            raise BypassFxGraphCache

        # The treatment of guards in the caching implementation requires that
        # we have a shape env.
        if FxGraphCache._get_shape_env() is None:
            log.debug("fx graph cache no shape env")
            raise BypassFxGraphCache("No shape env")

        # We skip caching if there are any HOPs or torchbind objects.
        FxGraphCache._check_for_hop(gm)

    @staticmethod
    def prepare_key(
        gm: torch.fx.GraphModule,
        example_inputs: Sequence[InputType],
        fx_kwargs: _CompileFxKwargs,
        inputs_to_check: Sequence[int],
        remote: bool,
    ) -> tuple[tuple[str, list[str]] | None, dict[str, Any]]:
        """
        Checks that the inductor input is cacheable, then computes
        and returns the cache key for the input.
        Returns (key_info, cache_info) where:
        - key_info is (hash_key, debug_lines), and
        - cache_info will contain debug info in the event of BypassFxGraphCache.

        NB: It is possible to have this function return a union instead. But
        I personally believe it is more annoying/difficult to read in that format.
        """
        try:
            FxGraphCache._check_can_cache(gm)
            key, debug_lines = compiled_fx_graph_hash(
                gm, example_inputs, fx_kwargs, inputs_to_check
            )
        except BypassFxGraphCache as e:
            counters["inductor"]["fxgraph_cache_bypass"] += 1
            log.info("Bypassing FX Graph Cache because '%s'", e)  # noqa: G200
            if remote:
                log_cache_bypass("bypass_fx_graph", str(e))
            cache_info = {
                "cache_state": "bypass",
                "cache_bypass_reason": str(e),
                "cache_event_time": time_ns(),
            }
            return None, cache_info
        # If key exists, then cache_info will come from load_with_key
        return (key, debug_lines), {}

    @staticmethod
    def get_remote_cache() -> RemoteCache[JsonDataTy] | None:
        """
        Attempts to load the remote cache, returns None on error.
        """
        cache_id = "fx-graph-v1"
        return create_cache(
            cache_id,
            config.is_fbcode(),
            "FbRemoteFxGraphCache",
            "RemoteFxGraphCache",
        )

    @staticmethod
    def load_with_key(
        key: str,
        debug_lines: list[str],
        example_inputs: Sequence[InputType],
        local: bool,
        remote_cache: RemoteCache[JsonDataTy] | None,
        is_backward: bool,
        constants: CompiledFxGraphConstants,
        evaluate_guards: Callable[[str, list[int] | list[torch.SymInt]], bool]
        | None = None,
    ) -> tuple[CompiledFxGraph | None, dict[str, Any]]:
        """
        Lookup the graph with the given key, and return results and metadata.
        Doesn't do any logging on its own, because AOTAutograd handles a cache miss
        differently from FXGraphCache.
        """
        compiled_graph, cache_info = FxGraphCache._lookup_graph(
            key, example_inputs, local, remote_cache, constants, evaluate_guards
        )
        cache_info = {
            **cache_info,
            "key": key,
            "components": debug_lines,
            "cache_event_time": time_ns(),
        }
        if compiled_graph is not None:
            log.info("fx graph cache hit for key %s", key)
            counters["inductor"]["fxgraph_cache_hit"] += 1
            cache_info["cache_state"] = "hit"

            if (time_saved_ns := compiled_graph._time_taken_ns) is not None:
                cache_info["time_saved_ns"] = time_saved_ns
                CompileEventLogger.try_(
                    CompileEventLogger.increment_toplevel,
                    "distributed_ephemeral_timeout_us",
                    time_saved_ns // 1000,
                )
                if (
                    ephemeral_increase
                    := add_ephemeral_timeout_increase_for_distributed(time_saved_ns)
                ) != 0:
                    cache_info["ephemeral_timeout_increase"] = ephemeral_increase
        else:
            log.info("fx graph cache miss for key %s", key)
            counters["inductor"]["fxgraph_cache_miss"] += 1
            cache_info["cache_state"] = "miss"

        return compiled_graph, cache_info

    @staticmethod
    def clear() -> None:
        """
        Clear out the on-disk cache.
        """
        try:
            shutil.rmtree(FxGraphCache._get_tmp_dir())
        except FileNotFoundError:
            pass


@functools.cache
def split_aot_inductor_output_path(path: str) -> tuple[str, str]:
    def get_module_ext_type() -> str:
        if _IS_WINDOWS:
            return ".pyd"
        else:
            return ".so"

    """Returns the path where the AOT Inductor compiled kernels are stored."""
    if path.endswith(get_module_ext_type()):
        return os.path.split(path)
    elif path.endswith(".pt2"):
        return os.path.split(path)
    else:
        return path, ""


@clear_on_fresh_cache
class CudaKernelParamCache:
    cache: dict[str, dict[str, Any]] = {}
    cache_clear = staticmethod(cache.clear)

    @classmethod
    def set(
        cls,
        key: str,
        params: dict[str, str | None],
        cubin: str,
        bin_type: str,
        asm: str | None = None,
        asm_type: str | None = None,
    ) -> None:
        basename = None
        if config.aot_inductor.package_cpp_only:
            assert config.triton.unique_kernel_names, (
                "package_cpp_only requires triton kernel names to be unique"
            )
            assert params["mangled_name"], "Missing kernel name"
            basename = params["mangled_name"]

        _, bin_path = write(
            cubin,
            bin_type,
            hash_type=bin_type,
            specified_dir=split_aot_inductor_output_path(
                config.aot_inductor.output_path
            )[0],
            key=basename,
        )
        # Retrieve the basename again in case it is a generated hashcode
        basename, _ = get_name_and_dir_from_output_file_path(bin_path)

        if config.aot_inductor.emit_multi_arch_kernel:
            bin_type_to_ext = {
                "cubin": ".fatbin",
                "spv": f".{XPU_KERNEL_FORMAT}",
                "hsaco": ".hsaco",
            }
            assert bin_type in bin_type_to_ext, (
                "multi_arch_kernel_binary only supported in CUDA/XPU/ROCm"
            )
            base_path, _ = os.path.splitext(bin_path)
            bin_path = base_path + bin_type_to_ext[bin_type]

        asm_path: str = ""

        # Kernel assembly/IR requirements for AOT Inductor:
        # - CUDA/XPU: Always require PTX/SPV
        # - ROCm multi-arch: Require LLVM IR (.ll) for bundle compilation
        if (
            config.aot_inductor.emit_multi_arch_kernel
            or config.aot_inductor.package_cpp_only
        ):
            # Allow ROCm single-arch to skip (asm=None OK), require for everything else
            if torch.version.hip is None or (asm and asm_type):
                assert asm, "Missing kernel assembly code"
                assert asm_type, "Missing kernel assembly type"

                # Cache directory mapping: asm_type → hash_type
                # Problem: LLVM IR extension ".ll" isn't a recognized cache category
                # Solution: Map to "code" (generic category for non-standard formats)
                # Recognized categories: "ptx", "amdgcn", "spv", "code"
                hash_kind = asm_type if asm_type in {"amdgcn", "ptx", "spv"} else "code"

                _, asm_path = write(
                    asm,
                    asm_type,
                    hash_type=hash_kind,
                    specified_dir=split_aot_inductor_output_path(
                        config.aot_inductor.output_path
                    )[0],
                    key=basename,
                )

        params[get_cpp_wrapper_cubin_path_name()] = bin_path
        params["asm"] = asm_path
        cls.cache[key] = params

    @classmethod
    def get(cls, key: str) -> dict[str, Any] | None:
        return cls.cache.get(key, None)

    @classmethod
    def get_keys(cls) -> KeysView[str]:
        return cls.cache.keys()


class AotCodeCompiler:
    """
    Compile AOT Inductor generated code.
    """

    @classmethod
    def compile(
        cls,
        graph: GraphLowering,
        wrapper_code: str,
        kernel_code: str,
        serialized_extern_kernel_nodes: str | None,
        *,
        device_type: str,
        additional_files: list[str],
    ) -> list[Union[str, Weights]] | str:
        """
        Returns the .so path, or returns a list of files that were generated if
        config.aot_inductor.package=True.
        """
        generated_files: list[str | Weights] = additional_files  # type: ignore[assignment]

        _set_gpu_runtime_env()  # cpp_extension consults the env

        picked_vec_isa = pick_vec_isa()
        vec_isa_cmd_gen = CppBuilder(
            name="o",
            sources="i",
            BuildOption=CppTorchDeviceOptions(
                vec_isa=picked_vec_isa,
                device_type=device_type,
                aot_mode=graph.aot_mode,
            ),
        )
        # write function will calc source_code hash, the same source code with different
        # ISA level should be generate different hash.
        # So we need get a command_line which contains isa related parameter as a part of hash key.
        # And then pass the command_line to below write function as extra parameter to
        # guarantee the source code hash contains ISA difference.
        cpp_command = repr(vec_isa_cmd_gen.get_command_line())

        # Meta internal AOTInductor CPU
        use_relative_path = (
            config.is_fbcode() and device_type == "cpu" and graph.aot_mode
        )

        (
            specified_output_path,
            specified_artifact_name,
        ) = split_aot_inductor_output_path(config.aot_inductor.output_path)

        # TODO (benjaminglass1): the CMake packaging path doesn't support linking files
        # built with different flags.  Until that's implemented, append the kernel code
        # to the wrapper and build everything at max optimization.
        if config.aot_inductor.package_cpp_only:
            wrapper_code = "\n".join((wrapper_code, kernel_code))
            kernel_code = ""

        wrapper_key, wrapper_path = write(
            wrapper_code,
            "wrapper.cpp",
            extra=cpp_command,
            specified_dir=specified_output_path,
            key=config.aot_inductor.model_name_for_generated_files,
        )
        kernel_code = (
            f"// Triton kernels are embedded as comments in {wrapper_path}\n"
            + kernel_code
        )
        _, kernel_path = write(
            kernel_code,
            "kernel.cpp",
            extra=cpp_command,
            specified_dir=specified_output_path,
            key=config.aot_inductor.model_name_for_generated_files,
        )

        header_code = ""
        header_path = ""
        if not config.aot_inductor.dynamic_linkage:
            # to link statically, we also need a header file
            with open(
                os.path.join(
                    os.path.dirname(os.path.dirname(__file__)),
                    "csrc",
                    "inductor",
                    "aoti_runtime",
                    "model.h",
                )
            ) as f:
                # model_name_for_generated_files is guaranteed to be non-empty when compile_standalone
                model_class_name = config.aot_inductor.model_name_for_generated_files
                class_name = f"AOTInductorModel{model_class_name}"
                header_code = f.read()

                # we replace like this to avoid replacing
                # AOTInductorModelBase and AOTInductorModelKernelsBase
                header_code = (
                    header_code.replace("<AOTInductorModel>", f"<{class_name}>")
                    .replace("AOTInductorModel(", f"{class_name}(")
                    .replace("AOTInductorModel :", f"{class_name} :")
                )
                _, header_path = write(
                    header_code,
                    "h",
                    specified_dir=specified_output_path,
                    key=model_class_name,
                )

        # Log the AOTInductor wrapper and kernel code, if needed.
        with WritableTempFile("w+") as t:
            """
            Avoid "Permission denied error" on Windows:
            with tempfile.NamedTemporaryFile("w", suffix=".gv") as temp_file:
                # Not writable on Windows:
                # https://docs.python.org/3/library/tempfile.html#tempfile.NamedTemporaryFile

            Example:
                with WritableTempFile("w", suffix=".gv") as temp_file:
                    tree.to_dotfile(temp_file.name)
            """
            t.writelines((wrapper_code, "\n", kernel_code, "\n"))
            t.flush()
            V.debug.output_code(t.name, extension="cpp")

        if config.aot_inductor.package:
            generated_files.append(wrapper_path)
            if not config.aot_inductor.package_cpp_only:
                generated_files.append(kernel_path)
            if not config.aot_inductor.dynamic_linkage:
                generated_files.append(header_path)

        output_code_log.info("Wrapper code written to: %s", wrapper_path)
        output_code_log.info("Kernel code written to: %s", kernel_path)
        trace_structured(
            "graph_dump",
            lambda: {
                "name": "inductor_aot_wrapper_code",
                "type": "cpp",
                "filename": wrapper_path,
            },
            payload_fn=lambda: wrapper_code,
        )
        trace_structured(
            "graph_dump",
            lambda: {
                "name": "inductor_aot_kernel_code",
                "type": "cpp",
                "filename": kernel_path,
            },
            payload_fn=lambda: kernel_code,
        )
        if not config.aot_inductor.dynamic_linkage:
            output_code_log.info("Header code written to: %s", header_path)
            trace_structured(
                "graph_dump",
                lambda: {
                    "name": "inductor_aot_header_code",
                    "type": "cpp",
                    "filename": header_path,
                },
                payload_fn=lambda: header_code,
            )

        # We use a file lock below to protect FS operations. The lock file
        # is scoped to the 'key', so make sure the consts_s is protected
        # by the same lock:
        wrapper_path_operator = Path(wrapper_path)
        kernel_path_operator = Path(kernel_path)
        specified_sub_dir = wrapper_path_operator.parent / wrapper_key
        if not specified_sub_dir.exists():
            specified_sub_dir.mkdir(exist_ok=True)
        cmake_path = str(Path(specified_sub_dir) / "CMakeLists.txt")

        def _compile_consts(consts: bytes, platform: str) -> str:
            # Load from aot_inductor, and update the value on demand.
            use_asm_build: bool = config.aot_inductor.use_consts_asm_build

            if platform == "linux":
                if graph.mutated_buffers & OrderedSet(graph.constants.keys()):
                    # .data section is between .text and .bss. When the size of .data is large,
                    # during the linking, the relocation of .text against .bss may overflow.
                    # Rename it to .ldata so that it won't be in between the .text and .bss section
                    if len(consts) > 2_000_000_000:
                        raise ValueError(
                            "Models with buffer mutation included doesn't support constants greater than 2GB!"
                        )
                    section_attr = '.ldata, "aw"'
                else:
                    section_attr = '.lrodata, "a"'
                symbol_prefix = ""
            elif platform == "darwin":
                section_attr = "__DATA,__data"
                symbol_prefix = "_"
            elif platform == "win32":
                symbol_prefix = ""
                # ASM build is not supported on Windows, force use CPP build.
                use_asm_build = False
            else:
                raise RuntimeError(f"Unsupported platform: {platform}")

            # Intel compiler failed to compile this manually constructed assembly file.
            # Switch XPU to use consts cpp build.
            if device_type == "xpu":
                use_asm_build = False

            is_large_consts = len(consts) > 1024
            is_zero_size_consts = len(consts) == 0

            def format_consts_to_gnu_asm(
                consts: bytes,
                align_bytes: int,
                symbol_prefix: str,
                is_large_consts: bool,
            ) -> tuple[str, str]:
                consts_asm = f"\t.section\t{section_attr}\n"
                consts_asm += f"\t.balign {align_bytes}\n"
                consts_asm += f"\t.globl\t{symbol_prefix}_binary_constants_bin_start\n"
                consts_asm += f"{symbol_prefix}_binary_constants_bin_start:\n"
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
                consts_asm += f".globl\t{symbol_prefix}_binary_constants_bin_end\n"
                consts_asm += f"{symbol_prefix}_binary_constants_bin_end:\n"
                return consts_asm, "weights.S"

            # Use c++ to convert consts to object file can support more compilers, such as msvc and icx.
            def format_consts_to_cpp(
                consts: bytes, align_bytes: int, symbol_prefix: str
            ) -> tuple[str, str]:
                consts_size = len(consts)
                asan_attr = """#if defined(__clang__) || defined (__GNUC__)\t\n\
#define ATTRIBUTE_NO_SANITIZE_ADDRESS __attribute__((no_sanitize("address")))\t\n\
#else\t\n\
#define ATTRIBUTE_NO_SANITIZE_ADDRESS\t\n\
#endif\t\n\
\t\n\
ATTRIBUTE_NO_SANITIZE_ADDRESS\t\n"""
                const_cpp = asan_attr
                const_cpp += f"alignas({align_bytes}) extern "
                const_cpp += f"unsigned char {symbol_prefix}_binary_constants_bin_start[{consts_size}] = {{\t\n"
                count_bytes = 0
                for c in consts:
                    const_cpp += f"{c}, "
                    count_bytes = count_bytes + 1
                    if count_bytes % 16 == 0:
                        const_cpp += "\t\n"
                const_cpp += "};\t\n"
                const_cpp += f"alignas({align_bytes}) extern unsigned char * {symbol_prefix}_binary_constants_bin_end;\t\n"
                return const_cpp, "weights.cpp"

            def get_zero_consts_asm_code(
                align_bytes: int,
                symbol_prefix: str,
            ) -> tuple[str, str]:
                """
                This function handles zero-sized constants because the C++ standard prohibits zero-length arrays:
                https://stackoverflow.com/questions/9722632/what-happens-if-i-define-a-0-size-array-in-c-c

                On Windows (MSVC):
                    The compiler reports error C2466 for zero-sized arrays:
                    https://learn.microsoft.com/en-us/cpp/error-messages/compiler-errors-1/compiler-error-c2466
                    Solution: Use assembly compilation to handle this case.

                Why not use Win32 assembly for all paths?
                    ml64 only supports alignment up to 16 bytes, which isn't optimal for performance.

                Cross-platform implementation:
                    Linux: Added '-pedantic' to disable zero-sized arrays in C++ compiler
                    Windows: MSVC naturally rejects zero-sized arrays by default
                """
                if _IS_WINDOWS:
                    # Windows ml64 is max support align to 16, but it is no effect to zero size data.
                    asm_code = """
option casemap:none
.data
?_binary_constants_bin_start@@3PAEA:
align 16
?_binary_constants_bin_end@@3PAEA:
align 16
public ?_binary_constants_bin_start@@3PAEA
public ?_binary_constants_bin_end@@3PAEA
end
"""
                    asm_ext = "asm"
                else:
                    asm_code = f"\t.section\t{section_attr}\n"
                    asm_code += f"\t.balign {align_bytes}\n"
                    asm_code += (
                        f"\t.globl\t{symbol_prefix}_binary_constants_bin_start\n"
                    )
                    asm_code += f"{symbol_prefix}_binary_constants_bin_start:\n"
                    asm_code += f".globl\t{symbol_prefix}_binary_constants_bin_end\n"
                    asm_code += f"{symbol_prefix}_binary_constants_bin_end:\n"
                    asm_ext = "S"
                return asm_code, asm_ext

            if use_asm_build:
                consts_code, code_ext = format_consts_to_gnu_asm(
                    consts, ALIGN_BYTES, symbol_prefix, is_large_consts
                )
            else:
                if is_zero_size_consts:
                    consts_code, code_ext = get_zero_consts_asm_code(
                        ALIGN_BYTES, symbol_prefix
                    )
                else:
                    consts_code, code_ext = format_consts_to_cpp(
                        consts, ALIGN_BYTES, symbol_prefix
                    )

            _, consts_s = write(
                consts_code,
                code_ext,
                specified_dir=str(specified_sub_dir),
                key=config.aot_inductor.model_name_for_generated_files,
            )
            consts_s = Path(consts_s)
            object_build_options = CppTorchDeviceOptions(
                device_type=device_type,
                aot_mode=graph.aot_mode,
                compile_only=True,
                use_relative_path=use_relative_path,
            )
            object_builder = CppBuilder(
                name=str(consts_s.stem),
                sources=str(consts_s),
                output_dir=str(consts_s.parent),
                BuildOption=object_build_options,
            )
            consts_o = object_builder.get_target_file_path()
            if use_asm_build is False and is_zero_size_consts:
                run_asm_build_object(str(consts_s), consts_o, str(consts_s.parent))
            else:
                object_builder.build()

            if is_large_consts and use_asm_build:
                with open(consts_o, "r+b") as f:
                    f.seek(0)
                    hdr = f.read(1024)
                    # Search for magic number and write the actual data over it
                    start_idx = (
                        hdr.find(b"\xef\xcd\xab\x99\x78\x56\x34\x12")
                        if sys.byteorder == "little"
                        else hdr.find(b"\x12\x34\x56\x78\x99\xab\xcd\xef")
                    )
                    assert start_idx != -1
                    f.seek(start_idx)
                    pos = 0
                    while pos < len(consts):
                        rc = f.write(consts[pos:])
                        pos += rc

            # Remove the .S file to save space
            os.remove(consts_s)

            return consts_o

        from torch.utils._filelock import FileLock

        lock_dir = get_lock_dir()
        lock = FileLock(
            os.path.join(lock_dir, wrapper_key + ".lock"), timeout=LOCK_TIMEOUT
        )
        with lock:
            if serialized_extern_kernel_nodes:
                extern_kernel_nodes_json = str(
                    wrapper_path_operator.with_suffix(".json")
                )
                with open(extern_kernel_nodes_json, "w") as f:
                    f.write(serialized_extern_kernel_nodes)

                if config.aot_inductor.package:
                    generated_files.append(extern_kernel_nodes_json)

            metadata = config.aot_inductor.metadata
            metadata["AOTI_DEVICE_KEY"] = device_type

            # Add environment information to ensure .so compatibility
            metadata.update(get_device_information(device_type))

            # Save user provided metadata
            meta_json = str(
                wrapper_path_operator.with_name(
                    f"{wrapper_path_operator.stem}_metadata.json"
                )
            )
            for k, v in config.aot_inductor.metadata.items():
                assert isinstance(k, str) and isinstance(v, (str)), (
                    "Metadata must only contain strings"
                )

            with open(meta_json, "w") as f:
                f.write(json.dumps(config.aot_inductor.metadata))

            kernel_meta_json = str(
                kernel_path_operator.with_name(
                    f"{kernel_path_operator.stem}_metadata.json"
                )
            )
            shutil.copy(meta_json, kernel_meta_json)

            if config.aot_inductor.package:
                generated_files.append(meta_json)
                if not config.aot_inductor.package_cpp_only:
                    generated_files.append(kernel_meta_json)

            output_so = (
                config.aot_inductor.output_path
                if specified_artifact_name
                else str(wrapper_path_operator.with_suffix(".so"))
            )
            all_cuda = all(
                graph.get_original_value_of_constant(name).is_cuda
                for name in graph.constants
                if name not in graph.folded_constants
            )

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
                # pyrefly: ignore [missing-attribute]
                raw_bytes = bytes(raw_array.contents)
                return raw_bytes if all_cuda else _pad_to_alignment(raw_bytes)

            if (
                config.aot_inductor.package_constants_in_so
                or config.aot_inductor.package_constants_on_disk_format == "binary_blob"
            ):
                serialized_weights = b"".join(
                    _to_bytes(graph.get_original_value_of_constant(name), all_cuda)
                    for name in graph.constants
                    if name not in graph.folded_constants
                )
            else:
                serialized_weights = b""

            if config.aot_inductor.package_constants_on_disk_format == "pickle_weights":
                # We need to return a storage key here because the original value tensor might be a clone
                weights_dict = Weights(
                    {
                        graph.allocated_constant_name[name]: (
                            graph.get_original_value_of_constant(name),
                            TensorProperties(graph.constants[name]),
                        )
                        for name in graph.constants
                        if name not in graph.folded_constants
                    }
                )
                generated_files.append(weights_dict)

            consts_size = len(serialized_weights)

            use_external_weights, use_mmap_weights = determine_aoti_mmap_flags(
                consts_size
            )
            if use_external_weights and use_mmap_weights:
                # Should never reach here, just a check for sanity
                raise RuntimeError(
                    "use_external_weights and  use_mmap_weights cannot both be True."
                )

            external_weights_path = None
            if use_external_weights:
                external_weights_filename = f"{wrapper_path_operator.stem}_weights.blob"
                external_weights_path = str(
                    wrapper_path_operator.with_name(external_weights_filename)
                )

            compile_command: dict[str, Any] = {
                "aot_mode": graph.aot_mode,
                "device_type": device_type,
                "use_mmap_weights": use_mmap_weights,
                "use_mmap_weights_external": use_external_weights,
                "use_relative_path": use_relative_path,
                "vec_isa": picked_vec_isa,
            }
            # If we're packaging via CMake, we build the whole code at max optimization.
            wrapper_build_options = CppTorchDeviceOptions(
                compile_only=True,
                min_optimize=not config.aot_inductor.package_cpp_only,
                **compile_command,
            )
            kernel_build_options = CppTorchDeviceOptions(
                compile_only=True,
                **compile_command,
            )

            # potentially, precompile the AOT header for this device
            if config.aot_inductor.precompile_headers and not _IS_WINDOWS:
                header_file = _get_cpp_wrapper_header(
                    device_type, aot_mode=graph.aot_mode
                )
                wrapper_build_options.precompiled_header = _precompile_header(
                    header_file,
                    cpp_command,
                    min_optimize=not config.aot_inductor.package_cpp_only,
                    **compile_command,
                )
                if cpp_prefix := _get_cpp_prefix_header(device_type):
                    kernel_build_options.precompiled_header = _precompile_header(
                        cpp_prefix,
                        cpp_command,
                        **compile_command,
                    )

            wrapper_builder = CppBuilder(
                name=str(wrapper_path_operator.stem),
                sources=wrapper_path,
                output_dir=str(wrapper_path_operator.parent),
                BuildOption=wrapper_build_options,
            )
            wrapper_compile_cmd = wrapper_builder.get_command_line()
            wrapper_o = wrapper_builder.get_target_file_path()

            kernel_builder = CppBuilder(
                name=str(kernel_path_operator.stem),
                sources=kernel_path,
                output_dir=str(wrapper_path_operator.parent),
                BuildOption=kernel_build_options,
            )
            kernel_compile_cmd = kernel_builder.get_command_line()
            kernel_o = kernel_builder.get_target_file_path()

            log.debug("aot wrapper compilation command: %s", wrapper_compile_cmd)
            log.debug("aot kernel compilation command: %s", kernel_compile_cmd)
            if config.aot_inductor.package_cpp_only:
                # Not doing the actual compilation here
                compile_flags = str(
                    wrapper_path_operator.with_name(
                        f"{wrapper_path_operator.stem}_compile_flags.json"
                    )
                )
                wrapper_build_options.save_flags_to_json(compile_flags)
                generated_files.append(compile_flags)
                wrapper_builder.save_compile_cmd_to_cmake(cmake_path, device_type)
                wrapper_builder.save_src_to_cmake(cmake_path, wrapper_path)
                generated_files.append(cmake_path)
            else:
                try:
                    wrapper_builder.build()
                except (exc.CppCompileError, SkipFrame) as e:
                    if " is too big to optimize" in str(e):
                        raise RuntimeError(
                            "Please use torch._inductor.config.aot_inductor.compile_wrapper_opt_level = 'O0' flag."
                        ) from e
                    raise e
                kernel_builder.build()

            if not use_mmap_weights:
                aot_constants = serialized_weights
                magic_number = 0
                if use_external_weights:
                    aot_constants = struct.pack("q", consts_size)
                    assert external_weights_path is not None
                    # For external weights, write weights to separate file and embed minimal placeholder
                    with open(external_weights_path, "wb") as f_weights:
                        f_weights.write(serialized_weights)
                    generated_files.append(external_weights_path)
            else:
                # we'll append weights binary to the end of .so file and mmap it when loading
                magic_number = cast(
                    int, torch.randint(0, torch.iinfo(torch.int64).max, (1,)).item()
                )
                aot_constants = struct.pack("qq", consts_size + 8, magic_number)

            consts_o = _compile_consts(aot_constants, sys.platform)
            custom_obj_idx = 0
            # Note that custom_objs_config.json file is different from the model_constants_config.json file produced
            # in package_sigmoid(). The keys in custom_objs_config.json directly correspond to the arg name in extern
            # nodes json. The key in model_constants_config.json produced by package_sigmoid is the attribute name in the
            # user model code.

            qual_name_to_id = {}  # Map from constant name to its name in constants folder
            for custom_obj_idx, (name, constant) in enumerate(
                graph.torchbind_constants.items()
            ):
                if isinstance(
                    constant, torch._library.fake_class_registry.FakeScriptObject
                ):
                    constant = constant.real_obj
                assert isinstance(constant, torch._C.ScriptObject)
                custom_obj_name = f"{CUSTOM_OBJ_FILENAME_PREFIX}{custom_obj_idx}"

                log.debug("saving script object %s as %s", name, custom_obj_name)

                qual_name_to_id[name] = custom_obj_name
                custom_obj_bytes = torch._C._pickle_save(constant)
                custom_obj_path = os.path.join(
                    wrapper_path_operator.parent, custom_obj_name
                )

                write_atomic(custom_obj_path, custom_obj_bytes, True)
                generated_files.append(custom_obj_path)

            if qual_name_to_id:
                constants_config_json = os.path.join(
                    wrapper_path_operator.parent, "custom_objs_config.json"
                )
                with open(constants_config_json, "w") as f:
                    f.write(json.dumps(qual_name_to_id))
                generated_files.append(constants_config_json)

            gpu_codecache: ROCmCodeCache | CUDACodeCache = (
                ROCmCodeCache() if torch.version.hip else CUDACodeCache()
            )
            gpu_kernels_o = gpu_codecache.aot_kernels_o.copy()
            # clear the list of aot kernels after each linking
            gpu_codecache.aot_kernels_o.clear()

            if gpu_kernels_o:
                assert not config.aot_inductor.emit_multi_arch_kernel, (
                    "TODO: add emit_multi_arch_kernel support for cutlass kernels"
                )

            cubins_o = []
            asm_files = []
            if not _IS_WINDOWS:
                ld, objcopy = get_ld_and_objcopy(use_relative_path)
                kernels = getattr(V.graph.wrapper_code, "_kernel_name_to_body", {})
                for kernel_name, value in CudaKernelParamCache.cache.items():
                    if kernel_name not in kernels:
                        # It is possible that CudaKernelParamCache contains more Triton kernels
                        # than what the current graph uses
                        continue

                    if asm_file := value["asm"]:
                        asm_files.append(asm_file)

                    cubin_file = value[get_cpp_wrapper_cubin_path_name()]
                    if (
                        config.aot_inductor.emit_multi_arch_kernel
                        and device_type == "cuda"
                    ):
                        if torch.version.hip is None:
                            current_arch = _nvcc_arch_as_compile_option()
                            cmd = (
                                # pyrefly: ignore [unbound-name]
                                f"{_cuda_compiler()} -fatbin {asm_file} -o {cubin_file} "
                                # Triton only allows generating PTX version as same as the current arch
                                f"-gencode arch=compute_{current_arch},code=compute_{current_arch} "
                                # Include SASS for the current specific arch
                                f"-gencode arch=compute_{current_arch},code=sm_{current_arch} "
                            )
                            try:
                                subprocess.run(
                                    cmd.split(),
                                    capture_output=True,
                                    text=True,
                                    check=True,
                                )
                            except subprocess.CalledProcessError as e:
                                print(
                                    f"{cmd} failed with:\nstdout:\n{e.stdout}\nstderr:\n{e.stderr}",
                                    file=sys.stderr,
                                )
                                raise

                        else:
                            # ROCm multi-arch: compile LLVM IR to multi-arch bundle
                            from torch._inductor.rocm_multiarch_utils import (
                                compile_multiarch_bundle_from_llvm_ir,
                            )

                            if not os.path.exists(asm_file):
                                raise RuntimeError(
                                    f"Multi-arch ROCm compilation requires LLVM IR file, "
                                    f"but {asm_file} not found. "
                                    f"Ensure asm_type='ll' is captured in triton_heuristics.py"
                                )

                            # Compile for multiple archs and bundle them
                            success = compile_multiarch_bundle_from_llvm_ir(
                                llvm_ir_path=asm_file,
                                output_bundle_path=cubin_file,
                                target_archs=None,
                            )

                            if not success:
                                raise RuntimeError(
                                    f"Failed to compile multi-arch bundle for kernel {kernel_name}. "
                                    f"Check that ROCm toolchain is available and LLVM IR is valid."
                                )

                            log.info("Created multi-arch bundle: %s", cubin_file)

                    if config.aot_inductor.embed_kernel_binary:
                        # Embed cubin files into model.so using objcopy
                        cubins_o.append(
                            convert_cubin_to_obj(cubin_file, kernel_name, ld, objcopy)
                        )

            output_name, output_dir = get_name_and_dir_from_output_file_path(output_so)
            so_build_options = CppTorchDeviceOptions(
                vec_isa=picked_vec_isa,
                device_type=device_type,
                aot_mode=graph.aot_mode,
                use_relative_path=use_relative_path,
            )

            obj_srcs = [wrapper_o, kernel_o, consts_o, *gpu_kernels_o, *cubins_o]
            so_builder = CppBuilder(
                name=output_name,
                sources=obj_srcs,
                output_dir=output_dir,
                BuildOption=so_build_options,
            )
            link_cmd = so_builder.get_command_line()
            output_so = so_builder.get_target_file_path()

            log.debug("aot linkage command: %s", link_cmd)

            # Append cmds to the end of codegen-ed wrapper file
            with open(wrapper_path, "a") as f:
                f.write("\n")
                f.write(f"// Compile cmd\n// {wrapper_compile_cmd}\n")
                f.write(f"// Link cmd\n// {link_cmd}\n")

            with open(kernel_path, "a") as f:
                f.write("\n")
                f.write(f"// Compile cmd\n// {kernel_compile_cmd}\n")
                f.write(f"// Link cmd\n// {link_cmd}\n")

            if config.aot_inductor.package_cpp_only:
                linker_flags = str(
                    wrapper_path_operator.with_name(
                        f"{wrapper_path_operator.stem}_linker_flags.json"
                    )
                )
                so_build_options.save_flags_to_json(linker_flags)
                generated_files.append(linker_flags)
                generated_files.append(_LINKER_SCRIPT)

                # If we only want to package the cpp, then we need to save the
                # weights separately into a bin, and we also need to prevent compiling the so
                if use_mmap_weights:
                    weight_file = str(
                        wrapper_path_operator.with_name(
                            f"{wrapper_path_operator.stem}_serialized_weights.bin"
                        )
                    )
                    with open(weight_file, "wb") as f_weights:
                        f_weights.write(serialized_weights)
                        f_weights.write(struct.pack("q", magic_number))

                    generated_files.append(weight_file)
                else:
                    # TODO: unify to always use mmap_weights
                    generated_files.append(consts_o)
                    so_builder.save_src_to_cmake(cmake_path, consts_o)

                # Different CMake strategies for CUDA vs ROCm:
                # - CUDA: Save asm for CMake to recompile (user has nvcc)
                # - ROCm: Link pre-compiled bundle (user may lack dev tools)
                if (
                    config.aot_inductor.emit_multi_arch_kernel
                    and torch.version.hip is None
                ):
                    so_builder.save_kernel_asm_to_cmake(cmake_path, asm_files)
                    generated_files.extend(asm_files)
                else:
                    # ROCm multi-arch + all single-arch: Link pre-compiled objects
                    # Bundle already embedded in .o files - just link into .so
                    obj_srcs = [*gpu_kernels_o, *cubins_o]
                    generated_files.extend(obj_srcs)
                    for obj in obj_srcs:
                        so_builder.save_src_to_cmake(cmake_path, obj)

                so_builder.save_link_cmd_to_cmake(cmake_path)
            else:
                so_builder.build()
                for o_file in obj_srcs:
                    if o_file in gpu_kernels_o:
                        continue
                    # Remove these as they are not needed anymore
                    os.remove(o_file)

                if use_mmap_weights:
                    if config.aot_inductor.cross_target_platform == "windows":
                        raise RuntimeError(
                            "when cross_target_platform is windows, use_mmap_weights should not be true."
                        )

                    def get_page_size() -> int:
                        # Don't use resource.getpagesize() on Windows, as it is a Unix specific package
                        # as seen in https://docs.python.org/2/library/resource.html
                        if _IS_WINDOWS:
                            from ctypes import (  # type: ignore[attr-defined]
                                byref,
                                Structure,
                                windll,
                            )
                            from ctypes.wintypes import DWORD, LPVOID, WORD

                            class SYSTEM_INFO(Structure):
                                _fields_ = [
                                    ("wProcessorArchitecture", WORD),
                                    ("wReserved", WORD),
                                    ("dwPageSize", DWORD),
                                    ("lpMinimumApplicationAddress", LPVOID),
                                    ("lpMaximumApplicationAddress", LPVOID),
                                    ("dwActiveProcessorMask", DWORD),
                                    ("dwNumberOfProcessors", DWORD),
                                    ("dwProcessorType", DWORD),
                                    ("dwAllocationGranularity", DWORD),
                                    ("wProcessorLevel", WORD),
                                    ("wProcessorRevision", WORD),
                                ]

                            si = SYSTEM_INFO()
                            windll.kernel32.GetSystemInfo(byref(si))
                            sys_page_size = si.dwPageSize
                        else:
                            import resource

                            sys_page_size = resource.getpagesize()

                        return sys_page_size

                    page_size_ = get_page_size()
                    page_size = max(16384, page_size_)

                    with open(output_so, "a+b") as f_so:
                        so_size = f_so.tell()
                        # Page align the weights
                        f_so.write(b" " * (page_size - so_size % page_size))
                        f_so.write(serialized_weights)
                        f_so.write(struct.pack("q", magic_number))

                if config.aot_inductor.package:
                    generated_files.append(output_so)

        if config.trace.provenance_tracking_level != 0:
            kernel_info = torch._inductor.debug.create_kernel_information_json()
            kernel_info_json = os.path.join(
                wrapper_path_operator.parent, "kernel_information.json"
            )
            with open(kernel_info_json, "w") as f:
                f.write(json.dumps(kernel_info, indent=4))
            generated_files.append(kernel_info_json)

        if config.aot_inductor.package:
            # We want to return the directory that contains all the AOTI
            # generated files, not just the so
            # return os.path.split(output_so)[0]
            return generated_files

        return output_so


_libgomp: CDLL | None = None


def custom_op_wrapper(op: str, *args: Any) -> list[c_void_p] | c_void_p | None:
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

    # convert any kwarg-only arguments to kwargs
    kwargs = dict()
    # pyrefly: ignore [missing-attribute]
    for func_arg, conv_arg in zip(func._schema.arguments, converted_args):
        if func_arg.kwarg_only:
            kwargs[func_arg.name] = conv_arg
    if kwargs:
        del converted_args[-len(kwargs) :]

    result = func(*converted_args, **kwargs)
    if result is None:
        return None

    if isinstance(result, (list, tuple)):
        # unsafe_alloc_void_ptrs_from_tensors expects result contains tensor only
        result = [torch.tensor([]) if r is None else r for r in result]
        for r in result:
            assert isinstance(r, torch.Tensor), op + " returns a list of non-tensors"
        return torch._C._aoti.unsafe_alloc_void_ptrs_from_tensors(result)  # type: ignore[arg-type]

    assert isinstance(result, torch.Tensor), op + " returns a non-tensor"
    return torch._C._aoti.unsafe_alloc_void_ptr_from_tensor(result)


# Precompiled headers are persistent past program runtime, but associated with one
# specific compiler version and set of flags.  We explicitly use default_cache_dir here
# because these headers need to be global, rather than ignored by fresh_cache.
_HEADER_DIR = os.path.join(default_cache_dir(), "precompiled_headers")
_HEADER_LOCK_DIR = os.path.join(_HEADER_DIR, "locks")


@functools.cache
def _precompile_header(
    header: str,
    hashable_cmd_line: str,
    **compile_command: Any,
) -> str:
    assert not _IS_WINDOWS, (
        "CppBuilder does not currently support precompiling on Windows!"
    )

    # Get the preprocessed output from the header file to be precompiled.  This allows
    # us to properly invalidate the file cache when any header dependency changes.  This
    # is thread-safe, as each thread will get its own temporary directory.
    #
    # N.B. we can't use NamedTemporaryFile here because Windows errors out on attempts
    # to read from a file with an open write handle.
    with tempfile.TemporaryDirectory() as preprocessing_dir:
        preprocessing_header = Path(preprocessing_dir) / "header.hpp"
        preprocessing_header.write_text(f"#include <{header}>\n")
        preprocessor = CppBuilder(
            name=str(preprocessing_header)[:-4],  # strip off the .hpp extension
            sources=str(preprocessing_header),
            BuildOption=CppTorchDeviceOptions(**compile_command, preprocessing=True),
        )
        preprocessor.build()

        def _get_file_checksum(filename: str) -> str:
            """Reading the whole preprocessed header in for hashing is very expensive,
            but calling a fast hashing utility in a subprocess is cheap."""
            # If Windows support needs to be added here, use certutil -hashfile.
            cmd_output = subprocess.run(
                ("openssl", "sha512", filename), capture_output=True, text=True
            )
            return cmd_output.stdout.split()[-1]

        preprocessor_hash = _get_file_checksum(preprocessor.get_target_file_path())

    header_build_option = CppTorchDeviceOptions(**compile_command, precompiling=True)
    header_hash, header_full_path = write(
        content=f"#include <{header}>\n",
        extension="h",
        extra=(
            hashable_cmd_line
            + preprocessor_hash
            + get_compiler_version_info(header_build_option.get_compiler())
        ),
        specified_dir=_HEADER_DIR,
    )
    cpp_builder = CppBuilder(
        name=header_full_path,
        sources=header_full_path,
        BuildOption=header_build_option,
    )
    # _worker_compile_cpp will automatically ignore any compilation whose result already
    # exists, so this is always safe.
    os.makedirs(_HEADER_LOCK_DIR, exist_ok=True)
    _worker_compile_cpp(
        os.path.join(_HEADER_LOCK_DIR, f"{header_hash}.lock"),
        (cpp_builder,),
    )

    return header_full_path


def _get_cpp_prefix_header(device: str) -> str | None:
    if device.startswith("cpu"):
        return "torch/csrc/inductor/cpp_prefix.h"
    return None


def _get_cpp_wrapper_header(device: str, aot_mode: bool = False) -> str:
    """Given a device type (and optionally whether we're in AOT Inductor mode), returns
    the path to the cpp_wrapper header file to be precompiled."""
    base_device = device.split(":", maxsplit=1)[0]
    is_array_ref = config.aot_inductor.allow_stack_allocation and base_device == "cpu"
    return (
        "torch/csrc/inductor/"
        f"{'aoti_include' if aot_mode else 'cpp_wrapper'}/"
        f"{'array_ref' if is_array_ref else base_device}.h"
    )


@clear_on_fresh_cache
class CppCodeCache:
    """Compiles and caches C++ libraries.  Users of this class supply the source code to
    be compiled, while compilation flags are set by CppBuilder."""

    cache: dict[str, Callable[[], CDLL | ModuleType]] = {}
    cache_clear = staticmethod(cache.clear)
    cpp_compile_command_flags: dict[str, Any] = {}

    @staticmethod
    def _load_library_inner(path: str, key: str) -> CDLL | ModuleType:
        return cdll.LoadLibrary(path)

    @classmethod
    def _load_library(cls, path: str, key: str) -> CDLL | ModuleType:
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
    def _get_uncompiled_header(cls, device: str) -> str | None:
        """
        Given a device type, returns the path to a CPP header file to be precompiled.
        """
        return None

    @classmethod
    def load_async(
        cls,
        main_code: str,
        device_type: str = "cpu",
        submit_fn: Any = None,
        extra_flags: Sequence[str] = (),
        optimized_code: str | None = None,
    ) -> Any:
        """Compile and load a C++ library.  Returns a callable that returns the loaded
        library."""
        compile_command = {
            **cls.cpp_compile_command_flags,
            "device_type": device_type,
            "extra_flags": extra_flags,
            "use_relative_path": config.is_fbcode(),
            "vec_isa": pick_vec_isa(),
        }

        _set_gpu_runtime_env()  # cpp_extension consults the env

        # Note the distinction between the two booleans.  We do minimal optimization if
        # the optimized_code argument is present at all, since that's how the user of
        # this function opts in, but we do compilation and linking in one step if the
        # optimized_code argument is empty (as a micro-optimization).
        main_build_option = CppTorchDeviceOptions(
            compile_only=bool(optimized_code),
            min_optimize=optimized_code is not None,
            # pyrefly: ignore [bad-argument-type]
            **compile_command,
        )
        optimized_build_option = CppTorchDeviceOptions(
            # pyrefly: ignore [bad-argument-type]
            compile_only=True,
            # pyrefly: ignore [bad-argument-type]
            **compile_command,
        )

        def get_hashable_command_line(build_option: BuildOptionsBase) -> str:
            """Writing the code to file will calculate a hash, which we need to vary if
            the command line flags change.  This implements a mostly-generic way of
            validating that."""
            return CppBuilder(
                name="o", sources="i", BuildOption=build_option
            ).get_command_line()

        main_cmd_line = get_hashable_command_line(main_build_option)
        optimized_cmd_line = get_hashable_command_line(optimized_build_option)

        key, main_path = write(
            main_code, "main.cpp", extra=f"{optimized_code} {main_cmd_line}"
        )

        # Don't bother writing if the argument is empty.
        if optimized_code:
            _, optimized_path = write(
                optimized_code, "optimized.cpp", extra=optimized_cmd_line
            )
        else:
            # Unused, but makes type checkers happy.
            optimized_path = os.devnull

        if key not in cls.cache:
            from torch.utils._filelock import FileLock

            lock_path = os.path.join(get_lock_dir(), key + ".lock")
            future: Future[Any] | None = None
            lib = None

            # if requested, pre-compile any headers
            if config.cpp_cache_precompile_headers and not _IS_WINDOWS:
                if header := cls._get_uncompiled_header(device_type):
                    main_build_option.precompiled_header = _precompile_header(
                        header,
                        main_cmd_line,
                        min_optimize=optimized_code is not None,
                        **compile_command,
                    )

                # Currently, the optimized_code field is only used for cpp kernel code,
                # so go ahead and precompile the relevant header here.  Revisit this
                # decision if that ever changes.
                if optimized_code and (header := _get_cpp_prefix_header(device_type)):
                    optimized_build_option.precompiled_header = _precompile_header(
                        # pyrefly: ignore [unbound-name]
                        header,
                        optimized_cmd_line,
                        **compile_command,
                    )

            main_name, output_dir = get_name_and_dir_from_output_file_path(main_path)
            main_builder = CppBuilder(
                name=main_name,
                sources=main_path,
                BuildOption=main_build_option,
                output_dir=output_dir,
            )

            if optimized_code:
                optimized_name, _ = get_name_and_dir_from_output_file_path(
                    optimized_path
                )
                optimized_builder = CppBuilder(
                    name=optimized_name,
                    sources=optimized_path,
                    BuildOption=optimized_build_option,
                    output_dir=output_dir,
                )

                linker = CppBuilder(
                    name=main_name,
                    sources=[
                        main_builder.get_target_file_path(),
                        optimized_builder.get_target_file_path(),
                    ],
                    # pyrefly: ignore [bad-argument-type]
                    BuildOption=CppTorchDeviceOptions(**compile_command),
                    output_dir=output_dir,
                )

                worker_fn = functools.partial(
                    _worker_compile_cpp,
                    lock_path,
                    (main_builder, optimized_builder, linker),
                )
                binary_path = normalize_path_separator(linker.get_target_file_path())
            else:
                worker_fn = functools.partial(
                    _worker_compile_cpp, lock_path, (main_builder,)
                )
                binary_path = normalize_path_separator(
                    main_builder.get_target_file_path()
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
    def load(cls, *args: Any, **kwargs: Any) -> Any:
        return cls.load_async(*args, **kwargs)()


def _worker_compile_cpp(
    lock_path: str,
    cpp_builders: Sequence[CppBuilder],
) -> None:
    from torch.utils._filelock import FileLock

    with FileLock(lock_path, timeout=LOCK_TIMEOUT):
        for builder in cpp_builders:
            if not os.path.exists(builder.get_target_file_path()):
                builder.build()


# Customized Python binding for cpp kernels
@clear_on_fresh_cache
class CppPythonBindingsCodeCache(CppCodeCache):
    cache: dict[str, Callable[[], CDLL | ModuleType]] = {}
    cache_clear = staticmethod(cache.clear)
    cpp_compile_command_flags = {
        # kernels have no dependency on libtorch
        "include_pytorch": False,
        "shared": True,
    }
    entry_function = "kernel"
    call_entry_function = "kernel({}); Py_RETURN_NONE;"
    extra_parse_arg = ""
    suffix_template = textwrap.dedent(
        """
        // Python bindings to call {entry_func}():
        #define PY_SSIZE_T_CLEAN
        #include <Python.h>
        #include <sstream>
        #include <cstdlib>
        #include <cerrno>

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

        template <typename T> static inline T parse_arg(PyObject* args, size_t n) {{
            static_assert(std::is_pointer_v<T>, "arg type must be pointer or long");
            return static_cast<T>(_torchinductor_pyobject_tensor_data_ptr(PyTuple_GET_ITEM(args, n)));
        }}
        template <> inline int64_t parse_arg<int64_t>(PyObject* args, size_t n) {{
            auto result = PyLong_AsSsize_t(PyTuple_GET_ITEM(args, n));
            if(unlikely(result == -1 && PyErr_Occurred()))
                throw std::runtime_error("expected int arg");
            return result;
        }}
        template <> inline uintptr_t parse_arg<uintptr_t>(PyObject* args, size_t n) {{
            auto result = PyLong_AsVoidPtr(PyTuple_GET_ITEM(args, n));
            if(unlikely(result == reinterpret_cast<void*>(-1) && PyErr_Occurred()))
                throw std::runtime_error("expected int arg");
            return reinterpret_cast<uintptr_t>(result);
        }}
        template <> inline float parse_arg<float>(PyObject* args, size_t n) {{
            auto result = PyFloat_AsDouble(PyTuple_GET_ITEM(args, n));
            if(unlikely(result == -1.0 && PyErr_Occurred()))
                throw std::runtime_error("expected float arg");
            return static_cast<float>(result);
        }}

        {extra_parse_arg}

        static PyObject* {entry_func}_py(PyObject* self, PyObject* args) {{
            try {{
                if(unlikely(!PyTuple_CheckExact(args)))
                    throw std::runtime_error("tuple args required");
                if(unlikely(PyTuple_GET_SIZE(args) != {arg_len}))
                    throw std::runtime_error("requires {arg_len} args");
                {call_entry_func}
            }} catch(std::exception const& e) {{
                PyErr_SetString(PyExc_RuntimeError, e.what());
                return nullptr;
            }} catch(...) {{
                PyErr_SetString(PyExc_RuntimeError, "unhandled error");
                return nullptr;
            }}
        }}

        static PyMethodDef py_methods[] = {{
            {{"{entry_func}", {entry_func}_py, METH_VARARGS, ""}},
            {{NULL, NULL, 0, NULL}}}};

        static struct PyModuleDef py_module =
            {{PyModuleDef_HEAD_INIT, "{entry_func}", NULL, -1, py_methods}};

        PyMODINIT_FUNC PyInit_{entry_func}(void) {{
            const char* str_addr = std::getenv("_TORCHINDUCTOR_PYOBJECT_TENSOR_DATA_PTR");
            if(!str_addr) {{
                PyErr_SetString(PyExc_RuntimeError, "_TORCHINDUCTOR_PYOBJECT_TENSOR_DATA_PTR must be set");
                return nullptr;
            }}

            char* endptr = nullptr;
            errno = 0;
            uintptr_t addr = std::strtoull(str_addr, &endptr, 10);
            if(errno != 0 || endptr == str_addr || addr == 0) {{
                PyErr_SetString(PyExc_RuntimeError, "Failed to parse _TORCHINDUCTOR_PYOBJECT_TENSOR_DATA_PTR");
                return nullptr;
            }}
            _torchinductor_pyobject_tensor_data_ptr =
                reinterpret_cast<decltype(_torchinductor_pyobject_tensor_data_ptr)>(addr);
            PyObject* module = PyModule_Create(&py_module);
            if (module == NULL) {{
                return NULL;
            }}
            #ifdef Py_GIL_DISABLED
                PyUnstable_Module_SetGIL(module, Py_MOD_GIL_NOT_USED);
            #endif
            return module;
        }}
        """
    )

    @classmethod
    # pyrefly: ignore [bad-override]
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
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module

    @classmethod
    def _get_uncompiled_header(cls, device: str) -> str | None:
        return _get_cpp_prefix_header(device)

    @classmethod
    def load_pybinding_async(
        cls,
        argtypes: Sequence[str],
        main_code: str,
        device_type: str = "cpu",
        num_outputs: int = -1,
        submit_fn: Any = None,
        extra_flags: Sequence[str] = (),
        kernel_code: str | None = None,
    ) -> Any:
        """
        Wrap a C++ function in fast Python bindings.

        Args:
            argtypes: The types of args to ENTRY_FUNCTION(), e.g. ["float*", "long"]
            main_code: C++ source code containing ENTRY_FUNCTION().  Will be built at
                -O3 if kernel_code is None (to maximize performance in any kernels that
                are present), or -O1 otherwise (to minimize compile time).
            kernel_code: If present, C++ source code that will be built at -O3 and
                linked to main_code.

        Returns:
            A python version of ENTRY_FUNCTION()
        """
        parseargs = ", ".join(
            f"parse_arg<{argtype.replace('const ', '')}>(args, {n})"
            for n, argtype in enumerate(argtypes)
        )
        suffix = cls.suffix_template.format(
            arg_len=len(argtypes),
            call_entry_func=cls.call_entry_function.format(parseargs),
            entry_func=cls.entry_function,
            extra_parse_arg=cls.extra_parse_arg.format(array_len=num_outputs),
        )
        get_result = cls.load_async(
            main_code + suffix,
            device_type,
            submit_fn=submit_fn,
            extra_flags=extra_flags,
            optimized_code=kernel_code,
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


@clear_on_fresh_cache
class CppWrapperCodeCache(CppPythonBindingsCodeCache):
    cache: dict[str, Callable[[], CDLL | ModuleType]] = {}
    cache_clear = staticmethod(cache.clear)
    cpp_compile_command_flags = {
        "include_pytorch": True,
        "shared": True,
    }
    entry_function = "inductor_entry_cpp"
    call_entry_function = "return inductor_entry_cpp({});"
    extra_parse_arg = textwrap.dedent(
        """
        #include <torch/csrc/inductor/aoti_torch/c/shim.h>

        static inline std::vector<AtenTensorHandle> unpack_tensor_handle_list(PyObject* pyvec) {{
            std::vector<AtenTensorHandle> result;
            size_t result_len = PyList_GET_SIZE(pyvec);
            result.reserve(result_len);
            for (size_t i = 0; i < result_len; i++) {{
                // AtenTensorHandle is essentially a pointer
                void* elem = PyCapsule_GetPointer(PyList_GET_ITEM(pyvec, i), NULL);
                result.push_back(reinterpret_cast<AtenTensorHandle>(elem));
            }}
            return result;
        }}

        static inline PyObject* pack_tensor_handle_list(const std::array<AtenTensorHandle, {array_len}>& arr) {{
            PyObject* result = PyList_New({array_len});
            for (size_t i = 0; i < {array_len}; i++) {{
                PyObject *elem =
                    arr[i] == nullptr
                        ? Py_None
                        // Store AtenTensorHandle as PyCapsulate
                        : PyCapsule_New(reinterpret_cast<void*>(arr[i]), NULL, NULL);
                PyList_SET_ITEM(result, i, elem);
            }}
            return result;
        }}

        template <> inline std::vector<AtenTensorHandle> parse_arg<std::vector<AtenTensorHandle>>(PyObject* args, size_t n) {{
            return unpack_tensor_handle_list(PyTuple_GET_ITEM(args, n));
        }}

        PyObject* inductor_entry_cpp(std::vector<AtenTensorHandle>&& input_handles) {{
            // For outputs, we only allocate an array to hold returned tensor handles,
            // not the actual output tensor storage.
            std::array<AtenTensorHandle, {array_len}> output_handles{{}};
            try {{
                inductor_entry_impl(input_handles.data(), output_handles.data());
                if (PyErr_Occurred()) {{
                    return nullptr;
                }}
                return pack_tensor_handle_list(output_handles);
            }} catch(std::exception const& e) {{
                PyErr_SetString(PyExc_RuntimeError, e.what());
                return nullptr;
            }} catch(...) {{
                PyErr_SetString(PyExc_RuntimeError, "unhandled error");
                return nullptr;
            }}
        }}
        """
    )

    @classmethod
    def _get_uncompiled_header(cls, device: str) -> str | None:
        return _get_cpp_wrapper_header(device)


@clear_on_fresh_cache
class HalideCodeCache(CppPythonBindingsCodeCache):
    cache: dict[str, Callable[[], ModuleType | CDLL]] = {}
    cache_clear = staticmethod(cache.clear)
    _standalone_runtime_path: str | None = None
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
    def _codegen_buffer(cls, name: str, arg: HalideInputSpec, cuda: bool) -> list[str]:
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
            f"halide_dimension_t {name}_dims[] = {{{', '.join(dims)}}};"
            if len(dims) > 0
            else f"halide_dimension_t * {name}_dims = nullptr;",
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
                # pyrefly: ignore [bad-argument-type]
                buffer_names.append(f"&hl_buf_{i}")
                buffers.extend(cls._codegen_buffer(f"hl_buf_{i}", arg, is_cuda))
            else:
                assert "*" not in arg.ctype
                # pyrefly: ignore [bad-argument-type]
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
    @functools.cache
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
    @functools.cache
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
    @functools.cache
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
        jobs: list[Any] = []
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
            device_type="cuda" if meta.is_cuda() else "cpu",
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
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        libname = "libStandaloneHalideRuntime.so"
        target = "host-cuda" if device_type == "cuda" else "host"
        if cls._standalone_runtime_path:
            assert not os.path.exists(cls._standalone_runtime_path)
            # We hit this case in unittests when we run with fresh_cache()
            # Generating a fresh runtime over and over causes errors because we initialize
            # cuda hundreds of times in the same process and run out of file descriptors.
            # Workaround by jail breaking the current fresh_cache().
            base = default_cache_dir()
        else:
            base = cache_dir()
        dirpath = Path(base) / f"halide-runtime-{target}-{cls.config_hash()}"
        os.makedirs(dirpath, exist_ok=True)
        done_file = str(dirpath / "done")
        lock_file = str(dirpath / "lock")
        hook_file = str(dirpath / "hooks.cpp")
        a_file = str(dirpath / "standalone_halide_runtime.a")
        so_file = str(dirpath / libname)
        if not os.path.exists(done_file):
            import halide as hl  # type: ignore[import-untyped,import-not-found]

            from torch.utils._filelock import FileLock

            with FileLock(lock_file, LOCK_TIMEOUT):
                if not os.path.exists(done_file):
                    with open(hook_file, "w") as f:
                        if device_type == "cuda":
                            f.write(
                                cls.standalone_runtime_cuda_init.format(
                                    cls.find_header("HalideRuntimeCuda.h")
                                )
                            )
                    hl.compile_standalone_runtime(a_file, hl.Target(target))

                    name, output_dir = get_name_and_dir_from_output_file_path(so_file)
                    halide_cmd_gen = CppBuilder(
                        name=name,
                        sources=[hook_file, a_file],
                        output_dir=output_dir,
                        BuildOption=CppTorchDeviceOptions(
                            device_type=device_type,
                        ),
                    )

                    subprocess.check_call(
                        shlex.split(halide_cmd_gen.get_command_line())
                    )
                    touch(done_file)
        assert os.path.exists(so_file)
        cls._standalone_runtime_path = so_file
        return so_file

    @classmethod
    def _get_uncompiled_header(cls, device: str) -> str | None:
        """Header precompiling is currently disabled for halide."""
        return None


def _worker_task_halide(lockfile: str, jobs: list[partial[Any]]) -> None:
    from torch.utils._filelock import FileLock

    try:
        with FileLock(lockfile, LOCK_TIMEOUT):
            for job in jobs:
                job()
    except subprocess.SubprocessError as e:
        if os.environ.get("HALIDE_REPRO") == "1":
            cmd: list[Any]
            python, script, *cmd = getattr(e, "cmd", ("", "", ""))
            if os.path.basename(python).startswith("python"):
                code = Path(script).read_text()
                main = "    hl.main()"
                assert code.count(main) == 1

                class Out:
                    def __repr__(self) -> str:
                        return "out"

                ci = cmd.index("-o")
                assert isinstance(ci, int)
                # pyrefly: ignore [unsupported-operation]
                cmd[ci + 1] = Out()
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


def touch(filename: str) -> None:
    with open(filename, "a"):
        pass


@clear_on_fresh_cache
class PyCodeCache:
    # Track the loaded modules so we can remove the on-disk artifacts when
    # clearing the cache. Note also that we may load the same path more
    # than once, but attach different attributes, i.e., due to different
    # constant values.
    modules: list[ModuleType] = []

    # Modules loaded without extra attributes are stored here, those do not
    # need to be re-loaded.
    modules_no_attr: dict[str, ModuleType] = {}

    linemaps: dict[str, list[tuple[Any, ...]]] = {}

    @classmethod
    def write(cls, source_code: str, extra: str = "") -> tuple[str, str]:
        return write(source_code, "py", extra=extra)

    @classmethod
    def load(cls, source_code: str, extra: str = "") -> ModuleType:
        key, path = write(source_code, "py", extra=extra)
        return cls.load_by_key_path(key, path)

    @classmethod
    def load_by_key_path(
        cls,
        key: str,
        path: str,
        linemap: list[tuple[int, str]] | None = None,
        attrs: dict[str, Any] | None = None,
    ) -> ModuleType:
        if linemap is None:
            linemap = []

        # we only cache when attrs is None
        if attrs is None and path in cls.modules_no_attr:
            return cls.modules_no_attr[path]

        in_toplevel = in_toplevel_process()
        mod = _reload_python_module(key, path, set_sys_modules=in_toplevel)

        # unzip into separate lines/nodes lists
        if in_toplevel:
            cls.linemaps[path] = list(zip(*linemap))

        if attrs is not None:
            for k, v in attrs.items():
                setattr(mod, k, v)

        if in_toplevel:
            # we only cache when attrs is None
            if attrs is None:
                cls.modules_no_attr[path] = mod

            cls.modules.append(mod)
        return mod

    @classmethod
    def cache_clear(cls, purge: bool = False) -> None:
        """
        Clear the in-memory module cache. If purge=True, also delete all the
        corresponding on-disk source files.
        """
        if purge:
            for mod in cls.modules:
                try:
                    assert mod.__file__
                    os.remove(mod.__file__)
                except FileNotFoundError:
                    pass
        cls.modules.clear()
        cls.modules_no_attr.clear()

    @classmethod
    @functools.cache
    def stack_frames_for_code(
        cls, path: str, lineno: int
    ) -> list[dict[str, Any]] | None:
        if path not in cls.linemaps:
            return None
        if len(cls.linemaps[path]) == 0:
            return None
        # [(starting_line, <fx node>), ...]
        lines, nodes = cls.linemaps[path]
        p = bisect_right(lines, lineno)
        if p == 0:
            return None
        entry = nodes[p - 1]
        if not entry:
            return None

        def parse_stack_trace(stack_trace: str) -> list[dict[str, Any]]:
            # ideally fx stores stack traces as data rather than a string
            # but this is not along a performance critical path
            regex = r'File "(.+)", line (\d+), in (.+)\n'
            matches = re.findall(regex, stack_trace)
            return [
                {"filename": f, "line": int(l), "name": n}
                for f, l, n in reversed(matches)
            ]

        return parse_stack_trace(entry)


def _load_triton_kernel_from_source(
    kernel_name: str, source_code: str
) -> CachingAutotuner:
    return getattr(PyCodeCache.load(source_code), kernel_name)


def _cuda_compiler() -> str | None:
    if cuda_env.nvcc_exist(config.cuda.cuda_cxx):
        return config.cuda.cuda_cxx
    if config.is_fbcode():
        return os.path.join(build_paths.sdk_home, "bin", "nvcc")
    if cuda_env.nvcc_exist(os.getenv("CUDACXX")):
        return os.getenv("CUDACXX", "")
    if cuda_env.nvcc_exist(os.getenv("CUDA_HOME")):
        return os.path.realpath(os.path.join(os.getenv("CUDA_HOME", ""), "bin/nvcc"))
    return "nvcc"


def _cutlass_path() -> str:
    if config.is_fbcode():
        from libfb.py import parutil

        return parutil.get_dir_path("cutlass-4-headers")
    else:
        return config.cuda.cutlass_dir


def _cutlass_paths() -> list[str]:
    return [
        "include",
        "tools/library/include",
        "tools/library/src",
        "tools/util/include",
    ]


def _clone_cutlass_paths(build_root: str) -> list[str]:
    paths = _cutlass_paths()
    cutlass_root = _cutlass_path()
    for path in _cutlass_paths():
        old_path = os.path.join(cutlass_root, path)
        new_path = os.path.join(build_root, path)
        shutil.copytree(old_path, new_path, dirs_exist_ok=True)
    return paths


def _cutlass_include_paths() -> list[str]:
    cutlass_path = _cutlass_path()
    return [
        # Use realpath to get canonical absolute paths, in order not to mess up cache keys
        os.path.realpath(os.path.join(cutlass_path, path))
        for path in _cutlass_paths()
    ]


@torch_key_cache
def cutlass_key() -> bytes:
    """
    Compute a key representing the state of the CUTLASS library.

    Note: OSS and fbcode will have different keys.
    """
    if config.is_fbcode():
        with (
            importlib.resources.path(
                "cutlass_library", "src_hash.txt"
            ) as resource_path,
            open(resource_path) as resource_file,
        ):
            return resource_file.read().encode()

    combined_hash = hashlib.sha256()
    build_code_hash([config.cuda.cutlass_dir], "", combined_hash)
    return combined_hash.digest()


def _cuda_lib_options() -> list[str]:
    """
    Util function for CUTLASS backend to find the correct CUDA libraries.
    """
    _set_gpu_runtime_env()  # cpp_extension consults the env
    from torch.utils import cpp_extension

    lpaths = cpp_extension.library_paths(device_type="cuda")
    if use_re_build():
        lpaths += [
            build_paths.sdk_lib,
            os.path.join(build_paths.sdk_lib, "stubs"),
        ]
    extra_ldflags: list[str] = []
    if is_linux():
        _transform_cuda_paths(lpaths)
        for path in lpaths:
            if "torch/lib" in path:
                # don't want to depend on pytorch
                continue
            extra_ldflags.append(f"-L{path}")
            # -rpath ensures the DLL can find its dependencies when loaded, even
            # if the library path is non-standard.
            # But do not add the stubs folder to rpath as the driver is expected to be found at runtime
            if os.path.basename(path) != "stubs":
                extra_ldflags.extend(["-Xlinker", f"-rpath={path}"])
        extra_ldflags.append("-lcuda")
        extra_ldflags.append("-lcudart")
    else:
        raise NotImplementedError(
            "Unsupported env, failed to find cuda libs! Currently only Linux is supported."
        )
    return extra_ldflags


def _nvcc_host_compiler_options() -> list[str]:
    return [
        "-fPIC",
        "-fno-strict-aliasing",
        "-fvisibility=hidden",
        "-Wconversion",
    ]


def _nvcc_arch_as_compile_option() -> str:
    arch = cuda_env.get_cuda_arch()
    if arch == "90":
        # Required by cutlass compilation.
        return "90a"
    if arch == "103":
        return "100f"
    if arch == "100":
        return "100a"
    return arch


def _nvcc_compiler_options() -> list[str]:
    arch = _nvcc_arch_as_compile_option()
    code = [f"sm_{arch}", f"compute_{arch}"]
    if config.cuda.enable_cuda_lto:
        code += [f"lto_{arch}"]
    options = [
        "-t=0",
        "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
        "-DCUTLASS_ENABLE_SM90_EXTENDED_MMA_SHAPES=1",
        "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED",
        "-w",
        f"-gencode=arch=compute_{arch},code=[{','.join(code)}]",
        config.cuda.compile_opt_level,
        "-std=c++17",
        "--expt-relaxed-constexpr",
        "-DNDEBUG",
    ]
    if config.is_fbcode():
        options.extend(["-ccbin", os.path.dirname(build_paths.gcc)])
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
    src_files: list[str],
    dst_file: str,
    dst_file_ext: str,
    extra_args: list[str] | None = None,
) -> str:
    if extra_args is None:
        extra_args = []
    if use_re_build():
        build_path = os.path.dirname(dst_file)
        include_paths = _clone_cutlass_paths(build_path)
        src_files = [os.path.basename(src_file) for src_file in src_files]
        dst_file = os.path.basename(dst_file)
    else:
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
    if log.isEnabledFor(logging.DEBUG):
        log.debug("CUDA command: %s", res)
    else:
        autotuning_log.debug("CUDA command: %s", res)
    return res


class DLLWrapper:
    """A wrapper for a dynamic library."""

    def __init__(
        self,
        lib_path: str,
    ) -> None:
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
        elif is_windows():
            import ctypes

            kernel32 = ctypes.CDLL("kernel32", use_last_error=True)

            f_dlclose = kernel32.FreeLibrary
        else:
            raise NotImplementedError("Unsupported env, failed to do dlclose!")

        if f_dlclose is not None:
            if is_linux():
                f_dlclose.argtypes = [c_void_p]
                f_dlclose(self.DLL._handle)
            elif is_windows():
                import ctypes
                from ctypes import wintypes

                f_dlclose.argtypes = [wintypes.HMODULE]
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

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


@lru_cache
def binary_error_path(output_path: str) -> str:
    """
    standard format for the error path
    """
    return output_path + ".error"


@clear_on_fresh_cache
class CUDACodeCache:
    """
    A cache for managing the compilation and loading of CUDA source code specifically for CUTLASS.
    This class handles writing source code to files, compiling them into shared objects, and caching
    the results to avoid redundant compilations. It also manages error handling and logging for the
    compilation process.
    """

    @dataclasses.dataclass
    class CacheEntry:
        input_path: str
        output_path: str
        error_json: str | None = None

    cache: dict[str, CacheEntry] = {}
    aot_kernels_o: list[str] = []
    _SOURCE_CODE_SUFFIX = "cu"

    @staticmethod
    def cache_clear() -> None:
        CUDACodeCache.cache.clear()
        CUDACodeCache.aot_kernels_o.clear()

    @staticmethod
    @lru_cache(maxsize=4)
    def get_kernel_binary_remote_cache(
        caching_enabled: bool, caching_available: bool
    ) -> Any | None:
        """
        Get or create the class instance of the CUTLASSKernelBinaryRemoteCache.

        Args:
            caching_enabled: Whether binary remote caching is enabled
            caching_available: Whether we're in fbcode environment

        Returns:
            CUTLASSKernelBinaryRemoteCache: The class instance of the kernel binary remote cache
        """
        if not caching_enabled:
            log.debug("CUTLASSKernelBinaryRemoteCache not requested, skipping")
            return None
        if not caching_available:
            return None

        try:
            from torch._inductor.fb.kernel_binary_remote_cache import (
                CUTLASSKernelBinaryRemoteCache,
            )

            return CUTLASSKernelBinaryRemoteCache()
        except ImportError:
            log.debug(
                "CUTLASSKernelBinaryRemoteCache not available, remote caching disabled"
            )
            return None

    @classmethod
    @lru_cache(None)
    def write(cls, source_code: str, dst_file_ext: str) -> tuple[str, str]:
        """
        Writes source code into a file with dst_file_ext as the file extension.
        Returns the hash key of source code, and the path to the file.
        """

        if config.cuda.cutlass_hash_with_compile_cmd:
            cuda_command = repr(
                cuda_compile_command(["dummy_input"], "dummy_output", dst_file_ext)
            )
            extra = cuda_command
        else:
            extra = repr(
                [
                    # nvcc and cuda hash
                    _cuda_compiler(),
                    # cutlass flags and gcc hash
                    _nvcc_compiler_options(),
                    # flags
                    _nvcc_host_compiler_options(),
                    # cutlass key
                    cutlass_key(),
                    # hack to deal with AOTI .o compilation
                ]
            )
        key, input_path = write(source_code, cls._SOURCE_CODE_SUFFIX, extra=extra)
        return key, input_path

    @classmethod
    def compile(
        cls, source_code: str, dst_file_ext: str, extra_args: list[str] | None = None
    ) -> tuple[str, str, str]:
        """
        Compiles CUDA source_code into a file with dst_file_ext extension.
        If dst_file_ext is "so", first compiles to ".o" and then links to ".so".
        Returns a tuple of dst_file_path, hash_key, source_code_path
        """
        if dst_file_ext == "so":
            # Two-step compilation: first compile to .o, then link to .so
            obj_path, _, _ = cls.compile(source_code, "o", extra_args)
            key, input_path = cls.write(source_code, dst_file_ext)
            src_files, operation_name = [obj_path], "Linking"
        else:
            # Regular compilation for non-.so files
            key, input_path = cls.write(source_code, dst_file_ext)
            src_files, operation_name = [input_path], "Compilation"

        key_with_ext = key + dst_file_ext
        if key_with_ext not in cls.cache:
            from torch.utils._filelock import FileLock

            lock_dir = get_lock_dir()
            lock = FileLock(os.path.join(lock_dir, key + ".lock"), timeout=LOCK_TIMEOUT)
            with lock:
                output_path = input_path[: -len(cls._SOURCE_CODE_SUFFIX)] + dst_file_ext
                error_path = binary_error_path(output_path)
                binary_remote_cache = cls.get_kernel_binary_remote_cache(
                    caching_enabled=config.cuda.use_binary_remote_cache
                    and not config.force_disable_caches,
                    caching_available=config.is_fbcode(),
                )
                if binary_remote_cache is not None:
                    # The remote cache implementation will only download if the file does
                    # not already exist locally
                    binary_remote_cache.get(output_path, error_path)

                if os.path.exists(error_path):
                    with open(error_path, encoding="utf-8") as fh:
                        error_json = fh.read()
                    cmd_parts, error_output = json.loads(error_json)
                    if (
                        binary_remote_cache is not None
                        and config.cuda.upload_to_binary_remote_cache
                    ):
                        # This ensures that a local error is uploaded to the remote cache,
                        # as we make no assumptions about the remote cache having the same
                        # information as the local cache
                        binary_remote_cache.put(
                            error_path, config.cuda.binary_remote_cache_force_write
                        )
                    cls.cache[key_with_ext] = CUDACodeCache.CacheEntry(
                        input_path, output_path, error_json
                    )
                    raise exc.CUDACompileError(cmd_parts, error_output)
                if not os.path.exists(output_path):
                    cmd = cuda_compile_command(
                        src_files, output_path, dst_file_ext, extra_args
                    )
                    with open(input_path, "a") as f:
                        f.write("\n")
                        f.write(f"// CUDA {operation_name} cmd\n// {cmd}\n")
                    start_time = time()
                    log.debug("CUDA %s: %s", operation_name, cmd)
                    cmd_parts = cmd.split(" ")
                    try:
                        if use_re_build():
                            from triton.fb.re_build_helper import run_build_command

                            run_build_command(
                                cmd_parts,
                                os.path.dirname(input_path),
                                os.path.basename(output_path),
                            )
                        else:
                            subprocess.check_output(
                                cmd_parts, stderr=subprocess.STDOUT, env=os.environ
                            )
                    except subprocess.CalledProcessError as error:
                        cls._record_cuda_compile_error(
                            error.output.decode("utf-8"),
                            key_with_ext,
                            cmd_parts,
                            input_path,
                            output_path,
                            binary_remote_cache,
                        )
                        raise exc.CUDACompileError(cmd_parts, error.output) from error
                    except Exception as error:
                        if "COMPILE FAILED WITH" in str(error):
                            cls._record_cuda_compile_error(
                                str(error),
                                key_with_ext,
                                cmd_parts,
                                input_path,
                                output_path,
                                binary_remote_cache,
                            )
                            raise exc.CUDACompileError(cmd_parts, str(error)) from error
                        raise error
                    end_time = time()
                    log_duration_msg = f"CUDA {operation_name} took {end_time - start_time} seconds. Command: {cmd}"
                    log.info(log_duration_msg)

                else:
                    log.debug(
                        "CUDA %s skipped: %s since output already exists",
                        operation_name,
                        output_path,
                    )
                # Upload to remote cache if enabled
                if (
                    binary_remote_cache is not None
                    and config.cuda.upload_to_binary_remote_cache
                ):
                    # will log on errors, but not fail out
                    binary_remote_cache.put(
                        output_path, config.cuda.binary_remote_cache_force_write
                    )
                cls.cache[key_with_ext] = CUDACodeCache.CacheEntry(
                    input_path, output_path, None
                )

        cache_entry: CUDACodeCache.CacheEntry = cls.cache[key_with_ext]
        if cache_entry.error_json is not None:
            # Restore cached Exception and raise it as if we had compiled
            cmd_parts, error_output = json.loads(cache_entry.error_json)
            raise exc.CUDACompileError(cmd_parts, error_output.encode("utf-8"))
        return (cls.cache[key_with_ext].output_path, key, input_path)

    @classmethod
    def load(cls, source_code: str, dst_file_ext: str) -> tuple[DLLWrapper, str, str]:
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

    @classmethod
    def _record_cuda_compile_error(
        cls,
        error_str: str,
        key_with_ext: str,
        cmd_parts: list[str],
        input_path: str,
        output_path: str,
        # Any here, as the import and type will only work in fbcode
        # TODO: Make the typing hint strong here
        binary_remote_cache: Any = None,
    ) -> None:
        error_json = json.dumps([cmd_parts, error_str])
        cls.cache[key_with_ext] = CUDACodeCache.CacheEntry(
            input_path, output_path, error_json
        )
        error_path = binary_error_path(output_path)
        with open(error_path, "w", encoding="utf-8") as fh:
            fh.write(error_json)

        # Upload to remote cache directly from memory if enabled
        if (
            binary_remote_cache is not None
            and config.cuda.upload_to_binary_remote_cache
        ):
            binary_remote_cache.put(
                error_path, config.cuda.binary_remote_cache_force_write
            )


@clear_on_fresh_cache
class ROCmCodeCache:
    @dataclasses.dataclass
    class CacheEntry:
        input_path: str
        output_path: str

    cache: dict[str, CacheEntry] = {}
    aot_kernels_o: list[str] = []
    _SOURCE_CODE_SUFFIX = "cpp"
    _logged_compiler_version = False

    @staticmethod
    def cache_clear() -> None:
        ROCmCodeCache.cache.clear()
        ROCmCodeCache.aot_kernels_o.clear()

    @classmethod
    def write(cls, source_code: str, dst_file_ext: str) -> tuple[str, str]:
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
        cls, source_code: str, dst_file_ext: str, extra_args: list[str] | None = None
    ) -> tuple[str, str, str]:
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
            from torch.utils._filelock import FileLock

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
                        "Skip compiling %s: output %s already exists",
                        input_path,
                        output_path,
                    )
                cls.cache[key] = ROCmCodeCache.CacheEntry(input_path, output_path)

        return (cls.cache[key].output_path, key, input_path)

    @classmethod
    def load(cls, source_code: str, dst_file_ext: str) -> tuple[DLLWrapper, str, str]:
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
    def result(self) -> Callable[..., Any]:
        raise NotImplementedError


class LambdaFuture(CodeCacheFuture):
    def __init__(
        self, result_fn: Callable[..., Any], future: Future[Any] | None = None
    ) -> None:
        self.result_fn = result_fn
        self.future = future

    def result(self) -> Callable[..., Any]:
        return self.result_fn()


class StaticAutotunerFuture(CodeCacheFuture):
    """
    A statically launchable CachingAutotuner, loaded from TritonBundler
    """

    def __init__(self, static_autotuner: CachingAutotuner) -> None:
        # Pickled version of CachingAutotuner
        self.static_autotuner = static_autotuner
        # This needs to be set in AsyncCompile.triton, in case
        # we need to reload the CachingAutotuner from its source code
        # We don't store the source code on the CachingAutotuner itself
        # since it can be very large.
        self.reload_kernel_from_src: Callable[[], Any] | None = None

    def result(self) -> CachingAutotuner:
        assert self.reload_kernel_from_src is not None
        with dynamo_timed("StaticAutotunerFuture.warm_precompile"):
            self.static_autotuner.recheck_autotune_cache(
                reload_kernel_from_src=self.reload_kernel_from_src
            )
            self.static_autotuner.precompile(  # type: ignore[union-attr]
                warm_cache_only=False,
                reload_kernel=self.reload_kernel_from_src,
                static_triton_bundle_key=None,  # no need to save again
            )
            return self.static_autotuner
