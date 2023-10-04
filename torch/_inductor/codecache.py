from __future__ import annotations

import base64
import dataclasses
import functools
import getpass
import hashlib
import importlib
import json
import logging
import multiprocessing
import os
import pathlib
import platform
import re
import shlex
import shutil
import signal
import subprocess
import sys
import sysconfig
import tempfile
import threading
import warnings
import weakref
from bisect import bisect_right
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from ctypes import c_void_p, cdll, CDLL
from dataclasses import field
from functools import partial
from importlib import abc
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
from torch._inductor import config, exc
from torch._inductor.codegen.cuda import cuda_env
from torch._inductor.utils import developer_warning, is_linux

if TYPE_CHECKING:
    from torch._inductor.graph import GraphLowering
    from torch._inductor.select_algorithm import ChoiceCaller

from torch.hub import _Faketqdm, tqdm

_HERE = os.path.abspath(__file__)
_TORCH_PATH = os.path.dirname(os.path.dirname(_HERE))

if config.is_fbcode():
    from triton.fb import build_paths
    from triton.fb.build import _run_build_command

    from torch._inductor.fb.utils import (  # type: ignore[import]
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


LOCK_TIMEOUT = 600

# timing metrics for time spent in the compilation
_cumulative_compile_time = 0.0
_t0 = None


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


@functools.lru_cache(None)
def cache_dir() -> str:
    cache_dir = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    if cache_dir is None:
        cache_dir = f"{tempfile.gettempdir()}/torchinductor_{getpass.getuser()}"
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


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


class CacheBase:
    @staticmethod
    @functools.lru_cache(None)
    def get_system() -> Dict[str, Any]:
        try:
            import triton

            triton_version = triton.__version__
        except ModuleNotFoundError:
            triton_version = None

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
            "other": {
                "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            },
        }

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
    def lookup(self, *keys: Tuple[str]) -> Optional[Dict[str, Any]]:
        cache = self.get_local_cache()

        sub_cache = cache
        for key in keys:
            if key in cache:
                sub_cache = cache[key]  # type: ignore[index]
            else:
                return None

        return sub_cache

    def set_value(self, *keys: List[str], value: Any) -> None:
        cache = self.get_local_cache()

        sub_cache: Dict = cache  # type: ignore[type-arg]
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
        name: str,
        inputs: str,
        benchmark: Callable[[Any], Dict[ChoiceCaller, float]],
    ) -> Dict[ChoiceCaller, float]:
        """
        Check to see if we have benchmarked the given choice callers. For each
        choice caller:

            1. Check global_cache[name][inputs][choice], return benchmark if cached.
            2. Check local_cache[name][inputs][choice], return benchmark if cached.
            3.
                a. `max_autotune_gemm=True`: benchmark the choice, update
                    local_cache[name][inputs][choice], and return the benchmark.
                b. `max_autotune_gemm=False`: don't benchmark the choice, return nothing.
        """

        log_stats = partial(log_global_cache_stats, self.system, name, inputs)
        log_vals = partial(log_global_cache_vals, self.system, name, inputs)
        log_errors = partial(log_global_cache_errors, self.system, name, inputs)
        timings = {}

        def check_cache(cache, callback=None) -> bool:
            """Check if `cache` contains data for all the choices"""
            hit = True
            for choice in choices:
                choice_hash = choice.hash_key()
                if choice_hash in cache.get(name, {}).get(inputs, {}):
                    # cache hit
                    timings[choice] = cache[name][inputs][choice_hash]
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

                    local_cache.setdefault(name, {})
                    local_cache[name].setdefault(inputs, {})
                    for choice, timing in timings.items():
                        local_cache[name][inputs][choice.hash_key()] = timing
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


def code_hash(code: Union[str, bytes], extra: str = ""):
    hashing_str = code if isinstance(code, bytes) else code.encode("utf-8")
    if extra != "":
        hashing_str = hashing_str + b"||" + extra.encode("utf-8")
    return (
        "c"
        + base64.b32encode(hashlib.sha256(hashing_str).digest())[:51]
        .decode("utf-8")
        .lower()
    )


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
    if hash_type == "cubin":
        return code_hash(repr(content))
    raise AssertionError(f"Unknown hash type {hash_type}")


def write(
    content: Union[str, bytes],
    extension: str,
    extra: str = "",
    hash_type: str = "code",
    specified_dir: str = "",
) -> Tuple[str, str]:
    key: str = get_hash(content, extra, hash_type)
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
class CompiledFxGraph:
    """Class holding a compiled FX graph"""

    compiled_artifact: Optional[Callable[..., Any]] = None
    current_callable: Optional[Callable[..., Any]] = None
    cache_key: Optional[str] = None
    artifact_path: Optional[str] = None
    cache_linemap: Optional[List[Tuple[int, str]]] = None
    device_types: Set[str] = field(default_factory=set)
    device_idxs: Set[int] = field(default_factory=set)
    mutated_inputs: Set[str] = field(default_factory=set)
    mutated_input_idxs: Set[int] = field(default_factory=set)

    _boxed_call: Optional[bool] = None

    def __call__(self, inputs: List[Any]) -> Any:
        return self.get_current_callable()(inputs)

    def get_current_callable(self) -> Callable[..., Any]:
        if self.current_callable is None:
            # This prevents a circular reference that makes CompiledFxGraph
            # get stuck without getting garbage collected
            return functools.partial(_run_from_cache, weakref.proxy(self))
        else:
            return self.current_callable


def _run_from_cache(compiled_graph: CompiledFxGraph, inputs: List[Any]) -> Any:
    # We can't really serialize callables that may be C++/Triton/etc.,
    # so we serialize their disk cache location instead
    # TODO: When making an API that can save compiled models e2e to disk
    # this will need to be better
    if compiled_graph.compiled_artifact is None:
        from .codecache import PyCodeCache

        assert compiled_graph.cache_key
        assert compiled_graph.artifact_path
        compiled_graph.compiled_artifact = PyCodeCache.load_by_key_path(
            compiled_graph.cache_key,
            compiled_graph.artifact_path,
            compiled_graph.cache_linemap,
        ).call

    return compiled_graph.compiled_artifact(inputs)


def cpp_compiler() -> str:
    if config.is_fbcode():
        return build_paths.gcc()
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
    return bool(re.search(r"(gcc|g\+\+)", cpp_compiler()))


@functools.lru_cache(None)
def is_apple_clang() -> bool:
    cxx = cpp_compiler()
    version_string = subprocess.check_output([cxx, "--version"]).decode("utf8")
    return "Apple" in version_string.splitlines()[0]


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
#if defined(CPU_CAPABILITY_AVX512) || defined(CPU_CAPABILITY_AVX2)
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#endif

__attribute__((aligned(64))) float in_out_ptr0[16] = {0.0};

extern "C" void __avx_chk_kernel() {
    auto tmp0 = at::vec::Vectorized<float>(1);
    auto tmp1 = tmp0.exp();
    tmp1.store(in_out_ptr0);
}
"""

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
class VecAVX512(VecISA):
    _bit_width = 512
    _macro = "CPU_CAPABILITY_AVX512"
    _arch_flags = "-mavx512f -mavx512dq -mavx512vl -mavx512bw -mfma"
    _dtype_nelements = {torch.float: 16, torch.bfloat16: 32, torch.float16: 32}

    def __str__(self) -> str:
        return "avx512"

    __hash__: Callable[[VecISA], Any] = VecISA.__hash__


@dataclasses.dataclass
class VecAVX2(VecISA):
    _bit_width = 256
    _macro = "CPU_CAPABILITY_AVX2"
    _arch_flags = "-mavx2 -mfma"
    _dtype_nelements = {torch.float: 8, torch.bfloat16: 16, torch.float16: 16}

    def __str__(self) -> str:
        return "avx2"

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
    if sys.platform != "linux":
        return []

    isa_list = []
    with open("/proc/cpuinfo") as _cpu_info:
        _cpu_info_content = _cpu_info.read()
        for isa in supported_vec_isa_list:
            if str(isa) in _cpu_info_content and isa:
                isa_list.append(isa)
        return isa_list


def pick_vec_isa() -> VecISA:
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


def get_shared(shared: bool = True) -> str:
    return "-shared -fPIC" if shared else ""


def get_warning_all_flag(warning_all: bool = True) -> str:
    return "-Wall" if warning_all else ""


def get_glibcxx_abi_build_flags() -> str:
    return "-D_GLIBCXX_USE_CXX11_ABI=" + str(int(torch._C._GLIBCXX_USE_CXX11_ABI))


def cpp_flags() -> str:
    return "-std=c++17 -Wno-unused-variable -Wno-unknown-pragmas"


def cpp_wrapper_flags() -> str:
    return "-DTORCH_INDUCTOR_CPP_WRAPPER"


def optimization_flags() -> str:
    base_flags = "-O3 -ffast-math -fno-finite-math-only"
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
) -> Tuple[str, str, str, str]:
    if (
        config.is_fbcode()
        and "CUDA_HOME" not in os.environ
        and "CUDA_PATH" not in os.environ
    ):
        os.environ["CUDA_HOME"] = os.path.dirname(build_paths.cuda())
    from torch.utils import cpp_extension

    macros = ""
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
            libs += ["c10", "torch", "torch_cpu"]
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
        macros = vec_isa.build_macro()
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
            else:
                macros = f"-D{macros}"

        if aot_mode and cuda:
            if macros is None:
                macros = ""
            macros += " -D USE_CUDA"

        if cuda:
            if config.is_fbcode():
                libs += ["cuda"]
            else:
                libs += ["c10_cuda", "cuda", "torch_cuda"]
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
                header_path = os.path.join(os.getenv("OMP_PREFIX"), "include", "omp.h")
                valid_env = os.path.exists(header_path)
                if valid_env:
                    ipaths.append(os.path.join(os.getenv("OMP_PREFIX"), "include"))
                    lpaths.append(os.path.join(os.getenv("OMP_PREFIX"), "lib"))
                else:
                    warnings.warn("environment variable `OMP_PREFIX` is invalid.")
                omp_available = omp_available or valid_env

            libs = [] if omp_available else ["omp"]

            # prefer to use openmp from `conda install llvm-openmp`
            if not omp_available and os.getenv("CONDA_PREFIX") is not None:
                omp_available = is_conda_llvm_openmp_installed()
                if omp_available:
                    conda_lib_path = os.path.join(os.getenv("CONDA_PREFIX"), "lib")
                    ipaths.append(os.path.join(os.getenv("CONDA_PREFIX"), "include"))
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

    # third party libs
    if config.is_fbcode():
        ipaths.append(build_paths.sleef())
        ipaths.append(build_paths.openmp())
        ipaths.append(build_paths.gcc_include())
        ipaths.append(build_paths.libgcc())
        ipaths.append(build_paths.libgcc_arch())
        ipaths.append(build_paths.libgcc_backward())
        ipaths.append(build_paths.glibc())
        ipaths.append(build_paths.linux_kernel())
        ipaths.append(build_paths.gcc_install_tools_include())
        ipaths.append(build_paths.cuda())
        # We also need to bundle includes with absolute paths into a remote directory
        # (later on, we copy the include paths from cpp_extensions into our remote dir)
        ipaths.append("include")

    static_link_libs = []
    if aot_mode and cuda and config.is_fbcode():
        # For Meta internal cuda-12, it is recommended to static link cudart
        static_link_libs = ["-Wl,-Bstatic", "-lcudart_static", "-Wl,-Bdynamic"]

    ipaths_str = " ".join(["-I" + p for p in ipaths])
    lpaths_str = " ".join(["-L" + p for p in lpaths])
    libs_str = " ".join(static_link_libs + ["-l" + p for p in libs])
    return ipaths_str, lpaths_str, libs_str, macros


def cpp_compile_command(
    input: str,
    output: str,
    warning_all: bool = True,
    shared: bool = True,
    include_pytorch: bool = False,
    vec_isa: VecISA = invalid_vec_isa,
    cuda: bool = False,
    aot_mode: bool = False,
    compile_only: bool = False,
) -> str:
    ipaths, lpaths, libs, macros = get_include_and_linking_paths(
        include_pytorch, vec_isa, cuda, aot_mode
    )
    if config.is_fbcode():
        if aot_mode:
            inp_name = input
            out_name = output
        else:
            # We need to copy any absolute-path torch includes
            inp_name = os.path.basename(input)
            out_name = os.path.basename(output)
        linker_paths = [os.path.dirname(build_paths.ld()), build_paths.glibc_lib()]
        linker_paths = " ".join(["-B" + p for p in linker_paths])
    else:
        inp_name = input
        out_name = output
        linker_paths = ""  # let the compiler pick
    return re.sub(
        r"[ \n]+",
        " ",
        f"""
            {cpp_compiler()} {inp_name} {get_shared(shared)}
            {get_warning_all_flag(warning_all)} {cpp_flags()}
            {get_glibcxx_abi_build_flags()}
            {ipaths} {lpaths} {libs} {macros} {linker_paths}
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


class CudaKernelParamCache:
    cache: Dict[str, Dict[str, str]] = dict()
    clear = staticmethod(cache.clear)

    @classmethod
    def set(cls, key: str, params: Dict[str, str], cubin: str) -> None:
        _, path = write(
            cubin,
            "cubin",
            hash_type="cubin",
            specified_dir=config.aot_inductor.output_path,
        )
        params["cubin_path"] = path
        cls.cache[key] = params

    @classmethod
    def get(cls, key: str) -> Optional[Dict[str, str]]:
        return cls.cache.get(key, None)


class AotCodeCache:
    cache: Dict[str, str] = dict()
    clear = staticmethod(cache.clear)

    @classmethod
    def compile(
        cls,
        graph: GraphLowering,
        source_code: str,
        serialized_extern_kernel_nodes: Optional[str],
        cuda: bool,
    ) -> Callable[..., Any]:
        picked_vec_isa = pick_vec_isa()
        cpp_command = repr(
            cpp_compile_command(
                "i", "o", vec_isa=picked_vec_isa, cuda=cuda, aot_mode=graph.aot_mode
            )
        )
        if config.is_fbcode():
            ld_command = build_paths.ld()
            objcopy_command = build_paths.objcopy()
        else:
            ld_command = "ld"
            objcopy_command = "objcopy"
        key, input_path = write(
            source_code,
            "cpp",
            extra=cpp_command,
            specified_dir=config.aot_inductor.output_path,
        )

        def _to_bytes(t: torch.Tensor) -> bytes:
            # This serializes the tensor's untyped_storage to bytes by accessing
            # the raw data of the underlying structure.
            import ctypes

            t_cpu = t.untyped_storage().cpu()
            raw_array = ctypes.cast(
                t_cpu.data_ptr(), ctypes.POINTER(ctypes.c_ubyte * t_cpu.nbytes())
            )

            return bytes(raw_array.contents)

        aot_constants = b""
        for tensor in graph.constants.values():
            aot_constants += _to_bytes(tensor)

        consts_key, consts_path = write(
            aot_constants,
            "bin",
            specified_dir=config.aot_inductor.output_path,
        )

        if key not in cls.cache:
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

                output_so = os.path.splitext(input_path)[0] + ".so"

                if not os.path.exists(output_so):
                    output_o = os.path.splitext(input_path)[0] + ".o"
                    cmd = cpp_compile_command(
                        input=input_path,
                        output=output_o,
                        vec_isa=picked_vec_isa,
                        cuda=cuda,
                        aot_mode=graph.aot_mode,
                        compile_only=True,
                    )
                    log.debug("aot compilation command: %s", cmd)
                    run_command_and_check(cmd)

                    consts_o = os.path.splitext(consts_path)[0] + ".o"
                    cmd = f"{ld_command} -r -b binary -o {consts_o} {consts_path}"
                    run_command_and_check(cmd)
                    log.debug("aot constant binary command: %s", cmd)

                    cmd = (
                        f"{objcopy_command} --rename-section"
                        " .data=.lrodata,alloc,load,readonly,data,contents"
                        f" {consts_o} {consts_o}"
                    )
                    log.debug("aot constant obj command: %s", cmd)
                    run_command_and_check(cmd)

                    cmd = f"rm {consts_path}"
                    log.debug("aot constant bin removal command: %s", cmd)
                    run_command_and_check(cmd)

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
                    log.debug(
                        "aot constant binary redefine symbol: %s", " ".join(symbol_list)
                    )
                    for cmd in symbol_list:
                        run_command_and_check(cmd)

                    cmd = cpp_compile_command(
                        input=f"{output_o} {consts_o}",
                        output=output_so,
                        vec_isa=picked_vec_isa,
                        cuda=cuda,
                        aot_mode=graph.aot_mode,
                    )
                    log.debug("aot linkage command: %s", cmd)
                    run_command_and_check(cmd)
                else:
                    log.debug(
                        "aot_inductor dynamic library already exist: %s", output_so
                    )

                cls.cache[key] = output_so

        def wrapper_call(*args) -> Any:
            assert graph.graph_outputs is not None and len(graph.graph_outputs) > 0
            return cls.cache[key], *(None for i in range(len(graph.graph_outputs) - 1))

        return wrapper_call


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
def compile_file(input_path: str, output_path: str, cmd: List[str]) -> None:
    input_file = os.path.basename(input_path) if config.is_fbcode() else input_path
    try:
        if config.is_fbcode():
            # Need to copy our header into the same folder as the sourcecode.
            header_path = cpp_prefix_path()
            header_name = os.path.basename(header_path)
            output_name = os.path.basename(output_path)
            # When we build remotely, we need to make sure to carefully copy any files
            # that are required during the compilation process into our build directly.
            # This is where all of the ATen/c10/Torch includes come from.
            torch_includes_path = os.path.join(
                torch.utils.cpp_extension._TORCH_PATH, "include"
            )
            with tempfile.TemporaryDirectory() as tmp_dir:
                # Copy everything to tmp compilation folder
                shutil.copy(header_path, os.path.join(tmp_dir, header_name))
                shutil.copy(input_path, os.path.join(tmp_dir, input_file))
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


class CppCodeCache:
    cache: Dict[str, CDLL] = dict()
    clear = staticmethod(cache.clear)

    @staticmethod
    def _load_library(path: str) -> CDLL:
        try:
            return cdll.LoadLibrary(path)
        except OSError as e:
            if "gomp" in str(e) and os.path.exists("/usr/lib64/libgomp.so.1"):
                # hacky workaround for fbcode/buck
                global _libgomp
                _libgomp = cdll.LoadLibrary("/usr/lib64/libgomp.so.1")
                return cdll.LoadLibrary(path)
            if "failed to map segment from shared object" in str(e):
                raise OSError(
                    f"{e}.  The most common reason this may occur is if the {tempfile.gettempdir()} folder "
                    "is mounted with noexec (e.g., by default Docker mounts tmp file systems "
                    f"as noexec).  Please remount {tempfile.gettempdir()} with exec enabled, or set another "
                    "temporary directory with TORCHINDUCTOR_CACHE_DIR environment variable."
                ) from e
            raise

    @classmethod
    def load(cls, source_code: str) -> CDLL:
        picked_vec_isa = pick_vec_isa()
        cpp_command = repr(cpp_compile_command("i", "o", vec_isa=picked_vec_isa))
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
                            input=input_path, output=output_path, vec_isa=picked_vec_isa
                        )
                    )
                    compile_file(input_path, output_path, cmd)
                cls.cache[key] = cls._load_library(output_path)
                cls.cache[key].key = key  # type: ignore[attr-defined]

        return cls.cache[key]


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
    ) -> ModuleType:
        key, path = write(source_code, "py", extra=extra)
        return cls.load_by_key_path(key, path, linemap)

    @classmethod
    def load_by_key_path(
        cls,
        key: str,
        path: str,
        linemap: Optional[List[Tuple[int, str]]] = None,
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
                    )
                mod = ModuleType(f"{__name__}.{key}")
                mod.__file__ = path
                mod.key = key  # type: ignore[attr-defined]
                exec(code, mod.__dict__, mod.__dict__)
                sys.modules[mod.__name__] = mod
                # another thread might set this first
                cls.cache.setdefault(key, mod)
                # unzip into separate lines/nodes lists
                cls.linemaps[path] = list(zip(*linemap))

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


class CppWrapperCodeCache:
    cache: Dict[str, CDLL] = dict()
    clear = staticmethod(cache.clear)

    @classmethod
    def load(cls, source_code: str, func_name: str, key: str, cuda: bool) -> CDLL:
        name = f"inline_extension_{key}"
        cpp_wrapper_dir = cpp_wrapper_cache_dir(name)
        if not os.path.exists(cpp_wrapper_dir):
            os.makedirs(cpp_wrapper_dir)

        ext = "so"
        filepath = os.path.join(cpp_wrapper_dir, f"{name}.{ext}")
        log.debug("Cpp wrapper code path %s", filepath)

        if key not in cls.cache:
            log.debug("Cpp wrapper cache miss for %s", filepath)
            from filelock import FileLock

            lock_dir = get_lock_dir()
            lock = FileLock(os.path.join(lock_dir, key + ".lock"), timeout=LOCK_TIMEOUT)
            with lock:
                if not os.path.exists(filepath):
                    log.debug("Cpp wrapper building %s", filepath)

                    _cpp_flags = cpp_flags()
                    _opt_flags = optimization_flags()
                    _shared = get_shared()
                    _warning_all_flag = get_warning_all_flag()
                    _ipaths, _lpaths, _libs, _macros = get_include_and_linking_paths(
                        vec_isa=pick_vec_isa(),
                        cuda=cuda,
                    )
                    _use_custom_generated_macros = use_custom_generated_macros()
                    _cpp_wrapper_flags = cpp_wrapper_flags()

                    extra_cflags = f"{_cpp_flags} {_opt_flags} {_warning_all_flag} {_macros} {_cpp_wrapper_flags} \
                    {_use_custom_generated_macros}"
                    # For CPP wrapper, add -ffast-math during linking to make CPU flush denormals.
                    # CPP wrapper leverages cpp_extension which will do the compilation and linking in two stages.
                    # We need to explicitly add -ffast-math as a linking flag.
                    # For the default python wrapper, the compilation and linking are done in one command thus -ffast-math
                    # will take effect in both compilation and linking.
                    extra_ldflags = f"{_shared} {_lpaths} {_libs} -ffast-math"
                    extra_include_paths = f"{_ipaths}"

                    mod = torch.utils.cpp_extension.load_inline(
                        name=name,
                        build_directory=cpp_wrapper_dir,
                        cpp_sources=[source_code],
                        functions=[func_name],
                        extra_cflags=[extra_cflags],
                        extra_ldflags=[extra_ldflags],
                        extra_include_paths=[extra_include_paths],
                        use_pch=True,
                    )
                    log.debug("Cpp wrapper done building %s", filepath)
                else:
                    log.debug("Found target .so, cpp wrapper loading %s", filepath)
                    spec = importlib.util.spec_from_file_location(name, filepath)  # type: ignore[attr-defined]
                    assert spec is not None
                    mod = importlib.util.module_from_spec(spec)  # type: ignore[attr-defined]
                    assert isinstance(spec.loader, abc.Loader)
                    spec.loader.exec_module(mod)
                    log.debug("Cpp wrapper done loading %s", filepath)

                cls.cache[key] = mod

        return cls.cache[key]


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
        return os.path.join(os.getenv("CUDA_HOME", ""), "bin/nvcc")
    return "nvcc"


def _cutlass_include_paths() -> List[str]:
    cutlass_path = config.cuda.cutlass_dir
    return [
        os.path.join(cutlass_path, "include"),
        os.path.join(cutlass_path, "tools/library/include"),
        os.path.join(cutlass_path, "tools/library/src"),
        os.path.join(cutlass_path, "tools/util/include"),
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
) -> str:
    include_paths = _cutlass_include_paths()
    cuda_lib_options = _cuda_lib_options()
    nvcc_host_compiler_options = _nvcc_host_compiler_options()
    nvcc_compiler_options = _nvcc_compiler_options()
    options = (
        nvcc_compiler_options
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
    def compile(cls, source_code, dst_file_ext) -> Tuple[str, str, str]:
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
                        [input_path], output_path, dst_file_ext
                    ).split(" ")
                    try:
                        subprocess.check_output(
                            cmd, stderr=subprocess.STDOUT, env=os.environ
                        )
                    except subprocess.CalledProcessError as error:
                        raise exc.CUDACompileError(cmd, error.output) from error
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


def _worker_compile(
    kernel_name: str, source_code: str, cc: int, device: torch.device
) -> None:
    device_interface = get_interface_for_device(device.type)
    device_interface.Worker.set_device(device.index)
    kernel = TritonCodeCache.load(kernel_name, source_code)
    kernel.precompile(warm_cache_only_with_cc=cc)


def _load_kernel(kernel_name: str, source_code: str) -> ModuleType:
    kernel = TritonCodeCache.load(kernel_name, source_code)
    kernel.precompile()
    return kernel


class TritonFuture:
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
            return self.kernel  # type: ignore[has-type]
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


_watchdog_thread: Optional[Thread] = None


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

    @classmethod
    def map(cls, fn: Callable[..., Any], seq: List[Any]) -> List[Any]:
        if config.compile_threads <= 1 or len(seq) <= 1:
            return list(map(fn, seq))
        return [t.result() for t in [cls.pool().submit(fn, x) for x in seq]]

    def triton(
        self, kernel_name: str, source_code: str, device: str = "cuda"
    ) -> Union[TritonFuture, ModuleType]:
        _compile_start()

        if config.compile_threads > 1:
            device_interface = get_interface_for_device(device)
            device = torch.device(device, device_interface.current_device())
            cc = device_interface.get_compute_capability(device)
            future = self.process_pool().submit(
                _worker_compile, kernel_name, source_code, cc, device
            )
            return TritonFuture(kernel_name, source_code, future)
        else:
            return _load_kernel(kernel_name, source_code)

    def cpp(self, source_code: str) -> ModuleType:
        def task():
            return CppCodeCache.load(source_code).kernel

        return self.submit(task)

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


AsyncCompile.warm_pool()
