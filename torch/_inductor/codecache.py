import base64
import dataclasses
import functools
import getpass
import hashlib
import json
import logging
import multiprocessing
import os
import re
import shutil
import signal
import subprocess
import sys
import sysconfig
import tempfile
import types
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from ctypes import cdll
from threading import Thread
from time import sleep, time
from typing import Any, Callable, Dict, List

import torch

from torch.hub import _Faketqdm, tqdm
from torch.utils import cpp_extension
from . import config, cuda_properties, exc
from .utils import developer_warning

LOCK_TIMEOUT = 600

# timing metrics for time spent in the compilation
_cumulative_compile_time = 0
_t0 = None


def _compile_start():
    global _t0
    if _t0 is None:
        _t0 = time()


def _compile_end():
    global _cumulative_compile_time, _t0
    if _t0 is not None:
        t1 = time()
        _cumulative_compile_time += t1 - _t0
        _t0 = None
        # print("CUMULATIVE COMPILE TIME", _cumulative_compile_time)


log = logging.getLogger(__name__)
logging.getLogger("filelock").setLevel(logging.DEBUG if config.debug else logging.INFO)


@functools.lru_cache(None)
def cache_dir():
    return os.environ.get(
        "TORCHINDUCTOR_CACHE_DIR",
        f"{tempfile.gettempdir()}/torchinductor_{getpass.getuser()}",
    )


class DiskCache:
    @staticmethod
    @functools.lru_cache(None)
    def _subdir():
        subdir = os.path.join(cache_dir(), "cached_tunings")
        os.makedirs(subdir, exist_ok=True)
        return subdir

    @staticmethod
    @functools.lru_cache(4096)
    def _read_file(path):
        with open(path, "r") as fd:
            return json.loads(fd.read())

    def __init__(self, unique_name):
        super().__init__()
        self.unique_name = unique_name

    def lookup(self, key: Any, generate: Callable[[], Any]):
        """
        Check if we have already generated key, if not call generate()
        to populate the cache.
        """
        path = os.path.join(self._subdir(), code_hash(self.unique_name + repr(key)))
        if not os.path.exists(path):
            value = generate()
            write_atomic(path, json.dumps(value))
        return self._read_file(path)


def get_lock_dir():
    lock_dir = os.path.join(cache_dir(), "locks")
    if not os.path.exists(lock_dir):
        os.makedirs(lock_dir, exist_ok=True)
    return lock_dir


def code_hash(code):
    return (
        "c"
        + base64.b32encode(hashlib.sha256(code.encode("utf-8")).digest())[:51]
        .decode("utf-8")
        .lower()
    )


def get_code_path(source_code, ext, extra):
    basename = code_hash(source_code + extra)
    subdir = os.path.join(cache_dir(), basename[1:3])
    path = os.path.join(subdir, f"{basename}.{ext}")
    return basename, subdir, path


def write(source_code, ext, extra=""):
    basename, subdir, path = get_code_path(source_code, ext, extra)
    if not os.path.exists(subdir):
        os.makedirs(subdir, exist_ok=True)
    if not os.path.exists(path):
        write_atomic(path, source_code)
    return basename, path


def write_atomic(path: str, source_code: str):
    # use a temp file for thread safety
    fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path))
    with os.fdopen(fd, "w") as f:
        f.write(source_code)
    os.rename(tmp_path, path)


def cpp_compiler():
    if isinstance(config.cpp.cxx, (list, tuple)):
        search = tuple(config.cpp.cxx)
    else:
        search = (config.cpp.cxx,)
    return cpp_compiler_search(search)


@functools.lru_cache(1)
def cpp_compiler_search(search):
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


def install_gcc_via_conda():
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


def is_gcc():
    return re.search(r"(gcc|g\+\+)", cpp_compiler())


class VecISA:
    _bit_width: int
    _macro: str
    _arch_flags: str
    _dtype_nelements: Dict[torch.dtype, int]

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

    def bit_width(self):
        return self._bit_width

    def nelements(self, dtype: torch.dtype = torch.float):
        return self._dtype_nelements[dtype]

    def build_macro(self):
        return self._macro

    def build_arch_flags(self):
        return self._arch_flags

    def __hash__(self) -> int:
        return hash(str(self))

    @functools.lru_cache(None)
    def __bool__(self):
        key, input_path = write(VecISA._avx_code, "cpp", extra="")
        from filelock import FileLock

        lock_dir = get_lock_dir()
        lock = FileLock(os.path.join(lock_dir, key + ".lock"), timeout=LOCK_TIMEOUT)
        with lock:
            output_path = input_path[:-3] + "so"
            build_cmd = cpp_compile_command(
                input_path, output_path, warning_all=False, vec_isa=self
            ).split(" ")
            try:
                # Check build result
                subprocess.check_output(build_cmd, stderr=subprocess.STDOUT)
                subprocess.check_call(
                    [
                        "python",
                        "-c",
                        VecISA._avx_py_load.replace("__lib_path__", output_path),
                    ],
                    stderr=subprocess.DEVNULL,
                )
            except Exception as e:
                return False

            return True


@dataclasses.dataclass
class VecAVX512(VecISA):
    _bit_width = 512
    _macro = "CPU_CAPABILITY_AVX512"
    _arch_flags = "-mavx512f -mavx512dq -mavx512vl -mavx512bw -mfma"
    _dtype_nelements = {torch.float: 16, torch.bfloat16: 32}

    def __str__(self) -> str:
        return "avx512"

    __hash__: Callable[[VecISA], Any] = VecISA.__hash__


@dataclasses.dataclass
class VecAVX2(VecISA):
    _bit_width = 256
    _macro = "CPU_CAPABILITY_AVX2"
    _arch_flags = "-mavx2 -mfma"
    _dtype_nelements = {torch.float: 8, torch.bfloat16: 16}

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

    def __bool__(self):
        return False

    __hash__: Callable[[VecISA], Any] = VecISA.__hash__


invalid_vec_isa = InvalidVecISA()
supported_vec_isa_list = [VecAVX512(), VecAVX2()]


# Cache the cpuinfo to avoid I/O overhead. Meanwhile, the cpuinfo content
# might have too much redundant content that is useless for ISA check. Hence,
# we only cache some key isa information.
@functools.lru_cache(None)
def valid_vec_isa_list():
    if sys.platform != "linux":
        return []

    isa_list = []
    with open("/proc/cpuinfo") as _cpu_info:
        _cpu_info_content = _cpu_info.read()
        for isa in supported_vec_isa_list:
            if str(isa) in _cpu_info_content and isa:
                isa_list.append(isa)
        return isa_list


def pick_vec_isa():
    _valid_vec_isa_list: List[VecISA] = valid_vec_isa_list()
    if not _valid_vec_isa_list:
        return invalid_vec_isa

    # If the simdlen is None, it indicates determin the vectroization length automatically
    if config.cpp.simdlen is None:
        assert _valid_vec_isa_list
        return _valid_vec_isa_list[0]

    for isa in _valid_vec_isa_list:
        if config.cpp.simdlen == isa.bit_width():
            return isa

    return invalid_vec_isa


def get_shared(shared=True):
    return "-shared -fPIC" if shared else ""


def get_warning_all_flag(warning_all=True):
    return "-Wall" if warning_all else ""


def cpp_flags():
    return "-std=c++17 -Wno-unused-variable"


def optimization_flags():
    base_flags = "-O3 -ffast-math -fno-finite-math-only"
    if sys.platform == "darwin":
        # Per https://mac.r-project.org/openmp/ right way to pass `openmp` flags to MacOS is via `-Xclang`
        # Also, `-march=native` is unrecognized option on M1
        base_flags += " -Xclang -fopenmp"
    else:
        base_flags += " -march=native -fopenmp"
    return base_flags


def use_custom_generated_macros():
    return "-D C10_USING_CUSTOM_GENERATED_MACROS"


def get_include_and_linking_paths(
    include_pytorch=False, vec_isa: VecISA = invalid_vec_isa
):
    if sys.platform == "linux" and (
        include_pytorch
        or vec_isa != invalid_vec_isa
        or config.cpp.enable_kernel_profile
    ):
        # Note - We include pytorch only on linux right now. There is more work
        # to do to enable OMP build on darwin where PyTorch is built with IOMP
        # and we need a way to link to what PyTorch links.
        ipaths = cpp_extension.include_paths() + [sysconfig.get_path("include")]
        lpaths = cpp_extension.library_paths() + [sysconfig.get_config_var("LIBDIR")]
        libs = ["c10", "torch", "torch_cpu", "torch_python", "gomp"]
        macros = vec_isa.build_macro()
        if macros:
            macros = f"-D{macros}"
    else:
        # Note - this is effectively a header only inclusion. Usage of some header files may result in
        # symbol not found, if those header files require a library.
        # For those cases, include the lpath and libs command as we do for pytorch above.
        # This approach allows us to only pay for what we use.
        ipaths = cpp_extension.include_paths() + [sysconfig.get_path("include")]
        lpaths = []
        macros = ""
        if sys.platform == "darwin":
            # GNU OpenMP generally is not available on MacOS
            # There is either Intel OpenMP(for x86) or LLVM OpenMP (for both x86 and arm64)
            libs = ["omp"]
            if os.getenv("CONDA_PREFIX") is not None:
                # On MacOS OpenMP is not available via the system install
                # But on conda can be provided using https://anaconda.org/anaconda/llvm-openmp
                conda_lib_path = os.path.join(os.getenv("CONDA_PREFIX"), "lib")
                ipaths.append(os.path.join(os.getenv("CONDA_PREFIX"), "include"))
                lpaths.append(conda_lib_path)
                # Prefer Intel OpenMP on x86 machine
                if os.uname().machine == "x86_64" and os.path.exists(
                    os.path.join(conda_lib_path, "libiomp5.dylib")
                ):
                    libs = ["iomp5"]
        else:
            libs = ["gomp"]
    ipaths = " ".join(["-I" + p for p in ipaths])
    lpaths = " ".join(["-L" + p for p in lpaths])
    libs = " ".join(["-l" + p for p in libs])
    return ipaths, lpaths, libs, macros


def cpp_compile_command(
    input,
    output,
    warning_all=True,
    shared=True,
    include_pytorch=False,
    vec_isa: VecISA = invalid_vec_isa,
):
    ipaths, lpaths, libs, macros = get_include_and_linking_paths(
        include_pytorch, vec_isa
    )

    return re.sub(
        r"[ \n]+",
        " ",
        f"""
            {cpp_compiler()} {input} {get_shared(shared)} {get_warning_all_flag(warning_all)} {cpp_flags()}
            {ipaths} {lpaths} {libs} {macros}
            {optimization_flags()}
            {use_custom_generated_macros()}
            -o{output}
        """,
    ).strip()


class CppCodeCache:
    cache = dict()
    clear = staticmethod(cache.clear)

    @staticmethod
    def _load_library(path):
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
    def load(cls, source_code):
        picked_vec_isa = pick_vec_isa()
        key, input_path = write(
            source_code,
            "cpp",
            extra=cpp_compile_command("i", "o", vec_isa=picked_vec_isa),
        )
        if key not in cls.cache:
            from filelock import FileLock

            lock_dir = get_lock_dir()
            lock = FileLock(os.path.join(lock_dir, key + ".lock"), timeout=LOCK_TIMEOUT)
            with lock:
                output_path = input_path[:-3] + "so"
                if not os.path.exists(output_path):
                    cmd = cpp_compile_command(
                        input=input_path, output=output_path, vec_isa=picked_vec_isa
                    ).split(" ")
                    try:
                        subprocess.check_output(cmd, stderr=subprocess.STDOUT)
                    except subprocess.CalledProcessError as e:
                        raise exc.CppCompileError(cmd, e.output) from e

                cls.cache[key] = cls._load_library(output_path)
                cls.cache[key].key = key

        return cls.cache[key]


class PyCodeCache:
    cache = dict()
    clear = staticmethod(cache.clear)

    @classmethod
    def load(cls, source_code):
        key, path = write(source_code, "py")
        if key not in cls.cache:
            with open(path) as f:
                code = compile(f.read(), path, "exec")
                mod = types.ModuleType(f"{__name__}.{key}")
                mod.__file__ = path
                mod.key = key
                exec(code, mod.__dict__, mod.__dict__)
                # another thread might set this first
                cls.cache.setdefault(key, mod)
        return cls.cache[key]


class TritonCodeCache:
    @staticmethod
    def get_name(mod):
        (name,) = [n for n in dir(mod) if n.startswith("triton_")]
        return name

    @classmethod
    def load(cls, source_code):
        mod = PyCodeCache.load(source_code)
        return getattr(mod, cls.get_name(mod))


def _worker_compile(source_code, cc, device):
    cuda_properties.set_compiler_worker_current_device(device)
    kernel = TritonCodeCache.load(source_code)
    kernel.precompile(warm_cache_only_with_cc=cc)


def _load_kernel(source_code):
    kernel = TritonCodeCache.load(source_code)
    kernel.precompile()
    return kernel


def _load_kernel_name(source_code):
    return TritonCodeCache.get_name(PyCodeCache.load(source_code))


class TritonFuture:
    def __init__(self, source_code, future):
        self.source_code = source_code
        self.future = future

    # @dynamo_utils.dynamo_timed
    def result(self):
        t0 = time()
        if hasattr(self, "kernel"):
            return self.kernel
        # If the worker failed this will throw an exception.
        self.future.result()
        kernel = self.kernel = _load_kernel(self.source_code)
        latency = time() - t0
        if latency > 50:
            name = _load_kernel_name(self.source_code)
            developer_warning(
                f"Detected long compilation time of {latency} seconds for kernel name {name}"
            )
            developer_warning(self.source_code)
        del self.source_code, self.future
        return kernel


class AsyncCompile:
    def __init__(self):
        pass

    @staticmethod
    @functools.lru_cache(1)
    def pool():
        assert config.compile_threads > 1
        return ThreadPoolExecutor(config.compile_threads)

    @staticmethod
    @functools.lru_cache(1)
    def process_pool():
        # ensure properties have been calculated before processes
        # are forked
        cuda_properties._properties()
        assert config.compile_threads > 1
        orig_ppid = os.getpid()

        # if this process dies abnormally (e.g. segfault)
        # it will not shut down the workers. Instead
        # the workers will have their parent reassigned to the
        # init process. This launches a separate thread to
        # watch for the worker getting reassigned,
        # and cleans it up in this case.
        def init():
            def run():
                while True:
                    sleep(1)
                    if orig_ppid != os.getppid():
                        os.kill(os.getpid(), signal.SIGKILL)

            global _watchdog_thread
            _watchdog_thread = Thread(target=run, daemon=True)
            _watchdog_thread.start()

        # we rely on 'fork' because we cannot control whether users
        # have an `if __name__ == '__main__'` in their main process.
        fork_context = multiprocessing.get_context("fork")
        pool = ProcessPoolExecutor(
            config.compile_threads, mp_context=fork_context, initializer=init
        )
        # when this pool is created in a subprocess object, the normal exit handler
        # doesn't run, and we need to register our own handler.
        # exitpriority has to be high, because another one of the finalizers will
        # kill the worker thread that sends the shutdown message to the workers...
        multiprocessing.util.Finalize(None, pool.shutdown, exitpriority=sys.maxsize)
        return pool

    @classmethod
    def warm_pool(cls):
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
            pool._start_executor_manager_thread()
        _compile_end()

    @classmethod
    def submit(cls, task):
        if config.compile_threads <= 1:
            return task()
        return cls.pool().submit(task)

    @classmethod
    def map(cls, fn, seq):
        if config.compile_threads <= 1 or len(seq) <= 1:
            return list(map(fn, seq))
        return [t.result() for t in [cls.pool().submit(fn, x) for x in seq]]

    def triton(self, source_code):
        _compile_start()

        if config.compile_threads > 1:
            major, minor = torch.cuda.get_device_capability()
            device = torch.cuda.current_device()
            cc = major * 10 + minor
            future = self.process_pool().submit(
                _worker_compile, source_code, cc, device
            )
            return TritonFuture(source_code, future)
        else:
            return _load_kernel(source_code)

    def cpp(self, source_code):
        def task():
            return CppCodeCache.load(source_code).kernel

        return self.submit(task)

    def wait(self, scope: Dict[str, Any]):
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
