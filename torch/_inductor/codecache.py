import base64
import enum
import functools
import getpass
import hashlib
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
from typing import Any, Dict

import torch
from torch.utils import cpp_extension

from . import config, cuda_properties, exc

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
        "TORCHINDUCTOR_CACHE_DIR", f"/tmp/torchinductor_{getpass.getuser()}"
    )


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


def write(source_code, ext, extra=""):
    basename = code_hash(source_code + extra)
    subdir = os.path.join(cache_dir(), basename[1:3])
    if not os.path.exists(subdir):
        os.makedirs(subdir, exist_ok=True)
    path = os.path.join(subdir, f"{basename}.{ext}")
    if not os.path.exists(path):
        # use a temp file for thread safety
        fd, tmp_path = tempfile.mkstemp(dir=subdir)
        with os.fdopen(fd, "w") as f:
            f.write(source_code)
        os.rename(tmp_path, path)
    return basename, path


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


class SupportedVecIsa(enum.Enum):
    AVX512 = 1
    AVX2 = 2
    INVALID = -1

    def __bool__(self):
        return self != SupportedVecIsa.INVALID

    @staticmethod
    def isa_vec_bit_size(supported_isa: enum.Enum):
        if supported_isa == SupportedVecIsa.AVX512:
            return 512
        elif supported_isa == SupportedVecIsa.AVX2:
            return 256
        else:
            return 0

    @staticmethod
    def isa_vec_size(supported_isa: enum.Enum, dtype: torch.dtype = torch.float):
        if supported_isa == SupportedVecIsa.AVX512:
            if dtype == torch.float:
                return 16
            elif dtype == torch.bfloat16:
                return 32
            else:
                raise NotImplementedError(
                    f"Vectorization has not supported {dtype} yet"
                )
        elif supported_isa == SupportedVecIsa.AVX2:
            if dtype == torch.float:
                return 8
            elif dtype == torch.bfloat16:
                return 16
            else:
                raise NotImplementedError(
                    f"Vectorization has not supported {dtype} yet"
                )
        else:
            raise NotImplementedError(
                f"Vectorization has not supported {supported_isa} yet"
            )

    @staticmethod
    def vec_size(dtype: torch.dtype = torch.float):
        return SupportedVecIsa.isa_vec_size(supported_vector_isa(), dtype)

    @staticmethod
    def str(supported_isa: enum.Enum):
        if supported_isa == SupportedVecIsa.AVX512:
            return "avx512"
        elif supported_isa == SupportedVecIsa.AVX2:
            return "avx2"
        else:
            return ""

    @staticmethod
    def vec_macro(supported_isa: enum.Enum):
        if supported_isa == SupportedVecIsa.AVX512:
            return "CPU_CAPABILITY_AVX512"
        elif supported_isa == SupportedVecIsa.AVX2:
            return "CPU_CAPABILITY_AVX2"
        else:
            return ""

    @staticmethod
    def build_arch_flags(supported_isa: enum.Enum):
        if supported_isa == SupportedVecIsa.AVX512:
            return "-mavx512f -mavx512dq -mavx512vl -mavx512bw -mfma"
        elif supported_isa == SupportedVecIsa.AVX2:
            return "-mavx2 -mfma"
        else:
            return ""


# Cache the cpuinfo to avoid I/O overhead. Meanwhile, the cpuinfo content
# might have too much redundant content that is useless for ISA check. Hence,
# we only cache some key isa information.
@functools.lru_cache(1)
def get_cpu_proc_info():
    if sys.platform != "linux":
        return []

    AVX512_CODE = """
#include <immintrin.h>

int main()
{
    __m512i a = _mm512_set_epi8(0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0);
    __m512i b = a;
    __mmask64 equality_mask = _mm512_cmp_epi8_mask(a, b, _MM_CMPINT_EQ);
    return 0;
}
    """

    AVX2_CODE = """
#include <immintrin.h>

int main()
{
    __m256i a = {0};
    a = _mm256_abs_epi16(a);
    __m256i x;
    _mm256_extract_epi64(x, 0); // we rely on this in our AVX2 code
    return 0;
}
    """

    def is_legal_isa(isa: SupportedVecIsa):
        key, input_path = write(
            AVX512_CODE if isa == SupportedVecIsa.AVX512 else AVX2_CODE, "cpp", extra=""
        )
        from filelock import FileLock

        lock_dir = get_lock_dir()
        lock = FileLock(os.path.join(lock_dir, key + ".lock"), timeout=LOCK_TIMEOUT)
        with lock:
            output_path = input_path[:-3] + "isa_chk"
            if os.path.exists(output_path):
                os.remove(output_path)

            build_cmd = f"{cpp_compiler()} {input_path} {SupportedVecIsa.build_arch_flags(isa)} -o{output_path}".split(
                " "
            )
            try:
                subprocess.check_output(build_cmd, stderr=subprocess.STDOUT)
                subprocess.check_call(output_path)
            except Exception:
                return False

            return True

    isa_info = []
    with open("/proc/cpuinfo") as _cpu_info:
        _cpu_info_content = _cpu_info.read()
        for cur_isa in [SupportedVecIsa.AVX512, SupportedVecIsa.AVX2]:
            if SupportedVecIsa.str(cur_isa) in _cpu_info_content and is_legal_isa(
                cur_isa
            ):
                isa_info.append(cur_isa)
        return isa_info


def supported_vector_isa():
    # TODO: Add ARM Vec here.
    # Dict(k: isa, v: number of float element)
    vec_isa_info = [SupportedVecIsa.AVX512, SupportedVecIsa.AVX2]

    cpu_info_content = get_cpu_proc_info()

    # If the simdlen is None, it indicates determin the vectroization length automatically
    if config.cpp.simdlen is None:
        for isa in vec_isa_info:
            if isa in cpu_info_content:
                return isa
        return SupportedVecIsa.INVALID

    # If the simdlen is less that, it indicates to disable the vectorization.
    if config.cpp.simdlen <= 1:
        return SupportedVecIsa.INVALID

    for isa in vec_isa_info:
        if (
            isa in cpu_info_content
            and config.cpp.simdlen == SupportedVecIsa.isa_vec_bit_size(isa)
        ):
            return isa

    return SupportedVecIsa.INVALID


def cpp_compile_command(input, output, include_pytorch=False):
    valid_isa = supported_vector_isa()
    if include_pytorch or valid_isa:
        ipaths = cpp_extension.include_paths() + [sysconfig.get_path("include")]
        lpaths = cpp_extension.library_paths() + [sysconfig.get_config_var("LIBDIR")]
        libs = ["c10", "torch", "torch_cpu", "torch_python", "gomp"]
        macros = SupportedVecIsa.vec_macro(valid_isa)
        if macros:
            macros = f"-D{macros}"
    else:
        # Note - this is effectively a header only inclusion. Usage of some header files may result in
        # symbol not found, if those header files require a library.
        # For those cases, include the lpath and libs command as we do for pytorch above.
        # This approach allows us to only pay for what we use.
        ipaths = cpp_extension.include_paths() + [sysconfig.get_path("include")]
        lpaths = []
        libs = ["gomp"]
        macros = ""
    ipaths = " ".join(["-I" + p for p in ipaths])
    lpaths = " ".join(["-L" + p for p in lpaths])
    libs = " ".join(["-l" + p for p in libs])

    return re.sub(
        r"[ \n]+",
        " ",
        f"""
            {cpp_compiler()} {input} -shared -fPIC -Wall -std=c++14 -Wno-unused-variable
            {ipaths} {lpaths} {libs} {macros}
            -march=native -O3 -ffast-math -fno-finite-math-only -fopenmp
            -o{output}
        """,
    ).strip()


class CppCodeCache:
    cache = dict()
    clear = staticmethod(cache.clear)

    @classmethod
    def load(cls, source_code):
        key, input_path = write(source_code, "cpp", extra=cpp_compile_command("i", "o"))
        if key not in cls.cache:
            from filelock import FileLock

            lock_dir = get_lock_dir()
            lock = FileLock(os.path.join(lock_dir, key + ".lock"), timeout=LOCK_TIMEOUT)
            with lock:
                output_path = input_path[:-3] + "so"
                if not os.path.exists(output_path):
                    cmd = cpp_compile_command(
                        input=input_path, output=output_path
                    ).split(" ")
                    try:
                        subprocess.check_output(cmd, stderr=subprocess.STDOUT)
                    except subprocess.CalledProcessError as e:
                        raise exc.CppCompileError(cmd, e.output)

                cls.cache[key] = cdll.LoadLibrary(output_path)
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


@functools.lru_cache(None)
def patch_triton_dir():
    os.environ["TRITON_CACHE_DIR"] = os.environ.get(
        "TRITON_CACHE_DIR", os.path.join(cache_dir(), "triton")
    )


class TritonCodeCache:
    @staticmethod
    def get_name(mod):
        (name,) = [n for n in dir(mod) if n.startswith("kernel")]
        return name

    @classmethod
    def load(cls, source_code):
        patch_triton_dir()
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


class TritonFuture:
    def __init__(self, source_code, future):
        self.source_code = source_code
        self.future = future

    def result(self):
        if hasattr(self, "kernel"):
            return self.kernel
        # If the worker failed this will throw an exception.
        self.future.result()
        kernel = self.kernel = _load_kernel(self.source_code)
        del self.source_code, self.future
        return kernel


class AsyncCompile:
    def __init__(self):
        self._context_keepalive = None

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
            for i in range(config.compile_threads):
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
        if self._context_keepalive is None:
            # Workaround `CUDA: Error- context is destroyed`
            self._context_keepalive = torch.tensor([1], device="cuda")

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
        if config.compile_threads > 1:
            for key, result in list(scope.items()):
                if isinstance(result, (Future, TritonFuture)):
                    scope[key] = result.result()

        _compile_end()


AsyncCompile.warm_pool()
