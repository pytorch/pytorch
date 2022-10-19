import base64
import functools
import getpass
import hashlib
import logging
import os
import re
import shutil
import subprocess
import sysconfig
import tempfile
import types
from concurrent.futures import Future, ThreadPoolExecutor
from ctypes import cdll
from typing import Any, Dict

import torch
from torch.utils import cpp_extension

from . import config, exc

LOCK_TIMEOUT = 600

log = logging.getLogger(__name__)
logging.getLogger("filelock").setLevel(logging.DEBUG if config.debug else logging.INFO)


def cache_dir():
    return f"/tmp/torchinductor_{getpass.getuser()}"


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


def cpp_compile_command(input, output, include_pytorch=False):
    if include_pytorch or config.cpp.simdlen:
        ipaths = cpp_extension.include_paths() + [sysconfig.get_path("include")]
        lpaths = cpp_extension.library_paths() + [sysconfig.get_config_var("LIBDIR")]
        libs = ["c10", "torch", "torch_cpu", "torch_python", "gomp"]
        macros = ""
        if config.cpp.simdlen == 16:
            macros = " " + "-DCPU_CAPABILITY_AVX512"
        elif config.cpp.simdlen == 8:
            macros = " " + "-DCPU_CAPABILITY_AVX2"
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
            {cpp_compiler()} -shared -fPIC -Wall -std=c++14 -Wno-unused-variable
            {ipaths} {lpaths} {libs} {macros}
            -march=native -O3 -ffast-math -fno-finite-math-only -fopenmp
            -o{output} {input}
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


class AsyncCompile:
    def __init__(self):
        self._context_keepalive = None

    @staticmethod
    @functools.lru_cache(1)
    def pool():
        assert config.compile_threads > 1
        return ThreadPoolExecutor(config.compile_threads)

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
        if self._context_keepalive is None:
            # Workaround `CUDA: Error- context is destroyed`
            self._context_keepalive = torch.tensor([1], device="cuda")
        kernel = TritonCodeCache.load(source_code)

        def task():
            kernel.precompile()
            return kernel

        return self.submit(task)

    def cpp(self, source_code):
        def task():
            return CppCodeCache.load(source_code).kernel

        return self.submit(task)

    def wait(self, scope: Dict[str, Any]):
        if config.compile_threads > 1:
            for key, result in list(scope.items()):
                if isinstance(result, Future):
                    scope[key] = result.result()
