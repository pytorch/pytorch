import functools
import getpass
import importlib
import logging
import os
import sys
import tempfile
from filelock import FileLock
from torch.utils.cpp_extension import load_inline
from torch._inductor import codecache
import cpu_tla



log = logging.getLogger(__name__)
tla_build_cache = {}

 

@functools.lru_cache(None)
def cache_dir_base():
    cache_dir = f"{tempfile.gettempdir()}/cpu_tla_{getpass.getuser()}"
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir

 

def cache_dir(name):
    python_version = f"py{sys.version_info.major}{sys.version_info.minor}"
    build_folder = f"{python_version}"
    base_dir = os.path.join(cache_dir_base(), build_folder)
    dir = os.path.join(base_dir, name)
    os.makedirs(dir, exist_ok=True)
    return dir

 

def load_tla_extension(source_file, functions=[], debug=False, enable_fp16=False, enable_bf16=False, enable_vnni=False, use_mkl=False, symbol=False):
    def get_optimization_flags():
        base_flags = "-O0 -g" if debug else "-O3"
        base_flags += " -g" if symbol else ""
        base_flags += " -ffast-math -fno-finite-math-only"
        base_flags += " -march=native -fopenmp"
        return base_flags

    def get_mkl_compiler_options():
        mkl_cflags = "-DTLA_USE_MKL -DMKL_ILP64 -m64 -I${MKLROOT}/include"
        mkl_ldflags = "-L${MKLROOT}/lib/intel64 -lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl"
        mkl_include_paths = "-I${MKLROOT}/include"
        return mkl_cflags, mkl_ldflags, mkl_include_paths

    def get_compiler_options():
        shared = codecache.get_shared()
        warning_all_flag = codecache.get_warning_all_flag()
        cpp_flags = codecache.cpp_flags()
        if enable_fp16:
            cpp_flags += " -mavx512fp16"
        if enable_bf16:
            cpp_flags += " -mavx512bf16"
        if enable_vnni:
            cpp_flags += " -mavx512vnni"
        ipaths, lpaths, libs, macros = codecache.get_include_and_linking_paths(
            vec_isa=codecache.pick_vec_isa(),

        )
        optimization_flags = get_optimization_flags()
        use_custom_generated_macros = codecache.use_custom_generated_macros()
        warning_ignores = "-Wno-ignored-attributes -Wno-narrowing -Wno-sign-compare -Wno-format"
        extra_cflags = f"{cpp_flags} {optimization_flags} {warning_all_flag} {macros} {use_custom_generated_macros} {warning_ignores}"
        extra_ldflags = f"{shared} {lpaths} {libs}"
        tla_base_path = os.path.dirname(cpu_tla.__file__)
        extra_include_paths = f"{ipaths} -I{tla_base_path}/../include -I{tla_base_path}/../third_party/libxsmm/include"
        if use_mkl:
            mkl_cflags, mkl_ldflags, mkl_include_paths = get_mkl_compiler_options()
            extra_cflags += f" {mkl_cflags}"
            extra_ldflags += f" {mkl_ldflags}"
            extra_include_paths += f" {mkl_include_paths}"
        return extra_cflags, extra_ldflags, extra_include_paths

    extra_cflags, extra_ldflags, extra_include_paths = get_compiler_options()
    functions = functions if functions else ["kernel"]
    # load source_file into str and get cache_id
    with open(source_file, "r") as f:
        source_str = f.read()
        cache_id = codecache.code_hash(source_str + str(sorted(functions)) + extra_cflags + extra_ldflags + extra_include_paths)
    file_name = os.path.basename(source_file)
    basename = os.path.splitext(file_name)[0]
    name = f"{basename}_{cache_id}"
    build_dir = f"{cache_dir(name)}"
    if cache_id not in tla_build_cache:
        lock_dir = codecache.get_lock_dir()
        lock = FileLock(os.path.join(lock_dir, cache_id + ".lock"), timeout=codecache.LOCK_TIMEOUT)
        with lock:
            filepath = os.path.join(build_dir, f"{name}.so")
            if not os.path.exists(filepath):
                mod = load_inline(
                    name=f"{name}",
                    build_directory=f"{build_dir}",
                    cpp_sources=[f"#include \"{os.path.abspath(source_file)}\""],
                    functions=functions if functions else ["kernel"],
                    extra_cflags=[extra_cflags],
                    extra_ldflags=[extra_ldflags],
                    extra_include_paths=[extra_include_paths],
                )
                log.info(f"Built and load TLA extension: {filepath}")
            else:
                spec = importlib.util.spec_from_file_location(name, filepath)
                assert spec is not None
                mod = importlib.util.module_from_spec(spec)
                assert isinstance(spec.loader, importlib.abc.Loader)
                spec.loader.exec_module(mod)
                log.info(f"Loaded TLA extension: {filepath}")
            tla_build_cache[cache_id] = mod
    return tla_build_cache[cache_id]


kmod = load_tla_extension("/home/haozhe/torchdynamo/quantize.cpp", ["quantize_per_tensor"], enable_vnni=True)
import torch
x = torch.rand(2, 64 + 17, dtype=torch.bfloat16)


def func(x, scale, zp):
    tq = x / scale + zp
    x = torch.clamp(torch.round(tq), 0, 255)
    x = x.to(torch.uint8)
    return x
ref = func(x, torch.tensor([0.05], dtype=torch.float32), torch.tensor([1], dtype=torch.int32))
y = kmod.quantize_per_tensor(x, torch.tensor([0.05]), torch.tensor([1]))
print(y - ref)
print(y)
print(ref)
