# mypy: allow-untyped-defs
import logging
import os
import shutil
from pathlib import Path
from typing import Optional

import torch
from torch._inductor import config
from torch._inductor.codegen.cuda import cuda_env
from torch._inductor.cpp_builder import _set_gpu_runtime_env, _transform_cuda_paths
from torch._inductor.utils import is_linux


if config.is_fbcode():
    from triton.fb.build import build_paths


log = logging.getLogger(__name__)
autotuning_log = torch._logging.getArtifactLogger(__name__, "autotuning")


def use_re_build() -> bool:
    """
    Use for CUTLASS compilation only right now.
    """
    if config.is_fbcode() and not cuda_env.nvcc_exist(_cuda_compiler()):
        from triton.fb.re_build_helper import should_build_locally

        return not should_build_locally()
    return False


def _cutlass_path() -> str:
    if config.is_fbcode():
        from libfb.py import parutil

        return parutil.get_dir_path("cutlass-4-headers")
    else:
        return config.cutlass.cutlass_dir


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


def _cuda_compiler() -> Optional[str]:
    if cuda_env.nvcc_exist(config.cuda.cuda_cxx):
        return config.cuda.cuda_cxx
    if config.is_fbcode():
        return os.path.join(build_paths.sdk_home, "bin", "nvcc")
    if cuda_env.nvcc_exist(os.getenv("CUDACXX")):
        return os.getenv("CUDACXX", "")
    if cuda_env.nvcc_exist(os.getenv("CUDA_HOME")):
        return os.path.realpath(os.path.join(os.getenv("CUDA_HOME", ""), "bin/nvcc"))
    return "nvcc"


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
        config.cutlass.compile_opt_level,
        "-std=c++17",
        "--expt-relaxed-constexpr",
        "-DNDEBUG",
    ]
    if config.is_fbcode():
        options.extend(["-ccbin", os.path.dirname(build_paths.gcc)])
    if config.cutlass.enable_debug_info:
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
    if config.cutlass.use_fast_math:
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


class CUDACompileSourceCapturingContext:
    # Helper class for Benchmarking and Testing CUTLASS Kernels in isolation.
    # Can be used to capture the sourcecode passed to CUDACodeCache.compile

    def __init__(self):
        self.sources = []
        self._compile_patch = None

    def __enter__(self, *args, **kwargs):
        import unittest.mock as mock

        import torch._inductor.codecache

        _compile_method_orig = torch._inductor.codecache.CUDACodeCache.compile

        def my_compile(
            source_code, dst_file_ext, extra_args: Optional[list[str]] = None
        ):
            self.sources.append(source_code)
            return _compile_method_orig(source_code, dst_file_ext)

        # pyrefly: ignore [bad-assignment]
        self._compile_patch = mock.patch(
            "torch._inductor.codecache.CUDACodeCache.compile", my_compile
        )
        self._compile_patch.__enter__(*args, **kwargs)  # type: ignore[union-attr]
        return self

    def __exit__(self, *args, **kwargs):
        self._compile_patch.__exit__(*args, **kwargs)  # type: ignore[union-attr]


def cuda_standalone_runner_compile_command(srcpath: Path, exepath: Path):
    # returns command string to compile a (captured) CUDA GEMM Kernel source to a standalone executable that's ready to run
    # Passes the correct preprocessor define to nvcc to ensure the standalone runner is enabled.

    extra_args = ["-DGENERATE_STANDALONE_RUNNER=1", "-DCUTLASS_DEBUG_TRACE_LEVEL=1"]
    compile_command = cuda_compile_command(
        [str(srcpath)], str(exepath), "exe", extra_args=extra_args
    )
    return compile_command
