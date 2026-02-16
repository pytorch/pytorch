# mypy: allow-untyped-defs
import logging
import os
import shutil
from typing import Optional

from torch._inductor import config
from torch._inductor.codegen.xpu.xpu_env import get_xpu_arch
from torch._inductor.utils import is_linux

from ..cuda.compile_utils import _cutlass_include_paths


log = logging.getLogger(__name__)


def _sycl_compiler() -> Optional[str]:
    # Search order:
    # 0) which icpx
    # 1) config.xpu.oneapi_root
    # 2) ONEAPI_ROOT environment variable
    # 3) default system search PATH.
    if shutil.which("icpx"):
        return "icpx"

    if os.path.exists(config.xpu.oneapi_root or ""):
        oneapi_root = config.xpu.oneapi_root
    elif os.path.exists(os.getenv("ONEAPI_ROOT") or ""):
        oneapi_root = os.getenv("ONEAPI_ROOT")
    else:
        oneapi_root = None

    if oneapi_root:
        oneapi_inclue = os.path.join(oneapi_root, "include")
        if "CPLUS_INCLUDE_PATH" in os.environ:
            os.environ["CPLUS_INCLUDE_PATH"] += ":" + oneapi_inclue
        else:
            os.environ["CPLUS_INCLUDE_PATH"] = oneapi_inclue
        return os.path.realpath(os.path.join(oneapi_root, "bin/icpx"))
    else:
        raise RuntimeError("Can not find Intel compiler.")


def _sycl_lib_options() -> list[str]:
    """
    Util function for CUTLASS backend to find the correct XPU libraries.
    """
    # _set_gpu_runtime_env()  # cpp_extension consults the env
    from torch.utils import cpp_extension

    lpaths = cpp_extension.library_paths(device_type="xpu")
    extra_ldflags: list[str] = []
    if is_linux():
        for path in lpaths:
            if "torch/lib" in path:
                # don't want to depend on pytorch
                continue
            # -rpath ensures the DLL can find its dependencies when loaded, even
            # if the library path is non-standard.
            extra_ldflags.extend([f"-L{path}", "-Xlinker", f"-rpath={path}"])

        extra_ldflags.append("-lsycl")
    else:
        raise NotImplementedError(
            "Unsupported env, failed to find xpu libs! Currently only Linux is supported."
        )
    return extra_ldflags


def _sycl_arch_as_compile_option() -> str:
    arc_option_map = {"Xe12": "intel_gpu_pvc", "Xe20": "intel_gpu_bmg_g21"}
    arch = get_xpu_arch()
    return arc_option_map.get(arch, "intel_gpu_pvc")


def _sycl_compiler_options() -> list[str]:
    options = [
        "-DCUTLASS_ENABLE_SYCL",
        "-DSYCL_INTEL_TARGET",
        "-DCUTLASS_VERSIONS_GENERATED",
        "-O3",
        "-DNDEBUG",
        "-std=c++17",
        "-fPIC",
        "-fsycl",
        f"-fsycl-targets={_sycl_arch_as_compile_option()}",
        "-Xspirv-translator",
        "-spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate",
        "-fno-sycl-instrument-device-code",
        "-DMKL_ILP64",
        "-MD",
        "-MT",
    ]
    if config.cutlass.enable_debug_info:
        options.extend(["-lineinfo", "-g", "-DCUTLASS_DEBUG_TRACE_LEVEL=1"])
    return options


def xpu_compile_command(
    src_files: list[str],
    dst_file: str,
    dst_file_ext: str,
    extra_args: Optional[list[str]] = None,
) -> str:
    if extra_args is None:
        extra_args = []
    include_paths = _cutlass_include_paths()
    sycl_lib_options = _sycl_lib_options()
    sycl_compiler_options = _sycl_compiler_options()
    options = (
        extra_args
        + ["-I" + path for path in include_paths]
        + ["-isystem /include"]
        + sycl_compiler_options
        + sycl_lib_options
    )
    src_file = " ".join(src_files)
    res = ""
    if dst_file_ext == "o":
        res = f"{_sycl_compiler()} {' '.join(options)} -c -o {dst_file} {src_file}"
    elif dst_file_ext == "so":
        options.append("-shared")
        res = f"{_sycl_compiler()} {' '.join(options)} -o {dst_file} {src_file}"
    elif dst_file_ext == "exe":
        res = f"{_sycl_compiler()} {' '.join(options)} -o {dst_file} {src_file}"
    else:
        raise NotImplementedError(f"Unsupported output file suffix {dst_file_ext}!")
    log.debug("XPU command: %s", res)
    return res
