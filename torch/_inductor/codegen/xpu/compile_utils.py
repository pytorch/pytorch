# mypy: allow-untyped-defs
import logging
from typing import Optional

from torch._inductor import config
from torch._inductor.utils import is_linux

from ..cuda.compile_utils import _cutlass_include_paths
from .xpu_env import get_xpu_arch


log = logging.getLogger(__name__)


def _sycl_compiler() -> Optional[str]:
    return "icpx"


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
    else:
        raise NotImplementedError(
            "Unsupported env, failed to find xpu libs! Currently only Linux is supported."
        )
    return extra_ldflags


def _sycl_host_compiler_options() -> list[str]:
    return [
        "-fPIC",
    ]


def _sycl_arch_as_compile_option() -> str:
    arc_option_map = {"pvc": "intel_gpu_pvc", "bmg": "intel_gpu_bmg"}
    arch = get_xpu_arch()
    return arc_option_map.get(arch, "intel_gpu_pvc")


def _sycl_compiler_options() -> list[str]:
    options = [
        "-DCUTLASS_ENABLE_SYCL",
        "-DCUTLASS_SYCL_PROFILING_ENABLED",
        "-DSYCLCOMPAT_PROFILING_ENABLED",
        "-DSYCL_INTEL_TARGET",
        "-gline-tables-only",
        "-DCUTLASS_VERSIONS_GENERATED",
        "-O3",
        "-DNDEBUG",
        "-std=c++17",
        "-fPIE",
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
    sycl_host_compiler_options = _sycl_host_compiler_options()
    sycl_compiler_options = _sycl_compiler_options()
    options = (
        ["-I" + path for path in include_paths]
        + ["-isystem /include"]
        + sycl_compiler_options
        + extra_args
        + sycl_host_compiler_options
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
