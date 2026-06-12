# mypy: allow-untyped-defs
import functools
import logging
import os
import shutil
import subprocess

from torch._inductor import config
from torch._inductor.codegen.xpu.xpu_env import get_xpu_arch
from torch._inductor.utils import is_linux

from ..cuda.compile_utils import _cutlass_include_paths


log = logging.getLogger(__name__)


@functools.cache
def icpx_exist() -> bool:
    """Return True if the SYCL compiler (icpx) is available."""
    try:
        _sycl_compiler()
        return True
    except RuntimeError:
        return False


def _sycl_compiler() -> str:
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


def _sycl_arch_as_compile_option() -> str | None:
    """Return the -fsycl-targets value for the current XPU device.

    Returns None when the specific device variant cannot be reliably mapped to
    an AOT target. In that case the caller should omit -fsycl-targets entirely
    and let SYCL JIT-compile the kernel for the running device.
    """
    arch_option_map = {"Xe12": "intel_gpu_pvc"}
    arch = get_xpu_arch()
    return arch_option_map.get(arch)


def _sycl_compiler_options() -> list[str]:
    options = [
        "-DCUTLASS_ENABLE_SYCL",
        "-DSYCL_INTEL_TARGET",
        "-DCUTLASS_VERSIONS_GENERATED",
        "-O3",
        "-DNDEBUG",
        "-std=c++20",
        "-fPIC",
        "-fsycl",
    ]
    sycl_target = _sycl_arch_as_compile_option()
    if sycl_target is not None:
        options.append(f"-fsycl-targets={sycl_target}")
    options += [
        "-Xspirv-translator",
        "-spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate",
        "-fno-sycl-instrument-device-code",
        "-DMKL_ILP64",
        "-MD",
        "-Xs",
        (
            "-options \"-igc_opts 'VISAOptions=-perfmodel,VectorAliasBBThreshold=100000000000,"
            "ExtraOCLOptions=-cl-intel-256-GRF-per-thread'\" "
            "-options -ze-opt-large-register-file"
        ),
    ]
    if config.cutlass.enable_debug_info:
        options.extend(["-lineinfo", "-g", "-DCUTLASS_DEBUG_TRACE_LEVEL=1"])
    return options


def xpu_compile_command(
    src_files: list[str],
    dst_file: str,
    dst_file_ext: str,
    extra_args: list[str] | None = None,
) -> str:
    if extra_args is None:
        extra_args = []
    include_paths = _cutlass_include_paths()
    sycl_lib_options = _sycl_lib_options()
    sycl_compiler_options = _sycl_compiler_options()

    # Build command as a list to preserve arguments with spaces
    cmd_parts = (
        [_sycl_compiler()]
        + extra_args
        + ["-I" + path for path in include_paths]
        + ["-isystem", "/include"]
        + sycl_compiler_options
        + sycl_lib_options
    )
    if dst_file_ext == "o":
        cmd_parts.extend(["-c", "-o", dst_file] + src_files)
    elif dst_file_ext == "so":
        cmd_parts.extend(["-shared", "-o", dst_file] + src_files)
    elif dst_file_ext == "exe":
        cmd_parts.extend(["-o", dst_file] + src_files)
    else:
        raise NotImplementedError(f"Unsupported output file suffix {dst_file_ext}!")

    res = subprocess.list2cmdline(cmd_parts)
    log.debug("XPU command: %s", res)
    return res
