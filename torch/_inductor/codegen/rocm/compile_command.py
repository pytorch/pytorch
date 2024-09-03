# mypy: allow-untyped-defs
import logging
import os
from typing import List, Optional

from torch._inductor import config
from torch._inductor.utils import is_linux


log = logging.getLogger(__name__)


def _rocm_include_paths() -> List[str]:
    from torch.utils import cpp_extension

    rocm_include = (
        os.path.join(config.rocm.rocm_home, "include")
        if config.rocm.rocm_home
        else cpp_extension._join_rocm_home("include")
    )
    if not config.rocm.ck_dir:
        log.warning("Unspecified Composable Kernel include dir")
    ck_include = os.path.join(
        config.rocm.ck_dir or cpp_extension._join_rocm_home("composable_kernel"),
        "include",
    )
    return [os.path.realpath(rocm_include), os.path.realpath(ck_include)]


def _rocm_lib_options() -> List[str]:
    from torch.utils import cpp_extension

    rocm_lib_dir = (
        os.path.join(config.rocm.rocm_home, "lib")
        if config.rocm.rocm_home
        else cpp_extension._join_rocm_home("lib")
    )
    hip_lib_dir = (
        os.path.join(config.rocm.rocm_home, "hip", "lib")
        if config.rocm.rocm_home
        else cpp_extension._join_rocm_home("hip", "lib")
    )

    return [
        f"-L{os.path.realpath(rocm_lib_dir)}",
        f"-L{os.path.realpath(hip_lib_dir)}",
        "-lamdhip64",
    ]


def _rocm_compiler_options() -> List[str]:
    arch_list = config.rocm.arch or ["native"]
    gpu_arch_flags = [f"--offload-arch={arch}" for arch in arch_list]
    opts = [
        config.rocm.compile_opt_level,
        "-x",
        "hip",
        "-std=c++17",
        *gpu_arch_flags,
        "-fno-gpu-rdc",
        "-fPIC",
        "-mllvm",
        "-amdgpu-early-inline-all=true",
        "-mllvm",
        "-amdgpu-function-calls=false",
        "-mllvm",
        "-enable-post-misched=0",
    ]
    if config.rocm.is_debug:
        opts += ["-DDEBUG_LOG=1", "-g"]
    if config.rocm.save_temps:
        opts += ["--save-temps=obj"]
    if config.rocm.print_kernel_resource_usage:
        opts += ["-Rpass-analysis=kernel-resource-usage"]
    if config.rocm.flush_denormals:
        opts += ["-fgpu-flush-denormals-to-zero"]
    if config.rocm.use_fast_math:
        opts += ["-ffast-math"]
    return opts


def rocm_compiler() -> Optional[str]:
    if is_linux():
        if config.rocm.rocm_home:
            return os.path.realpath(
                os.path.join(config.rocm.rocm_home, "llvm", "bin", "clang")
            )
        try:
            from torch.utils import cpp_extension

            return os.path.realpath(
                cpp_extension._join_rocm_home("llvm", "bin", "clang")
            )
        except OSError:
            # neither config.rocm.rocm_home nor env variable ROCM_HOME are set
            return "clang"
    return None


def rocm_compile_command(
    src_files: List[str],
    dst_file: str,
    dst_file_ext: str,
    extra_args: Optional[List[str]] = None,
) -> str:
    include_paths = _rocm_include_paths()
    lib_options = _rocm_lib_options()
    compiler_options = _rocm_compiler_options()
    compiler = rocm_compiler()
    options = (
        compiler_options
        + (extra_args if extra_args else [])
        + ["-I" + path for path in include_paths]
        + lib_options
    )
    src_file = " ".join(src_files)
    res = ""
    if dst_file_ext == "o":
        res = f"{compiler} {' '.join(options)} -c -o {dst_file} {src_file}"
    elif dst_file_ext == "so":
        options.append("-shared")
        res = f"{compiler} {' '.join(options)} -o {dst_file} {src_file}"
    elif dst_file_ext == "exe":
        res = f"{compiler} {' '.join(options)} -o {dst_file} {src_file}"
    else:
        raise NotImplementedError(f"Unsupported output file suffix {dst_file_ext}!")
    return res
