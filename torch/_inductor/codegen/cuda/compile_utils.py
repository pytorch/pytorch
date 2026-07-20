# mypy: allow-untyped-defs
import logging
import os
import re
import shutil

import torch
from torch._inductor import config
from torch._inductor.codegen.cuda import cuda_env
from torch._inductor.cpp_builder import _set_gpu_runtime_env, _transform_cuda_paths
from torch._inductor.utils import is_linux
from torch.utils._ordered_set import OrderedSet


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


def _cutlass_path() -> str | None:
    if config.is_fbcode():
        from libfb.py import parutil

        return parutil.get_dir_path("cutlass-4-headers")
    else:
        from torch._inductor.codegen.cutlass.utils import try_import_cutlass

        return config.cutlass.cutlass_dir if try_import_cutlass() else None


def _cutlass_paths() -> list[str]:
    return [
        "include",
        "tools/library/include",
        "tools/library/src",
        "tools/util/include",
    ]


def _clone_cutlass_paths(build_root: str) -> list[str]:
    cutlass_root = _cutlass_path()
    if cutlass_root is None:
        return []
    paths = []
    for path in _cutlass_paths():
        old_path = os.path.join(cutlass_root, path)
        new_path = os.path.join(build_root, path)
        shutil.copytree(old_path, new_path, dirs_exist_ok=True)
        paths.append(new_path)
    return paths


def _cutlass_include_paths() -> list[str]:
    cutlass_root = _cutlass_path()
    if cutlass_root is None:
        return []
    return [
        # Use realpath to get canonical absolute paths, in order not to mess up cache keys
        os.path.realpath(os.path.join(cutlass_root, path))
        for path in _cutlass_paths()
    ]


def _cuda_compiler() -> str | None:
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
    if arch == "101":
        return "101a"
    if arch == "103":
        return "103a"
    if arch == "110":
        return "110a"
    if arch == "120":
        return "120a"
    if arch == "121":
        return "121a"
    return arch


def _normalize_cuda_arch(arch: str) -> str:
    arch = arch.removeprefix("sm_").removeprefix("compute_").replace(".", "")
    if not re.fullmatch(r"\d+[a-z]?", arch):
        raise ValueError(f"Unrecognized CUDA arch: {arch}")
    return arch


def _cuda_arch_number(arch: str) -> int:
    arch = _normalize_cuda_arch(arch)
    if arch[-1].isalpha():
        arch = arch[:-1]
    return int(arch)


def _cuda_arch_suffix(arch: str) -> str:
    arch = _normalize_cuda_arch(arch)
    return arch[-1] if arch[-1].isalpha() else ""


def _cuda_arch_same_generation(arch: str, other: str) -> bool:
    arch = _normalize_cuda_arch(arch)
    other = _normalize_cuda_arch(other)
    if _cuda_arch_suffix(arch) or _cuda_arch_suffix(other):
        return arch == other
    return _cuda_arch_number(arch) == _cuda_arch_number(other)


def _cuda_arch_is_compatible_with_current(arch: str, current_arch: str) -> bool:
    if _cuda_arch_suffix(current_arch):
        return _normalize_cuda_arch(arch) == _normalize_cuda_arch(current_arch)
    return _cuda_arch_number(arch) >= _cuda_arch_number(current_arch)


def _aoti_cuda_target_arch() -> str:
    arch = (
        _normalize_cuda_arch(str(config.cuda.arch))
        if config.cuda.arch is not None
        else _nvcc_arch_as_compile_option()
    )
    # Triton cc overrides are numeric compute capabilities. The suffix is only
    # used for native nvcc compilation, not for the PTX AOTI packages here.
    return str(_cuda_arch_number(arch))


def _parse_gencode_options(flags: list[str]) -> OrderedSet[tuple[str, str]]:
    options: OrderedSet[tuple[str, str]] = OrderedSet()
    for flag in flags:
        if flag.startswith("-gencode="):
            option = flag.removeprefix("-gencode=")
        elif flag.startswith("-gencode "):
            option = flag.removeprefix("-gencode ")
        else:
            continue

        try:
            _, code = option.split(",code=", 1)
        except ValueError:
            continue
        code = code.removeprefix("[").removesuffix("]")
        for entry in code.split(","):
            try:
                kind, arch = entry.split("_", 1)
            except ValueError:
                continue
            if kind in ("sm", "compute"):
                options.add((kind, _normalize_cuda_arch(arch)))
    return options


def _cuda_multi_arch_gencode_options(current_arch: str | None = None) -> list[str]:
    """
    Return nvcc -gencode option payloads for AOTI CUDA fatbins.

    AOTI captures PTX for the architecture Triton compiled on. That PTX can be
    used for the current architecture and newer architectures, but not older
    ones, so explicit TORCH_CUDA_ARCH_LIST entries below the current target are
    intentionally ignored.
    """
    current_arch = _normalize_cuda_arch(current_arch or _nvcc_arch_as_compile_option())
    options: OrderedSet[tuple[str, str]] = OrderedSet()

    arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if arch_list and arch_list != "native":
        from torch.utils.cpp_extension import _get_cuda_arch_flags

        for kind, arch in _parse_gencode_options(_get_cuda_arch_flags([])):
            if kind == "sm" and _cuda_arch_is_compatible_with_current(
                arch, current_arch
            ):
                options.add((kind, arch))
            elif kind == "sm":
                log.warning(
                    "Ignoring TORCH_CUDA_ARCH_LIST entry sm_%s for AOTI CUDA "
                    "multi-arch packaging because it is not compatible with "
                    "target arch %s.",
                    arch,
                    current_arch,
                )

    if not any(
        kind == "sm" and _cuda_arch_same_generation(arch, current_arch)
        for kind, arch in options
    ):
        options.add(("sm", current_arch))

    # Always keep a PTX image for the compile architecture as the fallback for
    # GPUs newer than the generated SASS images.
    options.add(("compute", current_arch))

    def sort_key(option: tuple[str, str]) -> tuple[int, int, str]:
        kind, arch = option
        return (_cuda_arch_number(arch), 0 if kind == "sm" else 1, arch)

    return [
        f"arch=compute_{arch},code={kind}_{arch}"
        for kind, arch in sorted(options, key=sort_key)
    ]


def _cuda_gencode_options_have_non_current_sass(
    gencode_options: list[str], current_arch: str | None = None
) -> bool:
    current_arch = _normalize_cuda_arch(current_arch or _nvcc_arch_as_compile_option())
    for kind, arch in _parse_gencode_options(
        [f"-gencode={option}" for option in gencode_options]
    ):
        if kind == "sm" and not _cuda_arch_same_generation(arch, current_arch):
            return True
    return False


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
        "-std=c++20",
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
