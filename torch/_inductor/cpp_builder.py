# This CPP JIT builder is designed to support both Windows and Linux OS.
# The design document please check this RFC: https://github.com/pytorch/pytorch/issues/124245

import copy
import errno
import functools
import logging
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import sysconfig
import warnings
from pathlib import Path
from typing import List, Sequence, Tuple, Union

import torch
from torch._inductor import config, exc
from torch._inductor.codecache import (
    _get_python_include_dirs,
    _LINKER_SCRIPT,
    _transform_cuda_paths,
    get_lock_dir,
    invalid_vec_isa,
    LOCK_TIMEOUT,
    VecISA,
)
from torch._inductor.runtime.runtime_utils import cache_dir

if config.is_fbcode():
    from triton.fb import build_paths  # noqa: F401

    from torch._inductor.fb.utils import (
        log_global_cache_errors,
        log_global_cache_stats,
        log_global_cache_vals,
        use_global_cache,
    )
else:

    def log_global_cache_errors(*args, **kwargs):
        pass

    def log_global_cache_stats(*args, **kwargs):
        pass

    def log_global_cache_vals(*args, **kwargs):
        pass

    def use_global_cache() -> bool:
        return False


# Windows need setup a temp dir to store .obj files.
_BUILD_TEMP_DIR = "CxxBuild"

# initialize variables for compilation
_IS_LINUX = sys.platform.startswith("linux")
_IS_MACOS = sys.platform.startswith("darwin")
_IS_WINDOWS = sys.platform == "win32"


log = logging.getLogger(__name__)


@functools.lru_cache(1)
def cpp_compiler_search(search: str) -> str:
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
    raise exc.InvalidCxxCompiler()  # noqa: RSE102


def install_gcc_via_conda() -> str:
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


def _get_cpp_compiler() -> str:
    if _IS_WINDOWS:
        compiler = os.environ.get("CXX", "cl")
    else:
        if config.is_fbcode():
            return build_paths.cc()
        if isinstance(config.cpp.cxx, (list, tuple)):
            search = tuple(config.cpp.cxx)
        else:
            search = (config.cpp.cxx,)
        compiler = cpp_compiler_search(search)
    return compiler


def _is_gcc(cpp_compiler) -> bool:
    return bool(re.search(r"(gcc|g\+\+)", cpp_compiler))


def is_gcc() -> bool:
    return _is_gcc(_get_cpp_compiler())


def _is_clang(cpp_compiler) -> bool:
    # Mac OS apple clang maybe named as gcc, need check compiler info.
    if sys.platform == "darwin":
        return is_apple_clang(cpp_compiler)
    return bool(re.search(r"(clang|clang\+\+)", cpp_compiler))


def is_clang() -> bool:
    compiler = _get_cpp_compiler()
    return _is_clang(compiler)


@functools.lru_cache(None)
def is_apple_clang(cpp_compiler) -> bool:
    version_string = subprocess.check_output([cpp_compiler, "--version"]).decode("utf8")
    return "Apple" in version_string.splitlines()[0]


def _append_list(dest_list: List[str], src_list: List[str]):
    for item in src_list:
        dest_list.append(copy.deepcopy(item))


def _remove_duplication_in_list(orig_list: List[str]) -> List[str]:
    new_list: List[str] = []
    for item in orig_list:
        if item not in new_list:
            new_list.append(item)
    return new_list


def _create_if_dir_not_exist(path_dir):
    if not os.path.exists(path_dir):
        try:
            Path(path_dir).mkdir(parents=True, exist_ok=True)
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise RuntimeError(  # noqa: TRY200 (Use `raise from`)
                    f"Fail to create path {path_dir}"
                )


def _remove_dir(path_dir):
    if os.path.exists(path_dir):
        for root, dirs, files in os.walk(path_dir, topdown=False):
            for name in files:
                file_path = os.path.join(root, name)
                os.remove(file_path)
            for name in dirs:
                dir_path = os.path.join(root, name)
                os.rmdir(dir_path)
        os.rmdir(path_dir)


def run_command_line(cmd_line, cwd=None):
    cmd = shlex.split(cmd_line)
    try:
        status = subprocess.check_output(args=cmd, cwd=cwd, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        output = e.output.decode("utf-8")
        openmp_problem = "'omp.h' file not found" in output or "libomp" in output
        if openmp_problem and sys.platform == "darwin":
            instruction = (
                "\n\nOpenMP support not found. Please try one of the following solutions:\n"
                "(1) Set the `CXX` environment variable to a compiler other than Apple clang++/g++ "
                "that has builtin OpenMP support;\n"
                "(2) install OpenMP via conda: `conda install llvm-openmp`;\n"
                "(3) install libomp via brew: `brew install libomp`;\n"
                "(4) manually setup OpenMP and set the `OMP_PREFIX` environment variable to point to a path"
                " with `include/omp.h` under it."
            )
            output += instruction
        raise exc.CppCompileError(cmd, output) from e
    return status


class BuildOptionsBase:
    """
    This is the Base class for store cxx build options, as a template.
    Acturally, to build a cxx shared library. We just need to select a compiler
    and maintains the suitable args.
    """

    def __init__(self) -> None:
        self._compiler = ""
        self._definations: List[str] = []
        self._include_dirs: List[str] = []
        self._cflags: List[str] = []
        self._ldflags: List[str] = []
        self._libraries_dirs: List[str] = []
        self._libraries: List[str] = []
        # Some args is hard to abstract to OS compatable, passthough it directly.
        self._passthough_args: List[str] = []

        self._aot_mode: bool = False
        self._use_absolute_path: bool = False
        self._compile_only: bool = False

    def _remove_duplicate_options(self):
        self._definations = _remove_duplication_in_list(self._definations)
        self._include_dirs = _remove_duplication_in_list(self._include_dirs)
        self._cflags = _remove_duplication_in_list(self._cflags)
        self._ldflags = _remove_duplication_in_list(self._ldflags)
        self._libraries_dirs = _remove_duplication_in_list(self._libraries_dirs)
        self._libraries = _remove_duplication_in_list(self._libraries)
        self._passthough_args = _remove_duplication_in_list(self._passthough_args)

    def get_compiler(self) -> str:
        return self._compiler

    def get_definations(self) -> List[str]:
        return self._definations

    def get_include_dirs(self) -> List[str]:
        return self._include_dirs

    def get_cflags(self) -> List[str]:
        return self._cflags

    def get_ldflags(self) -> List[str]:
        return self._ldflags

    def get_libraries_dirs(self) -> List[str]:
        return self._libraries_dirs

    def get_libraries(self) -> List[str]:
        return self._libraries

    def get_passthough_args(self) -> List[str]:
        return self._passthough_args

    def get_aot_mode(self) -> bool:
        return self._aot_mode

    def get_use_absolute_path(self) -> bool:
        return self._use_absolute_path

    def get_compile_only(self) -> bool:
        return self._compile_only


def _get_warning_all_cflag(warning_all: bool = True) -> List[str]:
    if not _IS_WINDOWS:
        return ["Wall"] if warning_all else []
    else:
        return []


def _get_cpp_std_cflag(std_num: str = "c++17") -> List[str]:
    if _IS_WINDOWS:
        return [f"std:{std_num}"]
    else:
        return [f"std={std_num}"]


def _get_linux_cpp_cflags(cpp_compiler) -> List[str]:
    if not _IS_WINDOWS:
        cflags = ["Wno-unused-variable", "Wno-unknown-pragmas"]
        if _is_clang(cpp_compiler):
            cflags.append("Werror=ignored-optimization-argument")
        return cflags
    else:
        return []


def _get_optimization_cflags() -> List[str]:
    if _IS_WINDOWS:
        return ["O2"]
    else:
        cflags = ["O0", "g"] if config.aot_inductor.debug_compile else ["O3", "DNDEBUG"]
        cflags.append("ffast-math")
        cflags.append("fno-finite-math-only")

        if not config.cpp.enable_unsafe_math_opt_flag:
            cflags.append("fno-unsafe-math-optimizations")
        if not config.cpp.enable_floating_point_contract_flag:
            cflags.append("ffp-contract=off")

        if config.is_fbcode():
            # FIXME: passing `-fopenmp` adds libgomp.so to the generated shared library's dependencies.
            # This causes `ldopen` to fail in fbcode, because libgomp does not exist in the default paths.
            # We will fix it later by exposing the lib path.
            return cflags

        if sys.platform == "darwin":
            # Per https://mac.r-project.org/openmp/ right way to pass `openmp` flags to MacOS is via `-Xclang`
            # Also, `-march=native` is unrecognized option on M1
            cflags.append("Xclang")
        else:
            if platform.machine() == "ppc64le":
                cflags.append("mcpu=native")
            else:
                cflags.append("march=native")

        # Internal cannot find libgomp.so
        if not config.is_fbcode():
            cflags.append("fopenmp")

        return cflags


def _get_shared_cflag(compile_only: bool) -> List[str]:
    if _IS_WINDOWS:
        SHARED_FLAG = ["DLL"]
    else:
        if compile_only:
            return ["fPIC"]
        if platform.system() == "Darwin" and "clang" in _get_cpp_compiler():
            # This causes undefined symbols to behave the same as linux
            return ["shared", "fPIC", "undefined dynamic_lookup"]
        else:
            return ["shared", "fPIC"]

    return SHARED_FLAG


def get_cpp_options(
    cpp_compiler,
    compile_only: bool,
    warning_all: bool = True,
    extra_flags: Sequence[str] = (),
):
    definations: List[str] = []
    include_dirs: List[str] = []
    cflags: List[str] = []
    ldflags: List[str] = []
    libraries_dirs: List[str] = []
    libraries: List[str] = []
    passthough_args: List[str] = []

    cflags = (
        _get_shared_cflag(compile_only)
        + _get_optimization_cflags()
        + _get_warning_all_cflag(warning_all)
        + _get_cpp_std_cflag()
        + _get_linux_cpp_cflags(cpp_compiler)
    )

    passthough_args.append(" ".join(extra_flags))

    return (
        definations,
        include_dirs,
        cflags,
        ldflags,
        libraries_dirs,
        libraries,
        passthough_args,
    )


class CppOptions(BuildOptionsBase):
    """
    This class is inherited from BuildOptionsBase, and as cxx build options.
    This option need contains basic cxx build option, which contains:
    1. OS related args.
    2. Toolchains related args.
    3. Cxx standard related args.
    Note:
    1. This Options is good for assist modules build, such as x86_isa_help.
    """

    def __init__(
        self,
        compile_only: bool,
        warning_all: bool = True,
        extra_flags: Sequence[str] = (),
        use_absolute_path: bool = False,
    ) -> None:
        super().__init__()
        self._compiler = _get_cpp_compiler()
        self._use_absolute_path = use_absolute_path
        self._compile_only = compile_only

        (
            definations,
            include_dirs,
            cflags,
            ldflags,
            libraries_dirs,
            libraries,
            passthough_args,
        ) = get_cpp_options(
            cpp_compiler=self._compiler,
            compile_only=compile_only,
            extra_flags=extra_flags,
            warning_all=warning_all,
        )

        _append_list(self._definations, definations)
        _append_list(self._include_dirs, include_dirs)
        _append_list(self._cflags, cflags)
        _append_list(self._ldflags, ldflags)
        _append_list(self._libraries_dirs, libraries_dirs)
        _append_list(self._libraries, libraries)
        _append_list(self._passthough_args, passthough_args)
        self._remove_duplicate_options()


def _get_glibcxx_abi_build_flags() -> List[str]:
    if not _IS_WINDOWS:
        return ["-D_GLIBCXX_USE_CXX11_ABI=" + str(int(torch._C._GLIBCXX_USE_CXX11_ABI))]
    else:
        return []


def _get_torch_cpp_wrapper_defination() -> List[str]:
    return ["TORCH_INDUCTOR_CPP_WRAPPER"]


def _use_custom_generated_macros() -> List[str]:
    return [" C10_USING_CUSTOM_GENERATED_MACROS"]


def _use_fb_internal_macros() -> List[str]:
    if not _IS_WINDOWS:
        if config.is_fbcode():
            fb_internal_macros = [
                "C10_USE_GLOG",
                "C10_USE_MINIMAL_GLOG",
                "C10_DISABLE_TENSORIMPL_EXTENSIBILITY",
            ]
            # TODO: this is to avoid FC breakage for fbcode. When using newly
            # generated model.so on an older verion of PyTorch, need to use
            # the v1 version for aoti_torch_create_tensor_from_blob
            create_tensor_from_blob_v1 = "AOTI_USE_CREATE_TENSOR_FROM_BLOB_V1"

            fb_internal_macros.append(create_tensor_from_blob_v1)

            # TODO: remove comments later:
            # Moved to _get_openmp_args
            # openmp_lib = build_paths.openmp_lib()
            # return [f"-Wp,-fopenmp {openmp_lib} {preprocessor_flags}"]
            return fb_internal_macros
        else:
            return []
    else:
        return []


def _setup_standard_sys_libs(
    cpp_compiler,
    aot_mode: bool,
    use_absolute_path: bool,
):
    cflags: List[str] = []
    include_dirs: List[str] = []
    passthough_args: List[str] = []
    if _IS_WINDOWS:
        return cflags, include_dirs, passthough_args

    if config.is_fbcode():
        cflags.append("nostdinc")
        include_dirs.append(build_paths.sleef())
        include_dirs.append(build_paths.cc_include())
        include_dirs.append(build_paths.libgcc())
        include_dirs.append(build_paths.libgcc_arch())
        include_dirs.append(build_paths.libgcc_backward())
        include_dirs.append(build_paths.glibc())
        include_dirs.append(build_paths.linux_kernel())
        include_dirs.append("include")

        if aot_mode and not use_absolute_path:
            linker_script = _LINKER_SCRIPT
        else:
            linker_script = os.path.basename(_LINKER_SCRIPT)

        if _is_clang(cpp_compiler):
            passthough_args.append(" --rtlib=compiler-rt")
            passthough_args.append(" -fuse-ld=lld")
            passthough_args.append(f" -Wl,--script={linker_script}")
            passthough_args.append(" -B" + build_paths.glibc_lib())
            passthough_args.append(" -L" + build_paths.glibc_lib())

    return cflags, include_dirs, passthough_args


@functools.lru_cache
def _cpp_prefix_path() -> str:
    from torch._inductor.codecache import write  # TODO

    path = Path(Path(__file__).parent).parent / "codegen/cpp_prefix.h"
    with path.open() as f:
        content = f.read()
        _, filename = write(
            content,
            "h",
        )
    return filename


def _get_build_args_of_chosen_isa(vec_isa: VecISA):
    macros = []
    build_flags = []
    if vec_isa != invalid_vec_isa:
        # Add Windows support later.
        for x in vec_isa.build_macro():
            macros.append(copy.deepcopy(x))

        build_flags = [vec_isa.build_arch_flags()]

    if config.is_fbcode() and vec_isa != invalid_vec_isa:
        cap = str(vec_isa).upper()
        macros = [
            f"CPU_CAPABILITY={cap}",
            f"CPU_CAPABILITY_{cap}",
            f"HAVE_{cap}_CPU_DEFINITION",
        ]

    return macros, build_flags


def _get_torch_related_args(include_pytorch: bool, aot_mode: bool):
    from torch.utils.cpp_extension import _TORCH_PATH, TORCH_LIB_PATH

    include_dirs = [
        os.path.join(_TORCH_PATH, "include"),
        os.path.join(_TORCH_PATH, "include", "torch", "csrc", "api", "include"),
        # Some internal (old) Torch headers don't properly prefix their includes,
        # so we need to pass -Itorch/lib/include/TH as well.
        os.path.join(_TORCH_PATH, "include", "TH"),
        os.path.join(_TORCH_PATH, "include", "THC"),
    ]
    libraries_dirs = [TORCH_LIB_PATH]
    libraries = []
    if sys.platform == "linux" and not config.is_fbcode():
        libraries = ["torch", "torch_cpu"]
        if not aot_mode:
            libraries.append("torch_python")

    if _IS_WINDOWS:
        libraries.append("sleef")

    # Unconditionally import c10 for non-abi-compatible mode to use TORCH_CHECK - See PyTorch #108690
    if not config.abi_compatible:
        libraries.append("c10")
        libraries_dirs.append(TORCH_LIB_PATH)

    return include_dirs, libraries_dirs, libraries


def _get_python_related_args():
    python_include_dirs = _get_python_include_dirs()
    python_include_path = sysconfig.get_path(
        "include", scheme="nt" if _IS_WINDOWS else "posix_prefix"
    )
    if python_include_path is not None:
        python_include_dirs.append(python_include_path)

    if _IS_WINDOWS:
        python_path = os.path.dirname(sys.executable)
        python_lib_path = [os.path.join(python_path, "libs")]
    else:
        python_lib_path = [sysconfig.get_config_var("LIBDIR")]

    if config.is_fbcode():
        python_include_dirs.append(build_paths.python())

    return python_include_dirs, python_lib_path


def _get_openmp_args(cpp_compiler):
    cflags: List[str] = []
    ldflags: List[str] = []
    include_dir_paths: List[str] = []
    lib_dir_paths: List[str] = []
    libs: List[str] = []
    passthough_args: List[str] = []
    if _IS_MACOS:
        from torch._inductor.codecache import (
            homebrew_libomp,
            is_conda_llvm_openmp_installed,
        )

        # only Apple builtin compilers (Apple Clang++) require openmp
        omp_available = not is_apple_clang(cpp_compiler)

        # check the `OMP_PREFIX` environment first
        omp_prefix = os.getenv("OMP_PREFIX")
        if omp_prefix is not None:
            header_path = os.path.join(omp_prefix, "include", "omp.h")
            valid_env = os.path.exists(header_path)
            if valid_env:
                include_dir_paths.append(os.path.join(omp_prefix, "include"))
                lib_dir_paths.append(os.path.join(omp_prefix, "lib"))
            else:
                warnings.warn("environment variable `OMP_PREFIX` is invalid.")
            omp_available = omp_available or valid_env

        if not omp_available:
            libs.append("omp")

        # prefer to use openmp from `conda install llvm-openmp`
        conda_prefix = os.getenv("CONDA_PREFIX")
        if not omp_available and conda_prefix is not None:
            omp_available = is_conda_llvm_openmp_installed()
            if omp_available:
                conda_lib_path = os.path.join(conda_prefix, "lib")
                include_dir_paths.append(os.path.join(conda_prefix, "include"))
                lib_dir_paths.append(conda_lib_path)
                # Prefer Intel OpenMP on x86 machine
                if os.uname().machine == "x86_64" and os.path.exists(
                    os.path.join(conda_lib_path, "libiomp5.dylib")
                ):
                    libs.append("iomp5")

        # next, try to use openmp from `brew install libomp`
        if not omp_available:
            omp_available, libomp_path = homebrew_libomp()
            if omp_available:
                include_dir_paths.append(os.path.join(libomp_path, "include"))
                lib_dir_paths.append(os.path.join(libomp_path, "lib"))

        # if openmp is still not available, we let the compiler to have a try,
        # and raise error together with instructions at compilation error later
    elif _IS_WINDOWS:
        # /openmp, /openmp:llvm
        # llvm on Windows, new openmp: https://devblogs.microsoft.com/cppblog/msvc-openmp-update/
        # msvc openmp: https://learn.microsoft.com/zh-cn/cpp/build/reference/openmp-enable-openmp-2-0-support?view=msvc-170

        cflags.append("openmp")
        libs = []
    else:
        if config.is_fbcode():
            include_dir_paths.append(build_paths.openmp())

            openmp_lib = build_paths.openmp_lib()
            fb_openmp_extra_flags = f"-Wp,-fopenmp {openmp_lib}"
            passthough_args.append(fb_openmp_extra_flags)

            libs.append("omp")
        else:
            if _is_clang(cpp_compiler):
                # TODO: fix issue, can't find omp.h
                cflags.append("fopenmp")
                libs.append("gomp")
            else:
                cflags.append("fopenmp")
                libs.append("gomp")

    return cflags, ldflags, include_dir_paths, lib_dir_paths, libs, passthough_args


def get_mmap_self_macro(use_mmap_weights: bool) -> List[str]:
    macros = []
    if use_mmap_weights:
        macros.append(" USE_MMAP_SELF")
    return macros


def get_cpp_torch_options(
    cpp_compiler,
    vec_isa: VecISA,
    include_pytorch: bool,
    aot_mode: bool,
    compile_only: bool,
    use_absolute_path: bool,
    use_mmap_weights: bool,
):
    definations: List[str] = []
    include_dirs: List[str] = []
    cflags: List[str] = []
    ldflags: List[str] = []
    libraries_dirs: List[str] = []
    libraries: List[str] = []
    passthough_args: List[str] = []

    torch_cpp_wrapper_definations = _get_torch_cpp_wrapper_defination()
    use_custom_generated_macros_definations = _use_custom_generated_macros()

    (
        sys_libs_cflags,
        sys_libs_include_dirs,
        sys_libs_passthough_args,
    ) = _setup_standard_sys_libs(cpp_compiler, aot_mode, use_absolute_path)

    isa_macros, isa_ps_args_build_flags = _get_build_args_of_chosen_isa(vec_isa)

    (
        torch_include_dirs,
        torch_libraries_dirs,
        torch_libraries,
    ) = _get_torch_related_args(include_pytorch=include_pytorch, aot_mode=aot_mode)

    python_include_dirs, python_libraries_dirs = _get_python_related_args()

    (
        omp_cflags,
        omp_ldflags,
        omp_include_dir_paths,
        omp_lib_dir_paths,
        omp_lib,
        omp_passthough_args,
    ) = _get_openmp_args(cpp_compiler)

    cxx_abi_passthough_args = _get_glibcxx_abi_build_flags()
    fb_macro_passthough_args = _use_fb_internal_macros()

    mmap_self_macros = get_mmap_self_macro(use_mmap_weights)

    definations = (
        torch_cpp_wrapper_definations
        + use_custom_generated_macros_definations
        + isa_macros
        + fb_macro_passthough_args
        + mmap_self_macros
    )
    include_dirs = (
        sys_libs_include_dirs
        + python_include_dirs
        + torch_include_dirs
        + omp_include_dir_paths
    )
    cflags = sys_libs_cflags + omp_cflags
    ldflags = omp_ldflags
    libraries_dirs = python_libraries_dirs + torch_libraries_dirs + omp_lib_dir_paths
    libraries = torch_libraries + omp_lib
    passthough_args = (
        sys_libs_passthough_args
        + isa_ps_args_build_flags
        + cxx_abi_passthough_args
        + omp_passthough_args
    )

    return (
        definations,
        include_dirs,
        cflags,
        ldflags,
        libraries_dirs,
        libraries,
        passthough_args,
    )


class CppTorchOptions(CppOptions):
    """
    This class is inherited from CppTorchOptions, which automatic contains
    base cxx build options. And then it will maintains torch related build
    args.
    1. Torch include_directories, libraries, libraries_directories.
    2. Python include_directories, libraries, libraries_directories.
    3. OpenMP related.
    4. Torch MACROs.
    5. MISC
    """

    def __init__(
        self,
        vec_isa: VecISA,
        include_pytorch: bool = False,
        warning_all: bool = True,
        aot_mode: bool = False,
        compile_only: bool = False,
        use_absolute_path: bool = False,
        use_mmap_weights: bool = False,
        shared: bool = True,
        extra_flags: Sequence[str] = (),
    ) -> None:
        super().__init__(
            compile_only=compile_only,
            warning_all=warning_all,
            extra_flags=extra_flags,
            use_absolute_path=use_absolute_path,
        )

        self._aot_mode = aot_mode

        (
            torch_definations,
            torch_include_dirs,
            torch_cflags,
            torch_ldflags,
            torch_libraries_dirs,
            torch_libraries,
            torch_passthough_args,
        ) = get_cpp_torch_options(
            cpp_compiler=self._compiler,
            vec_isa=vec_isa,
            include_pytorch=include_pytorch,
            aot_mode=aot_mode,
            compile_only=compile_only,
            use_absolute_path=use_absolute_path,
            use_mmap_weights=use_mmap_weights,
        )

        if compile_only:
            torch_libraries_dirs = []
            torch_libraries = []

        _append_list(self._definations, torch_definations)
        _append_list(self._include_dirs, torch_include_dirs)
        _append_list(self._cflags, torch_cflags)
        _append_list(self._ldflags, torch_ldflags)
        _append_list(self._libraries_dirs, torch_libraries_dirs)
        _append_list(self._libraries, torch_libraries)
        _append_list(self._passthough_args, torch_passthough_args)
        self._remove_duplicate_options()


def get_cpp_torch_cuda_options(cuda: bool, aot_mode: bool = False):
    definations: List[str] = []
    include_dirs: List[str] = []
    cflags: List[str] = []
    ldflags: List[str] = []
    libraries_dirs: List[str] = []
    libraries: List[str] = []
    passthough_args: List[str] = []

    if (
        config.is_fbcode()
        and "CUDA_HOME" not in os.environ
        and "CUDA_PATH" not in os.environ
    ):
        os.environ["CUDA_HOME"] = build_paths.cuda()

    from torch.utils import cpp_extension

    include_dirs = cpp_extension.include_paths(cuda)
    libraries_dirs = cpp_extension.library_paths(cuda)

    if cuda:
        definations.append(" USE_ROCM" if torch.version.hip else " USE_CUDA")

        if torch.version.hip is not None:
            if config.is_fbcode():
                libraries += ["amdhip64"]
            else:
                libraries += ["c10_hip", "torch_hip"]
                definations.append(" __HIP_PLATFORM_AMD__")
        else:
            if config.is_fbcode():
                libraries += ["cuda"]
            else:
                if config.is_fbcode():
                    libraries += ["cuda"]
                else:
                    libraries += ["c10_cuda", "cuda", "torch_cuda"]

    if aot_mode:
        cpp_prefix_include_dir = [f"{os.path.dirname(_cpp_prefix_path())}"]
        include_dirs += cpp_prefix_include_dir

        if cuda and torch.version.hip is None:
            _transform_cuda_paths(libraries_dirs)

    if config.is_fbcode():
        if torch.version.hip is not None:
            include_dirs.append(os.path.join(build_paths.rocm(), "include"))
        else:
            include_dirs.append(os.path.join(build_paths.cuda(), "include"))

    if aot_mode and cuda and config.is_fbcode():
        if torch.version.hip is None:
            # TODO: make static link better on Linux.
            passthough_args = ["-Wl,-Bstatic -lcudart_static -Wl,-Bdynamic"]

    return (
        definations,
        include_dirs,
        cflags,
        ldflags,
        libraries_dirs,
        libraries,
        passthough_args,
    )


class CppTorchCudaOptions(CppTorchOptions):
    """
    This class is inherited from CppTorchOptions, which automatic contains
    base cxx build options and torch common build options. And then it will
    maintains cuda device related build args.
    """

    def __init__(
        self,
        vec_isa: VecISA,
        include_pytorch: bool = False,
        cuda: bool = True,
        aot_mode: bool = False,
        compile_only: bool = False,
        use_absolute_path: bool = False,
        use_mmap_weights: bool = False,
        shared: bool = True,
        extra_flags: Sequence[str] = (),
    ) -> None:
        super().__init__(
            vec_isa=vec_isa,
            include_pytorch=include_pytorch,
            aot_mode=aot_mode,
            compile_only=compile_only,
            use_absolute_path=use_absolute_path,
            use_mmap_weights=use_mmap_weights,
            extra_flags=extra_flags,
        )

        cuda_definations: List[str] = []
        cuda_include_dirs: List[str] = []
        cuda_cflags: List[str] = []
        cuda_ldflags: List[str] = []
        cuda_libraries_dirs: List[str] = []
        cuda_libraries: List[str] = []
        cuda_passthough_args: List[str] = []

        (
            cuda_definations,
            cuda_include_dirs,
            cuda_cflags,
            cuda_ldflags,
            cuda_libraries_dirs,
            cuda_libraries,
            cuda_passthough_args,
        ) = get_cpp_torch_cuda_options(cuda=cuda, aot_mode=aot_mode)

        if compile_only:
            cuda_libraries_dirs = []
            cuda_libraries = []

        _append_list(self._definations, cuda_definations)
        _append_list(self._include_dirs, cuda_include_dirs)
        _append_list(self._cflags, cuda_cflags)
        _append_list(self._ldflags, cuda_ldflags)
        _append_list(self._libraries_dirs, cuda_libraries_dirs)
        _append_list(self._libraries, cuda_libraries)
        _append_list(self._passthough_args, cuda_passthough_args)
        self._remove_duplicate_options()


def get_name_and_dir_from_output_file_path(
    aot_mode: bool, use_absolute_path: bool, file_path: str
):
    name_and_ext = os.path.basename(file_path)
    name, ext = os.path.splitext(name_and_ext)
    dir = os.path.dirname(file_path)

    if config.is_fbcode():
        if not (aot_mode and not use_absolute_path):
            dir = "."
    return name, dir


class CppBuilder:
    """
    CppBuilder is a cpp jit builder, and it supports both Windows, Linux and MacOS.
    Args:
        name:
            1. Build target name, the final target file will append extension type automatically.
            2. Due to the CppBuilder is supports mutliple OS, it will maintains ext for OS difference.
        sources:
            Source code file list to be built.
        BuildOption:
            Build options to the builder.
        output_dir:
            1. The output_dir the taget file will output to.
            2. The default value is empty string, and then the use current dir as output dir.
            3. Final target file: output_dir/name.ext
    """

    def get_shared_lib_ext(self) -> str:
        SHARED_LIB_EXT = ".dll" if _IS_WINDOWS else ".so"
        return SHARED_LIB_EXT

    def get_object_ext(self) -> str:
        EXT = ".obj" if _IS_WINDOWS else ".o"
        return EXT

    def __init__(
        self,
        name: str,
        sources: Union[str, List[str]],
        BuildOption: BuildOptionsBase,
        output_dir: str = "",
    ) -> None:
        self._compiler = ""
        self._cflags_args = ""
        self._definations_args = ""
        self._include_dirs_args = ""
        self._ldflags_args = ""
        self._libraries_dirs_args = ""
        self._libraries_args = ""
        self._passthough_parameters_args = ""

        self._output_dir = ""
        self._target_file = ""

        self._use_absolute_path: bool = False

        self._name = name

        # Code start here, initial self internal veriables firstly.
        self._compiler = BuildOption.get_compiler()
        self._use_absolute_path = BuildOption.get_use_absolute_path()

        if len(output_dir) == 0:
            self._output_dir = os.path.dirname(os.path.abspath(__file__))
        else:
            self._output_dir = output_dir

        self._compile_only = BuildOption.get_compile_only()
        file_ext = (
            self.get_object_ext() if self._compile_only else self.get_shared_lib_ext()
        )
        self._target_file = os.path.join(self._output_dir, f"{self._name}{file_ext}")

        if isinstance(sources, str):
            sources = [sources]

        if config.is_fbcode():
            if BuildOption.get_aot_mode() and not self._use_absolute_path:
                inp_name = sources
                # output process @ get_name_and_dir_from_output_file_path
            else:
                # We need to copy any absolute-path torch includes
                inp_name = [os.path.basename(i) for i in sources]
                self._target_file = os.path.basename(self._target_file)

            self._sources_args = " ".join(inp_name)
        else:
            self._sources_args = " ".join(sources)

        for cflag in BuildOption.get_cflags():
            if _IS_WINDOWS:
                self._cflags_args += f"/{cflag} "
            else:
                self._cflags_args += f"-{cflag} "

        for defination in BuildOption.get_definations():
            if _IS_WINDOWS:
                self._definations_args += f"/D {defination} "
            else:
                self._definations_args += f"-D {defination} "

        for inc_dir in BuildOption.get_include_dirs():
            if _IS_WINDOWS:
                self._include_dirs_args += f"/I {inc_dir} "
            else:
                self._include_dirs_args += f"-I{inc_dir} "

        for ldflag in BuildOption.get_ldflags():
            if _IS_WINDOWS:
                self._ldflags_args += f"/{ldflag} "
            else:
                self._ldflags_args += f"-{ldflag} "

        for lib_dir in BuildOption.get_libraries_dirs():
            if _IS_WINDOWS:
                self._libraries_dirs_args += f'/LIBPATH:"{lib_dir}" '
            else:
                self._libraries_dirs_args += f"-L{lib_dir} "

        for lib in BuildOption.get_libraries():
            if _IS_WINDOWS:
                self._libraries_args += f'"{lib}.lib" '
            else:
                self._libraries_args += f"-l{lib} "

        for passthough_arg in BuildOption.get_passthough_args():
            self._passthough_parameters_args += f"{passthough_arg} "

    def get_command_line(self) -> str:
        def format_build_command(
            compiler,
            sources,
            include_dirs_args,
            definations_args,
            cflags_args,
            ldflags_args,
            libraries_args,
            libraries_dirs_args,
            passthougn_args,
            target_file,
        ):
            if _IS_WINDOWS:
                # https://learn.microsoft.com/en-us/cpp/build/walkthrough-compile-a-c-program-on-the-command-line?view=msvc-1704
                # https://stackoverflow.com/a/31566153
                cmd = (
                    f"{compiler} {include_dirs_args} {definations_args} {cflags_args} {sources} "
                    f"{passthougn_args} /LD /Fe{target_file} /link {libraries_dirs_args} {libraries_args} {ldflags_args} "
                )
                cmd = cmd.replace("\\", "/")
            else:
                compile_only_arg = "-c" if self._compile_only else ""
                cmd = re.sub(
                    r"[ \n]+",
                    " ",
                    f"""
                    {compiler} {sources} {definations_args} {cflags_args} {include_dirs_args}
                    {passthougn_args} {ldflags_args} {libraries_args} {libraries_dirs_args} {compile_only_arg} -o {target_file}
                    """,
                ).strip()
            return cmd

        command_line = format_build_command(
            compiler=self._compiler,
            sources=self._sources_args,
            include_dirs_args=self._include_dirs_args,
            definations_args=self._definations_args,
            cflags_args=self._cflags_args,
            ldflags_args=self._ldflags_args,
            libraries_args=self._libraries_args,
            libraries_dirs_args=self._libraries_dirs_args,
            passthougn_args=self._passthough_parameters_args,
            target_file=self._target_file,
        )
        return command_line

    def get_target_file_path(self):
        return self._target_file

    def convert_to_cpp_extension_args(self):
        include_dirs = self._include_dirs_args
        cflags = (
            self._cflags_args
            + self._definations_args
            + self._passthough_parameters_args
        )
        ldflags = self._ldflags_args + self._libraries_args + self._libraries_dirs_args

        return include_dirs, cflags, ldflags

    def build(self) -> Tuple[int, str]:
        """
        It is must need a temperary directory to store object files in Windows.
        After build completed, delete the temperary directory to save disk space.
        """
        _create_if_dir_not_exist(self._output_dir)
        _build_tmp_dir = os.path.join(
            self._output_dir, f"{self._name}_{_BUILD_TEMP_DIR}"
        )
        _create_if_dir_not_exist(_build_tmp_dir)

        build_cmd = self.get_command_line()

        status = run_command_line(build_cmd, cwd=_build_tmp_dir)

        _remove_dir(_build_tmp_dir)
        return status, self._target_file
