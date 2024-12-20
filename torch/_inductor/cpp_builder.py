# This CPP builder is designed to support both Windows and Linux OS.
# The design document please check this RFC: https://github.com/pytorch/pytorch/issues/124245

import copy
import errno
import functools
import json
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
from ctypes import cdll
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

import torch
from torch._dynamo.utils import dynamo_timed
from torch._inductor import config, exc
from torch._inductor.cpu_vec_isa import invalid_vec_isa, VecISA
from torch._inductor.runtime.runtime_utils import cache_dir
from torch.torch_version import TorchVersion


if config.is_fbcode():
    from triton.fb import build_paths  # noqa: F401

    from torch._inductor.fb.utils import (
        log_global_cache_errors,
        log_global_cache_stats,
        log_global_cache_vals,
        use_global_cache,
    )
else:

    def log_global_cache_errors(*args: Any, **kwargs: Any) -> None:
        pass

    def log_global_cache_stats(*args: Any, **kwargs: Any) -> None:
        pass

    def log_global_cache_vals(*args: Any, **kwargs: Any) -> None:
        pass

    def use_global_cache() -> bool:
        return False


# Windows need setup a temp dir to store .obj files.
_BUILD_TEMP_DIR = "CxxBuild"

# initialize variables for compilation
_IS_LINUX = sys.platform.startswith("linux")
_IS_MACOS = sys.platform.startswith("darwin")
_IS_WINDOWS = sys.platform == "win32"

SUBPROCESS_DECODE_ARGS = ("utf-8",) if _IS_WINDOWS else ()

log = logging.getLogger(__name__)


# =============================== toolchain ===============================
@functools.lru_cache(1)
def cpp_compiler_search(search: str) -> str:
    from torch._inductor.codecache import get_lock_dir, LOCK_TIMEOUT

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
                from torch.utils._filelock import FileLock

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
    raise exc.InvalidCxxCompiler


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


@functools.lru_cache(None)
def check_compiler_exist_windows(compiler: str) -> None:
    """
    Check if compiler is ready, in case end user not activate MSVC environment.
    """
    try:
        subprocess.check_output([compiler, "/help"], stderr=subprocess.STDOUT)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Compiler: {compiler} is not found.") from exc
    except subprocess.SubprocessError:
        # Expected that some compiler(clang, clang++) is exist, but they not support `/help` args.
        pass


def get_cpp_compiler() -> str:
    if _IS_WINDOWS:
        compiler = os.environ.get("CXX", "cl")
        check_compiler_exist_windows(compiler)
    else:
        if config.is_fbcode():
            return build_paths.cc
        if isinstance(config.cpp.cxx, (list, tuple)):
            search = tuple(config.cpp.cxx)
        else:
            search = (config.cpp.cxx,)
        compiler = cpp_compiler_search(search)
    return compiler


@functools.lru_cache(None)
def _is_apple_clang(cpp_compiler: str) -> bool:
    version_string = subprocess.check_output([cpp_compiler, "--version"]).decode("utf8")
    return "Apple" in version_string.splitlines()[0]


def _is_clang(cpp_compiler: str) -> bool:
    # Mac OS apple clang maybe named as gcc, need check compiler info.
    if sys.platform == "darwin":
        return _is_apple_clang(cpp_compiler)
    elif _IS_WINDOWS:
        # clang suite have many compilers, and only clang-cl is supported.
        if re.search(r"((clang$)|(clang\+\+$))", cpp_compiler):
            raise RuntimeError(
                "Please use clang-cl, due to torch.compile only support MSVC-like CLI (compiler flags syntax)."
            )
        return bool(re.search(r"(clang-cl)", cpp_compiler))
    return bool(re.search(r"(clang|clang\+\+)", cpp_compiler))


def _is_gcc(cpp_compiler: str) -> bool:
    if sys.platform == "darwin" and _is_apple_clang(cpp_compiler):
        return False
    return bool(re.search(r"(gcc|g\+\+)", cpp_compiler))


@functools.lru_cache(None)
def _is_msvc_cl(cpp_compiler: str) -> bool:
    if not _IS_WINDOWS:
        return False

    try:
        output_msg = (
            subprocess.check_output([cpp_compiler, "/help"], stderr=subprocess.STDOUT)
            .strip()
            .decode(*SUBPROCESS_DECODE_ARGS)
        )
        return "Microsoft" in output_msg.splitlines()[0]
    except FileNotFoundError:
        return False

    return False


@functools.lru_cache(None)
def _is_intel_compiler(cpp_compiler: str) -> bool:
    def _check_minimal_version(compiler_version: TorchVersion) -> None:
        """
        On Windows: early version icx has `-print-file-name` issue, and can't preload correctly for inductor.
        """
        min_version = "2024.2.1" if _IS_WINDOWS else "0.0.0"
        if compiler_version < TorchVersion(min_version):
            raise RuntimeError(
                f"Intel Compiler error: less than minimal version {min_version}."
            )

    try:
        output_msg = (
            subprocess.check_output(
                [cpp_compiler, "--version"], stderr=subprocess.DEVNULL
            )
            .strip()
            .decode(*SUBPROCESS_DECODE_ARGS)
        )
        is_intel_compiler = "Intel" in output_msg.splitlines()[0]
        if is_intel_compiler:
            if _IS_WINDOWS:
                if re.search(r"((icx$)|(icx-cc$))", cpp_compiler):
                    raise RuntimeError(
                        "Please use icx-cl, due to torch.compile only support MSVC-like CLI (compiler flags syntax)."
                    )

            # Version check
            icx_ver_search = re.search(r"(\d+[.]\d+[.]\d+[.]\d+)", output_msg)
            if icx_ver_search is not None:
                icx_ver = icx_ver_search.group(1)
                _check_minimal_version(TorchVersion(icx_ver))

        return is_intel_compiler
    except FileNotFoundError:
        return False
    except subprocess.SubprocessError:
        # --version args not support.
        return False

    return False


@functools.lru_cache(None)
def is_gcc() -> bool:
    return _is_gcc(get_cpp_compiler())


@functools.lru_cache(None)
def is_clang() -> bool:
    return _is_clang(get_cpp_compiler())


@functools.lru_cache(None)
def is_intel_compiler() -> bool:
    return _is_intel_compiler(get_cpp_compiler())


@functools.lru_cache(None)
def is_apple_clang() -> bool:
    return _is_apple_clang(get_cpp_compiler())


@functools.lru_cache(None)
def is_msvc_cl() -> bool:
    return _is_msvc_cl(get_cpp_compiler())


def get_compiler_version_info(compiler: str) -> str:
    env = os.environ.copy()
    env["LC_ALL"] = "C"  # Don't localize output
    try:
        version_string = subprocess.check_output(
            [compiler, "-v"], stderr=subprocess.STDOUT, env=env
        ).decode(*SUBPROCESS_DECODE_ARGS)
    except Exception:
        try:
            version_string = subprocess.check_output(
                [compiler, "--version"], stderr=subprocess.STDOUT, env=env
            ).decode(*SUBPROCESS_DECODE_ARGS)
        except Exception:
            return ""
    # Mutiple lines to one line string.
    version_string = version_string.replace("\r", "_")
    version_string = version_string.replace("\n", "_")
    return version_string


# =============================== cpp builder ===============================
def _append_list(dest_list: List[str], src_list: List[str]) -> None:
    dest_list.extend(copy.deepcopy(item) for item in src_list)


def _remove_duplication_in_list(orig_list: List[str]) -> List[str]:
    new_list: List[str] = []
    for item in orig_list:
        if item not in new_list:
            new_list.append(item)
    return new_list


def _create_if_dir_not_exist(path_dir: str) -> None:
    if not os.path.exists(path_dir):
        try:
            Path(path_dir).mkdir(parents=True, exist_ok=True)
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise RuntimeError(  # noqa: TRY200 (Use `raise from`)
                    f"Fail to create path {path_dir}"
                )


def _remove_dir(path_dir: str) -> None:
    if os.path.exists(path_dir):
        for root, dirs, files in os.walk(path_dir, topdown=False):
            for name in files:
                file_path = os.path.join(root, name)
                os.remove(file_path)
            for name in dirs:
                dir_path = os.path.join(root, name)
                os.rmdir(dir_path)
        os.rmdir(path_dir)


def _run_compile_cmd(cmd_line: str, cwd: str) -> bytes:
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


def run_compile_cmd(cmd_line: str, cwd: str) -> bytes:
    with dynamo_timed("compile_file"):
        return _run_compile_cmd(cmd_line, cwd)


def normalize_path_separator(orig_path: str) -> str:
    if _IS_WINDOWS:
        return orig_path.replace(os.sep, "/")
    return orig_path


class BuildOptionsBase:
    """
    This is the Base class for store cxx build options, as a template.
    Acturally, to build a cxx shared library. We just need to select a compiler
    and maintains the suitable args.
    """

    def __init__(
        self,
        compiler: str = "",
        definitions: Optional[List[str]] = None,
        include_dirs: Optional[List[str]] = None,
        cflags: Optional[List[str]] = None,
        ldflags: Optional[List[str]] = None,
        libraries_dirs: Optional[List[str]] = None,
        libraries: Optional[List[str]] = None,
        passthrough_args: Optional[List[str]] = None,
        aot_mode: bool = False,
        use_absolute_path: bool = False,
        compile_only: bool = False,
    ) -> None:
        self._compiler = compiler
        self._definations: List[str] = definitions or []
        self._include_dirs: List[str] = include_dirs or []
        self._cflags: List[str] = cflags or []
        self._ldflags: List[str] = ldflags or []
        self._libraries_dirs: List[str] = libraries_dirs or []
        self._libraries: List[str] = libraries or []
        # Some args is hard to abstract to OS compatable, passthrough it directly.
        self._passthrough_args: List[str] = passthrough_args or []

        self._aot_mode: bool = aot_mode
        self._use_absolute_path: bool = use_absolute_path
        self._compile_only: bool = compile_only

    def _process_compile_only_options(self) -> None:
        if self._compile_only:
            self._libraries_dirs = []
            self._libraries = []

    def _remove_duplicate_options(self) -> None:
        self._definations = _remove_duplication_in_list(self._definations)
        self._include_dirs = _remove_duplication_in_list(self._include_dirs)
        self._cflags = _remove_duplication_in_list(self._cflags)
        self._ldflags = _remove_duplication_in_list(self._ldflags)
        self._libraries_dirs = _remove_duplication_in_list(self._libraries_dirs)
        self._libraries = _remove_duplication_in_list(self._libraries)
        self._passthrough_args = _remove_duplication_in_list(self._passthrough_args)

    def _finalize_options(self) -> None:
        self._process_compile_only_options()
        self._remove_duplicate_options()

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

    def get_passthrough_args(self) -> List[str]:
        return self._passthrough_args

    def get_aot_mode(self) -> bool:
        return self._aot_mode

    def get_use_absolute_path(self) -> bool:
        return self._use_absolute_path

    def get_compile_only(self) -> bool:
        return self._compile_only

    def save_flags_to_file(self, file: str) -> None:
        attrs = {
            "compiler": self.get_compiler(),
            "definitions": self.get_definations(),
            "include_dirs": self.get_include_dirs(),
            "cflags": self.get_cflags(),
            "ldflags": self.get_ldflags(),
            "libraries_dirs": self.get_libraries_dirs(),
            "libraries": self.get_libraries(),
            "passthrough_args": self.get_passthrough_args(),
            "aot_mode": self.get_aot_mode(),
            "use_absolute_path": self.get_use_absolute_path(),
            "compile_only": self.get_compile_only(),
        }

        with open(file, "w") as f:
            json.dump(attrs, f)


def _get_warning_all_cflag(warning_all: bool = True) -> List[str]:
    if not _IS_WINDOWS:
        return ["Wall"] if warning_all else []
    else:
        return []


def _get_cpp_std_cflag(std_num: str = "c++17") -> List[str]:
    if _IS_WINDOWS:
        """
        On Windows, only c++20 can support `std::enable_if_t`.
        Ref: https://learn.microsoft.com/en-us/cpp/overview/cpp-conformance-improvements-2019?view=msvc-170#checking-for-abstract-class-types # noqa: B950
        Note:
            Only setup c++20 for Windows inductor. I tried to upgrade all project to c++20, but it is failed:
            https://github.com/pytorch/pytorch/pull/131504
        """
        std_num = "c++20"
        return [f"std:{std_num}"]
    else:
        return [f"std={std_num}"]


def _get_os_related_cpp_cflags(cpp_compiler: str) -> List[str]:
    if _IS_WINDOWS:
        cflags = [
            "wd4819",
            "wd4251",
            "wd4244",
            "wd4267",
            "wd4275",
            "wd4018",
            "wd4190",
            "wd4624",
            "wd4067",
            "wd4068",
            "EHsc",
        ]
    else:
        cflags = ["Wno-unused-variable", "Wno-unknown-pragmas"]
        if _is_clang(cpp_compiler):
            ignored_optimization_argument = (
                "Werror=ignored-optimization-argument"
                if config.aot_inductor.raise_error_on_ignored_optimization
                else "Wno-ignored-optimization-argument"
            )
            cflags.append(ignored_optimization_argument)
    return cflags


def _get_ffast_math_flags() -> List[str]:
    # ffast-math is equivalent to these flags as in
    # https://github.com/gcc-mirror/gcc/blob/4700ad1c78ccd7767f846802fca148b2ea9a1852/gcc/opts.cc#L3458-L3468
    # however gcc<13 sets the FTZ/DAZ flags for runtime on x86 even if we have
    # -ffast-math -fno-unsafe-math-optimizations because the flags for runtime
    # are added by linking in crtfastmath.o. This is done by the spec file which
    # only does globbing for -ffast-math.
    flags = [
        "fno-trapping-math",
        "funsafe-math-optimizations",
        "ffinite-math-only",
        "fno-signed-zeros",
        "fno-math-errno",
    ]

    if is_gcc():
        flags.append("fexcess-precision=fast")

    return flags


def _get_optimization_cflags(cpp_compiler: str) -> List[str]:
    if _IS_WINDOWS:
        return ["O2"]
    else:
        cflags = ["O0", "g"] if config.aot_inductor.debug_compile else ["O3", "DNDEBUG"]
        cflags += _get_ffast_math_flags()
        cflags.append("fno-finite-math-only")

        if not config.cpp.enable_unsafe_math_opt_flag:
            cflags.append("fno-unsafe-math-optimizations")
        cflags.append(f"ffp-contract={config.cpp.enable_floating_point_contract_flag}")

        if sys.platform != "darwin":
            # on macos, unknown argument: '-fno-tree-loop-vectorize'
            if _is_gcc(cpp_compiler):
                cflags.append("fno-tree-loop-vectorize")
            # https://stackoverflow.com/questions/65966969/why-does-march-native-not-work-on-apple-m1
            # `-march=native` is unrecognized option on M1
            if not config.is_fbcode():
                if platform.machine() == "ppc64le":
                    cflags.append("mcpu=native")
                else:
                    cflags.append("march=native")

        return cflags


def _get_shared_cflag(compile_only: bool) -> List[str]:
    if _IS_WINDOWS:
        """
        MSVC `/MD` using python `ucrtbase.dll` lib as runtime.
        https://learn.microsoft.com/en-us/cpp/c-runtime-library/crt-library-features?view=msvc-170
        """
        SHARED_FLAG = ["DLL", "MD"]
    else:
        if compile_only:
            return ["fPIC"]
        if platform.system() == "Darwin" and "clang" in get_cpp_compiler():
            # This causes undefined symbols to behave the same as linux
            return ["shared", "fPIC", "undefined dynamic_lookup"]
        else:
            return ["shared", "fPIC"]

    return SHARED_FLAG


def get_cpp_options(
    cpp_compiler: str,
    compile_only: bool,
    warning_all: bool = True,
    extra_flags: Sequence[str] = (),
) -> Tuple[List[str], List[str], List[str], List[str], List[str], List[str], List[str]]:
    definations: List[str] = []
    include_dirs: List[str] = []
    cflags: List[str] = []
    ldflags: List[str] = []
    libraries_dirs: List[str] = []
    libraries: List[str] = []
    passthrough_args: List[str] = []

    cflags = (
        _get_shared_cflag(compile_only)
        + _get_optimization_cflags(cpp_compiler)
        + _get_warning_all_cflag(warning_all)
        + _get_cpp_std_cflag()
        + _get_os_related_cpp_cflags(cpp_compiler)
    )

    passthrough_args.append(" ".join(extra_flags))

    return (
        definations,
        include_dirs,
        cflags,
        ldflags,
        libraries_dirs,
        libraries,
        passthrough_args,
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
        compile_only: bool = False,
        warning_all: bool = True,
        extra_flags: Sequence[str] = (),
        use_absolute_path: bool = False,
        compiler: str = "",
    ) -> None:
        super().__init__()
        self._compiler = compiler if compiler else get_cpp_compiler()
        self._use_absolute_path = use_absolute_path
        self._compile_only = compile_only

        (
            definations,
            include_dirs,
            cflags,
            ldflags,
            libraries_dirs,
            libraries,
            passthrough_args,
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
        _append_list(self._passthrough_args, passthrough_args)
        self._finalize_options()


def _get_glibcxx_abi_build_flags() -> List[str]:
    if not _IS_WINDOWS:
        return ["-D_GLIBCXX_USE_CXX11_ABI=" + str(int(torch._C._GLIBCXX_USE_CXX11_ABI))]
    else:
        return []


def _get_torch_cpp_wrapper_defination() -> List[str]:
    return ["TORCH_INDUCTOR_CPP_WRAPPER", "STANDALONE_TORCH_HEADER"]


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
            return fb_internal_macros
        else:
            return []
    else:
        return []


def _setup_standard_sys_libs(
    cpp_compiler: str,
    aot_mode: bool,
    use_absolute_path: bool,
) -> Tuple[List[str], List[str], List[str]]:
    from torch._inductor.codecache import _LINKER_SCRIPT

    cflags: List[str] = []
    include_dirs: List[str] = []
    passthrough_args: List[str] = []
    if _IS_WINDOWS:
        return cflags, include_dirs, passthrough_args

    if config.is_fbcode():
        # TODO(T203137008) Can we unify these flags with triton_cc_command?
        cflags.append("nostdinc")
        # Note that the order of include paths do matter, as a result
        # we need to have several branches interleaved here
        include_dirs.append(build_paths.sleef_include)
        include_dirs.append(build_paths.openmp_include)
        include_dirs.append(build_paths.python_include)
        include_dirs.append(build_paths.cc_include)
        include_dirs.append(build_paths.libgcc_include)
        include_dirs.append(build_paths.libgcc_arch_include)
        include_dirs.append(build_paths.libgcc_backward_include)
        include_dirs.append(build_paths.glibc_include)
        include_dirs.append(build_paths.linux_kernel_include)
        include_dirs.append("include")

        if aot_mode and not use_absolute_path:
            linker_script = _LINKER_SCRIPT
        else:
            linker_script = os.path.basename(_LINKER_SCRIPT)

        if _is_clang(cpp_compiler):
            passthrough_args.append(" --rtlib=compiler-rt")
            passthrough_args.append(" -fuse-ld=lld")
            passthrough_args.append(f" -Wl,--script={linker_script}")
            passthrough_args.append(" -B" + build_paths.glibc_lib)
            passthrough_args.append(" -L" + build_paths.glibc_lib)

    return cflags, include_dirs, passthrough_args


def _get_build_args_of_chosen_isa(vec_isa: VecISA) -> Tuple[List[str], List[str]]:
    macros: List[str] = []
    build_flags: List[str] = []
    if vec_isa != invalid_vec_isa:
        # Add Windows support later.
        macros.extend(copy.deepcopy(x) for x in vec_isa.build_macro())

        build_flags = [vec_isa.build_arch_flags()]

        if config.is_fbcode():
            cap = str(vec_isa).upper()
            macros = [
                f"CPU_CAPABILITY={cap}",
                f"CPU_CAPABILITY_{cap}",
                f"HAVE_{cap}_CPU_DEFINITION",
            ]

    return macros, build_flags


def _get_torch_related_args(
    include_pytorch: bool, aot_mode: bool
) -> Tuple[List[str], List[str], List[str]]:
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
    if sys.platform != "darwin" and not config.is_fbcode():
        libraries = ["torch", "torch_cpu"]
        if not aot_mode:
            libraries.append("torch_python")

    if _IS_WINDOWS and platform.machine().lower() != "arm64":
        libraries.append("sleef")

    return include_dirs, libraries_dirs, libraries


def _get_python_include_dirs() -> List[str]:
    include_dir = Path(sysconfig.get_path("include"))
    # On Darwin Python executable from a framework can return
    # non-existing /Library/Python/... include path, in which case
    # one should use Headers folder from the framework
    if not include_dir.exists() and platform.system() == "Darwin":
        std_lib = Path(sysconfig.get_path("stdlib"))
        include_dir = (std_lib.parent.parent / "Headers").absolute()
    if not (include_dir / "Python.h").exists():
        warnings.warn(f"Can't find Python.h in {str(include_dir)}")
    return [str(include_dir)]


def _get_python_related_args() -> Tuple[List[str], List[str]]:
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
        python_include_dirs.append(build_paths.python_include)

    return python_include_dirs, python_lib_path


@functools.lru_cache(None)
def is_conda_llvm_openmp_installed() -> bool:
    try:
        command = "conda list llvm-openmp --json"
        output = subprocess.check_output(command.split()).decode("utf8")
        return len(json.loads(output)) > 0
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


@functools.lru_cache(None)
def homebrew_libomp() -> Tuple[bool, str]:
    try:
        # check if `brew` is installed
        subprocess.check_output(["which", "brew"])
        # get the location of `libomp` if it is installed
        # this is the location that `libomp` **would** be installed
        # see https://github.com/Homebrew/brew/issues/10261#issuecomment-756563567 for details
        libomp_path = (
            subprocess.check_output(["brew", "--prefix", "libomp"])
            .decode("utf8")
            .strip()
        )
        # check if `libomp` is installed
        omp_available = os.path.exists(libomp_path)
        return omp_available, libomp_path
    except subprocess.SubprocessError:
        return False, ""


@functools.lru_cache(None)
def perload_clang_libomp_win(cpp_compiler: str, omp_name: str) -> None:
    try:
        output = subprocess.check_output([cpp_compiler, "-print-file-name=bin"]).decode(
            "utf8"
        )
        omp_path = os.path.join(output.rstrip(), omp_name)
        if os.path.isfile(omp_path):
            os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
            cdll.LoadLibrary(omp_path)
    except subprocess.SubprocessError:
        pass


@functools.lru_cache(None)
def perload_icx_libomp_win(cpp_compiler: str) -> None:
    def _load_icx_built_in_lib_by_name(cpp_compiler: str, lib_name: str) -> bool:
        try:
            output = subprocess.check_output(
                [cpp_compiler, f"-print-file-name={lib_name}"],
                stderr=subprocess.DEVNULL,
            ).decode(*SUBPROCESS_DECODE_ARGS)
            omp_path = output.rstrip()
            if os.path.isfile(omp_path):
                os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
                cdll.LoadLibrary(omp_path)
                return True
        except subprocess.SubprocessError:
            pass
        return False

    """
    Intel Compiler implenmented more math libraries than clang, for performance proposal.
    We need preload them like openmp library.
    """
    preload_list = [
        "libiomp5md.dll",  # openmp
        "svml_dispmd.dll",  # svml library
        "libmmd.dll",  # libm
    ]

    for lib_name in preload_list:
        _load_icx_built_in_lib_by_name(cpp_compiler, lib_name)


def _get_openmp_args(
    cpp_compiler: str,
) -> Tuple[List[str], List[str], List[str], List[str], List[str], List[str]]:
    cflags: List[str] = []
    ldflags: List[str] = []
    include_dir_paths: List[str] = []
    lib_dir_paths: List[str] = []
    libs: List[str] = []
    passthrough_args: List[str] = []
    if _IS_MACOS:
        # Per https://mac.r-project.org/openmp/ right way to pass `openmp` flags to MacOS is via `-Xclang`
        cflags.append("Xclang")
        cflags.append("fopenmp")

        # only Apple builtin compilers (Apple Clang++) require openmp
        omp_available = not _is_apple_clang(cpp_compiler)

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
        """
        On Windows, `clang` and `icx` have their specific openmp implenmention.
        And the openmp lib is in compiler's some sub-directory.
        For dynamic library(DLL) load, the Windows native APIs are `LoadLibraryA` and `LoadLibraryExA`, and their search
        dependencies have some rules:
        https://learn.microsoft.com/en-us/windows/win32/api/libloaderapi/nf-libloaderapi-loadlibraryexa#searching-for-dlls-and-dependencies
        In some case, the rules may not include compiler's sub-directories.
        So, it can't search and load compiler's openmp library correctly.
        And then, the whole application would be broken.

        To avoid the openmp load failed, we can automatic locate the openmp binary and preload it.
        1. For clang, the function is `perload_clang_libomp_win`.
        2. For icx, the function is `perload_icx_libomp_win`.
        """
        if _is_clang(cpp_compiler):
            cflags.append("openmp")
            libs.append("libomp")
            perload_clang_libomp_win(cpp_compiler, "libomp.dll")
        elif _is_intel_compiler(cpp_compiler):
            cflags.append("Qiopenmp")
            libs.append("libiomp5md")
            perload_icx_libomp_win(cpp_compiler)
        else:
            # /openmp, /openmp:llvm
            # llvm on Windows, new openmp: https://devblogs.microsoft.com/cppblog/msvc-openmp-update/
            # msvc openmp: https://learn.microsoft.com/zh-cn/cpp/build/reference/openmp-enable-openmp-2-0-support?view=msvc-170
            cflags.append("openmp")
            cflags.append("openmp:experimental")  # MSVC CL
    else:
        if config.is_fbcode():
            include_dir_paths.append(build_paths.openmp_include)

            openmp_lib = build_paths.openmp_lib_so
            fb_openmp_extra_flags = f"-Wp,-fopenmp {openmp_lib}"
            passthrough_args.append(fb_openmp_extra_flags)

            libs.append("omp")
        else:
            if _is_clang(cpp_compiler):
                # TODO: fix issue, can't find omp.h
                cflags.append("fopenmp")
                libs.append("gomp")
            elif _is_intel_compiler(cpp_compiler):
                cflags.append("fiopenmp")
            else:
                cflags.append("fopenmp")
                libs.append("gomp")

    return cflags, ldflags, include_dir_paths, lib_dir_paths, libs, passthrough_args


def get_mmap_self_macro(use_mmap_weights: bool) -> List[str]:
    macros = []
    if use_mmap_weights:
        macros.append(" USE_MMAP_SELF")
    return macros


def get_cpp_torch_options(
    cpp_compiler: str,
    vec_isa: VecISA,
    include_pytorch: bool,
    aot_mode: bool,
    compile_only: bool,
    use_absolute_path: bool,
    use_mmap_weights: bool,
) -> Tuple[List[str], List[str], List[str], List[str], List[str], List[str], List[str]]:
    definations: List[str] = []
    include_dirs: List[str] = []
    cflags: List[str] = []
    ldflags: List[str] = []
    libraries_dirs: List[str] = []
    libraries: List[str] = []
    passthrough_args: List[str] = []

    torch_cpp_wrapper_definations = _get_torch_cpp_wrapper_defination()
    use_custom_generated_macros_definations = _use_custom_generated_macros()

    (
        sys_libs_cflags,
        sys_libs_include_dirs,
        sys_libs_passthrough_args,
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
        omp_passthrough_args,
    ) = _get_openmp_args(cpp_compiler)

    cxx_abi_passthrough_args = _get_glibcxx_abi_build_flags()
    fb_macro_passthrough_args = _use_fb_internal_macros()

    mmap_self_macros = get_mmap_self_macro(use_mmap_weights)

    definations = (
        torch_cpp_wrapper_definations
        + use_custom_generated_macros_definations
        + isa_macros
        + fb_macro_passthrough_args
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
    passthrough_args = (
        sys_libs_passthrough_args
        + isa_ps_args_build_flags
        + cxx_abi_passthrough_args
        + omp_passthrough_args
    )

    return (
        definations,
        include_dirs,
        cflags,
        ldflags,
        libraries_dirs,
        libraries,
        passthrough_args,
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
        vec_isa: VecISA = invalid_vec_isa,
        include_pytorch: bool = False,
        warning_all: bool = True,
        aot_mode: bool = False,
        compile_only: bool = False,
        use_absolute_path: bool = False,
        use_mmap_weights: bool = False,
        shared: bool = True,
        extra_flags: Sequence[str] = (),
        compiler: str = "",
    ) -> None:
        super().__init__(
            compile_only=compile_only,
            warning_all=warning_all,
            extra_flags=extra_flags,
            use_absolute_path=use_absolute_path,
            compiler=compiler,
        )

        self._aot_mode = aot_mode

        (
            torch_definations,
            torch_include_dirs,
            torch_cflags,
            torch_ldflags,
            torch_libraries_dirs,
            torch_libraries,
            torch_passthrough_args,
        ) = get_cpp_torch_options(
            cpp_compiler=self._compiler,
            vec_isa=vec_isa,
            include_pytorch=include_pytorch,
            aot_mode=aot_mode,
            compile_only=compile_only,
            use_absolute_path=use_absolute_path,
            use_mmap_weights=use_mmap_weights,
        )

        _append_list(self._definations, torch_definations)
        _append_list(self._include_dirs, torch_include_dirs)
        _append_list(self._cflags, torch_cflags)
        _append_list(self._ldflags, torch_ldflags)
        _append_list(self._libraries_dirs, torch_libraries_dirs)
        _append_list(self._libraries, torch_libraries)
        _append_list(self._passthrough_args, torch_passthrough_args)
        self._finalize_options()


def _set_gpu_runtime_env() -> None:
    if (
        config.is_fbcode()
        and torch.version.hip is None
        and "CUDA_HOME" not in os.environ
        and "CUDA_PATH" not in os.environ
    ):
        os.environ["CUDA_HOME"] = build_paths.sdk_home


def _transform_cuda_paths(lpaths: List[str]) -> None:
    # This handles two cases:
    # 1. Cases where libs are in (e.g.) lib/cuda-12 and lib/cuda-12/stubs
    # 2. Linux machines may have CUDA installed under either lib64/ or lib/
    for i, path in enumerate(lpaths):
        if "CUDA_HOME" in os.environ and path.startswith(os.environ["CUDA_HOME"]):
            try:
                lib_dir = next(Path(path).rglob("libcudart_static.a")).resolve().parent
            except StopIteration:
                log_msg = f'"libcudart_static.a" not found under {path}'
                log.info(log_msg)
                continue

            lpaths[i] = str(lib_dir)
            stub_dir = lib_dir / "stubs"
            if stub_dir.exists():
                lpaths.append(str(stub_dir))


def get_cpp_torch_device_options(
    device_type: str,
    aot_mode: bool = False,
    compile_only: bool = False,
) -> Tuple[List[str], List[str], List[str], List[str], List[str], List[str], List[str]]:
    definations: List[str] = []
    include_dirs: List[str] = []
    cflags: List[str] = []
    ldflags: List[str] = []
    libraries_dirs: List[str] = []
    libraries: List[str] = []
    passthrough_args: List[str] = []
    if (
        config.is_fbcode()
        and "CUDA_HOME" not in os.environ
        and "CUDA_PATH" not in os.environ
    ):
        os.environ["CUDA_HOME"] = build_paths.sdk_home

    _set_gpu_runtime_env()
    from torch.utils import cpp_extension

    include_dirs = cpp_extension.include_paths(device_type)
    libraries_dirs = cpp_extension.library_paths(device_type)
    if device_type == "cuda":
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
                libraries += ["c10_cuda", "cuda", "torch_cuda"]
            _transform_cuda_paths(libraries_dirs)

    if device_type == "xpu":
        definations.append(" USE_XPU")
        # Add "-Wno-unsupported-floating-point-opt" here to
        # suppress compiler warning:
        # "warning: overriding currently unsupported use of floating point
        # exceptions on this target [-Wunsupported-floating-point-opt]".
        # Since the compiler has not support some features.
        cflags += ["fsycl", "Wno-unsupported-floating-point-opt"]
        libraries += ["c10_xpu", "sycl", "ze_loader", "torch_xpu"]

    if aot_mode:
        if config.is_fbcode():
            from torch._inductor.codecache import cpp_prefix_path

            cpp_prefix_include_dir = [f"{os.path.dirname(cpp_prefix_path())}"]
            include_dirs += cpp_prefix_include_dir

    if config.is_fbcode():
        include_dirs.append(build_paths.sdk_include)

        if aot_mode and device_type == "cuda":
            if torch.version.hip is None:
                if not compile_only:
                    # Only add link args, when compile_only is false.
                    passthrough_args = ["-Wl,-Bstatic -lcudart_static -Wl,-Bdynamic"]

    return (
        definations,
        include_dirs,
        cflags,
        ldflags,
        libraries_dirs,
        libraries,
        passthrough_args,
    )


class CppTorchDeviceOptions(CppTorchOptions):
    """
    This class is inherited from CppTorchOptions, which automatic contains
    base cxx build options and torch common build options. And then it will
    maintains cuda/xpu device related build args.
    """

    def __init__(
        self,
        vec_isa: VecISA = invalid_vec_isa,
        include_pytorch: bool = False,
        device_type: str = "cuda",
        aot_mode: bool = False,
        compile_only: bool = False,
        use_absolute_path: bool = False,
        use_mmap_weights: bool = False,
        shared: bool = True,
        extra_flags: Sequence[str] = (),
    ) -> None:
        if device_type == "xpu":
            from torch.utils.cpp_extension import _join_sycl_home

            compiler = _join_sycl_home("bin", "icpx")
        else:
            compiler = ""
        super().__init__(
            vec_isa=vec_isa,
            include_pytorch=include_pytorch,
            aot_mode=aot_mode,
            compile_only=compile_only,
            use_absolute_path=use_absolute_path,
            use_mmap_weights=use_mmap_weights,
            extra_flags=extra_flags,
            compiler=compiler,
        )

        device_definations: List[str] = []
        device_include_dirs: List[str] = []
        device_cflags: List[str] = []
        device_ldflags: List[str] = []
        device_libraries_dirs: List[str] = []
        device_libraries: List[str] = []
        device_passthrough_args: List[str] = []

        (
            device_definations,
            device_include_dirs,
            device_cflags,
            device_ldflags,
            device_libraries_dirs,
            device_libraries,
            device_passthrough_args,
        ) = get_cpp_torch_device_options(
            device_type=device_type, aot_mode=aot_mode, compile_only=compile_only
        )
        _append_list(self._definations, device_definations)
        _append_list(self._include_dirs, device_include_dirs)
        _append_list(self._cflags, device_cflags)
        _append_list(self._ldflags, device_ldflags)
        _append_list(self._libraries_dirs, device_libraries_dirs)
        _append_list(self._libraries, device_libraries)
        _append_list(self._passthrough_args, device_passthrough_args)
        self._finalize_options()

    def _finalize_options(self) -> None:
        super()._finalize_options()
        if config.is_fbcode():
            # Re-order library search paths in case there are lib conflicts
            # that also live in the FBCode python lib dir.
            _, python_lib_dirs = _get_python_related_args()
            assert len(python_lib_dirs) == 1, f"Python lib dirs: {python_lib_dirs}"
            if python_lib_dirs[0] in self._libraries_dirs:
                self._libraries_dirs.remove(python_lib_dirs[0])
                self._libraries_dirs.append(python_lib_dirs[0])


def get_name_and_dir_from_output_file_path(
    file_path: str,
) -> Tuple[str, str]:
    """
    This function help prepare parameters to new cpp_builder.
    Example:
        input_code: /tmp/tmpof1n5g7t/5c/c5crkkcdvhdxpktrmjxbqkqyq5hmxpqsfza4pxcf3mwk42lphygc.cpp
        name, dir = get_name_and_dir_from_output_file_path(input_code)
    Run result:
        name = c5crkkcdvhdxpktrmjxbqkqyq5hmxpqsfza4pxcf3mwk42lphygc
        dir = /tmp/tmpof1n5g7t/5c/

    put 'name' and 'dir' to CppBuilder's 'name' and 'output_dir'.
    CppBuilder --> get_target_file_path will format output path accoding OS:
    Linux: /tmp/tmppu87g3mm/zh/czhwiz4z7ca7ep3qkxenxerfjxy42kehw6h5cjk6ven4qu4hql4i.so
    Windows: [Windows temp path]/tmppu87g3mm/zh/czhwiz4z7ca7ep3qkxenxerfjxy42kehw6h5cjk6ven4qu4hql4i.dll
    """
    name_and_ext = os.path.basename(file_path)
    name, _ext = os.path.splitext(name_and_ext)
    dir = os.path.dirname(file_path)

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

    def __get_python_module_ext(self) -> str:
        SHARED_LIB_EXT = ".pyd" if _IS_WINDOWS else ".so"
        return SHARED_LIB_EXT

    def __get_object_ext(self) -> str:
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
        self._passthrough_parameters_args = ""

        self._output_dir = ""
        self._target_file = ""

        self._use_absolute_path: bool = False
        self._aot_mode: bool = False

        self._name = name

        # Code start here, initial self internal veriables firstly.
        self._compiler = BuildOption.get_compiler()
        self._use_absolute_path = BuildOption.get_use_absolute_path()
        self._aot_mode = BuildOption.get_aot_mode()

        self._output_dir = output_dir

        self._compile_only = BuildOption.get_compile_only()
        file_ext = (
            self.__get_object_ext()
            if self._compile_only
            else self.__get_python_module_ext()
        )
        self._target_file = os.path.join(self._output_dir, f"{self._name}{file_ext}")

        if isinstance(sources, str):
            sources = [sources]

        if config.is_fbcode():
            if self._aot_mode and not self._use_absolute_path:
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

        for passthrough_arg in BuildOption.get_passthrough_args():
            self._passthrough_parameters_args += f"{passthrough_arg} "

    def get_command_line(self) -> str:
        def format_build_command(
            compiler: str,
            sources: str,
            include_dirs_args: str,
            definations_args: str,
            cflags_args: str,
            ldflags_args: str,
            libraries_args: str,
            libraries_dirs_args: str,
            passthrough_args: str,
            target_file: str,
        ) -> str:
            if _IS_WINDOWS:
                # https://learn.microsoft.com/en-us/cpp/build/walkthrough-compile-a-c-program-on-the-command-line?view=msvc-1704
                # https://stackoverflow.com/a/31566153
                cmd = (
                    f"{compiler} {include_dirs_args} {definations_args} {cflags_args} {sources} "
                    f"{passthrough_args} /LD /Fe{target_file} /link {libraries_dirs_args} {libraries_args} {ldflags_args} "
                )
                cmd = normalize_path_separator(cmd)
            else:
                compile_only_arg = "-c" if self._compile_only else ""
                cmd = re.sub(
                    r"[ \n]+",
                    " ",
                    f"""
                    {compiler} {sources} {definations_args} {cflags_args} {include_dirs_args}
                    {passthrough_args} {ldflags_args} {libraries_args} {libraries_dirs_args} {compile_only_arg} -o {target_file}
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
            passthrough_args=self._passthrough_parameters_args,
            target_file=self._target_file,
        )
        return command_line

    def get_target_file_path(self) -> str:
        return normalize_path_separator(self._target_file)

    def build(self) -> Tuple[bytes, str]:
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

        status = run_compile_cmd(build_cmd, cwd=_build_tmp_dir)

        _remove_dir(_build_tmp_dir)
        return status, self._target_file
