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
import tempfile
import textwrap
import warnings
from collections.abc import Sequence
from ctypes import cdll
from ctypes.util import find_library
from pathlib import Path
from typing import Any, Optional, Union

import torch
from torch._dynamo.utils import dynamo_timed
from torch._inductor import config, exc
from torch._inductor.cpu_vec_isa import invalid_vec_isa, VecISA
from torch._inductor.runtime.runtime_utils import cache_dir
from torch.torch_version import TorchVersion


if config.is_fbcode():
    from triton.fb.build import _run_build_command, build_paths

    from torch._inductor.fb.utils import (
        log_global_cache_errors,
        log_global_cache_stats,
        log_global_cache_vals,
        use_global_cache,
    )
else:

    def log_global_cache_errors(*args: Any, **kwargs: Any) -> None:  # type: ignore[misc]
        pass

    def log_global_cache_stats(*args: Any, **kwargs: Any) -> None:  # type: ignore[misc]
        pass

    def log_global_cache_vals(*args: Any, **kwargs: Any) -> None:  # type: ignore[misc]
        pass

    def use_global_cache() -> bool:  # type: ignore[misc]
        return False


# Windows need setup a temp dir to store .obj files.
_BUILD_TEMP_DIR = "CxxBuild"
_HERE = os.path.abspath(__file__)
_TORCH_PATH = os.path.dirname(os.path.dirname(_HERE))
_LINKER_SCRIPT = os.path.join(_TORCH_PATH, "_inductor/script.ld")

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


@functools.cache
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


def get_ld_and_objcopy(use_relative_path: bool) -> tuple[str, str]:
    if _IS_WINDOWS:
        raise RuntimeError("Windows is not supported yet.")
    else:
        if config.is_fbcode():
            ld = build_paths.ld
            objcopy = (
                build_paths.objcopy_fallback
                if use_relative_path
                else build_paths.objcopy
            )
        else:
            ld = "ld"
            objcopy = "objcopy"
    return ld, objcopy


def convert_cubin_to_obj(
    cubin_file: str,
    kernel_name: str,
    ld: str,
    objcopy: str,
) -> str:
    obj_file = cubin_file + ".o"
    # Convert .cubin to .o
    cmd = f"{ld} -r -b binary -z noexecstack -o {obj_file} {cubin_file}"
    subprocess.run(cmd.split(), capture_output=True, text=True, check=True)
    # Rename .data to .rodata
    cmd = f"{objcopy} --rename-section .data=.rodata,alloc,load,readonly,data,contents {obj_file}"
    subprocess.run(cmd.split(), capture_output=True, text=True, check=True)
    # By default objcopy will create *_start, *_size, *_end symbols using the full path
    # Rename to use the unique kernel name
    file_name = re.sub(r"[\W]", "_", cubin_file)
    cmd = (
        objcopy
        + f" --redefine-sym _binary_{file_name}_start=__{kernel_name}_start "
        + f"--redefine-sym _binary_{file_name}_size=__{kernel_name}_size "
        + f"--redefine-sym _binary_{file_name}_end=__{kernel_name}_end "
        + obj_file
    )
    subprocess.run(cmd.split(), capture_output=True, text=True, check=True)
    return obj_file


@functools.cache
def _is_apple_clang(cpp_compiler: str) -> bool:
    version_string = subprocess.check_output([cpp_compiler, "--version"]).decode("utf8")
    return "Apple" in version_string.splitlines()[0]


@functools.cache
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


@functools.cache
def _is_gcc(cpp_compiler: str) -> bool:
    # Since "clang++" ends with "g++", the regex match below would validate on it.
    if _is_clang(cpp_compiler):
        return False
    return bool(re.search(r"(gcc|g\+\+|gnu-c\+\+)", cpp_compiler))


@functools.cache
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


@functools.cache
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


@functools.cache
def is_gcc() -> bool:
    return _is_gcc(get_cpp_compiler())


@functools.cache
def is_clang() -> bool:
    return _is_clang(get_cpp_compiler())


@functools.cache
def is_intel_compiler() -> bool:
    return _is_intel_compiler(get_cpp_compiler())


@functools.cache
def is_apple_clang() -> bool:
    return _is_apple_clang(get_cpp_compiler())


@functools.cache
def is_msvc_cl() -> bool:
    return _is_msvc_cl(get_cpp_compiler())


@functools.cache
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
    # Multiple lines to one line string.
    version_string = version_string.replace("\r", "_")
    version_string = version_string.replace("\n", "_")
    return version_string


# =============================== cpp builder ===============================
def _append_list(dest_list: list[str], src_list: list[str]) -> None:
    dest_list.extend(copy.deepcopy(item) for item in src_list)


def _remove_duplication_in_list(orig_list: list[str]) -> list[str]:
    new_list: list[str] = []
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


def _run_compile_cmd(cmd_line: str, cwd: str) -> None:
    cmd = shlex.split(cmd_line)
    try:
        subprocess.run(
            cmd, cwd=cwd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
    except subprocess.CalledProcessError as e:
        output = e.stdout.decode("utf-8")
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


def run_compile_cmd(cmd_line: str, cwd: str) -> None:
    with dynamo_timed("compile_file"):
        _run_compile_cmd(cmd_line, cwd)


def normalize_path_separator(orig_path: str) -> str:
    if _IS_WINDOWS:
        return orig_path.replace(os.sep, "/")
    return orig_path


class BuildOptionsBase:
    """
    This is the Base class for store cxx build options, as a template.
    Actually, to build a cxx shared library. We just need to select a compiler
    and maintains the suitable args.
    """

    def __init__(
        self,
        compiler: str = "",
        definitions: Optional[list[str]] = None,
        include_dirs: Optional[list[str]] = None,
        cflags: Optional[list[str]] = None,
        ldflags: Optional[list[str]] = None,
        libraries_dirs: Optional[list[str]] = None,
        libraries: Optional[list[str]] = None,
        passthrough_args: Optional[list[str]] = None,
        aot_mode: bool = False,
        use_relative_path: bool = False,
        compile_only: bool = False,
        precompiling: bool = False,
        preprocessing: bool = False,
    ) -> None:
        self._compiler = compiler
        self._definitions: list[str] = definitions or []
        self._include_dirs: list[str] = include_dirs or []
        self._cflags: list[str] = cflags or []
        self._ldflags: list[str] = ldflags or []
        self._libraries_dirs: list[str] = libraries_dirs or []
        self._libraries: list[str] = libraries or []
        # Some args are hard to abstract to OS compatible, passthrough directly.
        self._passthrough_args: list[str] = passthrough_args or []

        # Optionally, the path to a precompiled header which should be included on the
        # build command line.
        self.precompiled_header: Optional[str] = None

        self._aot_mode: bool = aot_mode
        self._use_relative_path: bool = use_relative_path
        self._compile_only: bool = compile_only
        self._precompiling: bool = precompiling
        self._preprocessing: bool = preprocessing

    def _process_compile_only_options(self) -> None:
        if self._compile_only:
            self._libraries_dirs = []
            self._libraries = []

    def _remove_duplicate_options(self) -> None:
        self._definitions = _remove_duplication_in_list(self._definitions)
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

    def get_definitions(self) -> list[str]:
        return self._definitions

    def get_include_dirs(self) -> list[str]:
        return self._include_dirs

    def get_cflags(self) -> list[str]:
        return self._cflags

    def get_ldflags(self) -> list[str]:
        return self._ldflags

    def get_libraries_dirs(self) -> list[str]:
        return self._libraries_dirs

    def get_libraries(self) -> list[str]:
        return self._libraries

    def get_passthrough_args(self) -> list[str]:
        return self._passthrough_args

    def get_aot_mode(self) -> bool:
        return self._aot_mode

    def get_use_relative_path(self) -> bool:
        return self._use_relative_path

    def get_compile_only(self) -> bool:
        return self._compile_only

    def get_precompiling(self) -> bool:
        return self._precompiling

    def get_preprocessing(self) -> bool:
        return self._preprocessing

    def save_flags_to_json(self, file: str) -> None:
        attrs = {
            "compiler": self.get_compiler(),
            "definitions": self.get_definitions(),
            "include_dirs": self.get_include_dirs(),
            "cflags": self.get_cflags(),
            "ldflags": self.get_ldflags(),
            "libraries_dirs": self.get_libraries_dirs(),
            "libraries": self.get_libraries(),
            "passthrough_args": self.get_passthrough_args(),
            "aot_mode": self.get_aot_mode(),
            "use_relative_path": self.get_use_relative_path(),
            "compile_only": self.get_compile_only(),
        }

        with open(file, "w") as f:
            json.dump(attrs, f)


def _get_warning_all_cflag(warning_all: bool = True) -> list[str]:
    if not _IS_WINDOWS:
        return ["Wall"] if warning_all else []
    else:
        return []


def _get_cpp_std_cflag(std_num: str = "c++17") -> list[str]:
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


def _get_os_related_cpp_cflags(cpp_compiler: str) -> list[str]:
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


def _get_ffast_math_flags() -> list[str]:
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


def _get_optimization_cflags(
    cpp_compiler: str, min_optimize: bool = False
) -> list[str]:
    if _IS_WINDOWS:
        return ["O1" if min_optimize else "O2"]
    else:
        wrapper_opt_level = config.aot_inductor.compile_wrapper_opt_level
        cflags = (
            ["O0", "g"]
            if config.aot_inductor.debug_compile
            else [wrapper_opt_level if min_optimize else "O3", "DNDEBUG"]
        )
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


def _get_shared_cflag(do_link: bool) -> list[str]:
    if _IS_WINDOWS:
        """
        MSVC `/MD` using python `ucrtbase.dll` lib as runtime.
        https://learn.microsoft.com/en-us/cpp/c-runtime-library/crt-library-features?view=msvc-170
        """
        return ["DLL", "MD"]
    if not do_link:
        return ["fPIC"]
    if platform.system() == "Darwin" and "clang" in get_cpp_compiler():
        # This causes undefined symbols to behave the same as linux
        return ["shared", "fPIC", "undefined dynamic_lookup"]
    return ["shared", "fPIC"]


def get_cpp_options(
    cpp_compiler: str,
    do_link: bool,
    warning_all: bool = True,
    extra_flags: Sequence[str] = (),
    min_optimize: bool = False,
) -> tuple[list[str], list[str], list[str], list[str], list[str], list[str], list[str]]:
    definitions: list[str] = []
    include_dirs: list[str] = []
    cflags: list[str] = []
    ldflags: list[str] = []
    libraries_dirs: list[str] = []
    libraries: list[str] = []
    passthrough_args: list[str] = []

    cflags = (
        _get_shared_cflag(do_link)
        + _get_optimization_cflags(cpp_compiler, min_optimize)
        + _get_warning_all_cflag(warning_all)
        + _get_cpp_std_cflag()
        + _get_os_related_cpp_cflags(cpp_compiler)
    )

    passthrough_args.append(" ".join(extra_flags))

    return (
        definitions,
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
        use_relative_path: bool = False,
        compiler: str = "",
        min_optimize: bool = False,
        precompiling: bool = False,
        preprocessing: bool = False,
    ) -> None:
        super().__init__(
            compile_only=compile_only,
            use_relative_path=use_relative_path,
            precompiling=precompiling,
            preprocessing=preprocessing,
        )
        self._compiler = compiler if compiler else get_cpp_compiler()

        (
            definitions,
            include_dirs,
            cflags,
            ldflags,
            libraries_dirs,
            libraries,
            passthrough_args,
        ) = get_cpp_options(
            cpp_compiler=self._compiler,
            do_link=not (compile_only or precompiling or preprocessing),
            extra_flags=extra_flags,
            warning_all=warning_all,
            min_optimize=min_optimize,
        )

        _append_list(self._definitions, definitions)
        _append_list(self._include_dirs, include_dirs)
        _append_list(self._cflags, cflags)
        _append_list(self._ldflags, ldflags)
        _append_list(self._libraries_dirs, libraries_dirs)
        _append_list(self._libraries, libraries)
        _append_list(self._passthrough_args, passthrough_args)
        self._finalize_options()


def _get_glibcxx_abi_build_flags() -> list[str]:
    if not _IS_WINDOWS:
        return ["-D_GLIBCXX_USE_CXX11_ABI=" + str(int(torch._C._GLIBCXX_USE_CXX11_ABI))]
    else:
        return []


def _get_torch_cpp_wrapper_definition() -> list[str]:
    return ["TORCH_INDUCTOR_CPP_WRAPPER", "STANDALONE_TORCH_HEADER"]


def _use_custom_generated_macros() -> list[str]:
    return [" C10_USING_CUSTOM_GENERATED_MACROS"]


def _use_fb_internal_macros() -> list[str]:
    if not _IS_WINDOWS:
        if config.is_fbcode():
            fb_internal_macros = [
                "C10_USE_GLOG",
                "C10_USE_MINIMAL_GLOG",
                "C10_DISABLE_TENSORIMPL_EXTENSIBILITY",
            ]
            return fb_internal_macros
        else:
            return []
    else:
        return []


def _setup_standard_sys_libs(
    cpp_compiler: str,
    aot_mode: bool,
    use_relative_path: bool,
) -> tuple[list[str], list[str], list[str]]:
    cflags: list[str] = []
    include_dirs: list[str] = []
    passthrough_args: list[str] = []
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

        if aot_mode and not use_relative_path:
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


def _get_build_args_of_chosen_isa(vec_isa: VecISA) -> tuple[list[str], list[str]]:
    macros: list[str] = []
    build_flags: list[str] = []
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
) -> tuple[list[str], list[str], list[str]]:
    from torch.utils.cpp_extension import include_paths, TORCH_LIB_PATH

    include_dirs = include_paths()
    libraries_dirs = [TORCH_LIB_PATH]
    libraries = []
    if sys.platform != "darwin" and not config.is_fbcode():
        libraries = ["torch", "torch_cpu"]
        if not aot_mode:
            libraries.append("torch_python")

    if _IS_WINDOWS:
        libraries.append("sleef")

    return include_dirs, libraries_dirs, libraries


def _get_python_include_dirs() -> list[str]:
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


def _get_python_related_args() -> tuple[list[str], list[str]]:
    python_include_dirs = _get_python_include_dirs()
    python_include_path = sysconfig.get_path(
        "include", scheme="nt" if _IS_WINDOWS else "posix_prefix"
    )
    if python_include_path is not None:
        python_include_dirs.append(python_include_path)

    if _IS_WINDOWS:
        python_lib_path = [
            str(
                (
                    Path(sysconfig.get_path("include", scheme="nt")).parent / "libs"
                ).absolute()
            )
        ]
    else:
        python_lib_path = [sysconfig.get_config_var("LIBDIR")]

    if config.is_fbcode():
        python_include_dirs.append(build_paths.python_include)

    return python_include_dirs, python_lib_path


@functools.cache
def is_conda_llvm_openmp_installed() -> bool:
    try:
        command = "conda list llvm-openmp --json"
        output = subprocess.check_output(command.split()).decode("utf8")
        return len(json.loads(output)) > 0
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


@functools.cache
def homebrew_libomp() -> tuple[bool, str]:
    try:
        # check if `brew` is installed
        if shutil.which("brew") is None:
            return False, ""
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


@functools.cache
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


@functools.cache
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
    Intel Compiler implemented more math libraries than clang, for performance proposal.
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
) -> tuple[list[str], list[str], list[str], list[str], list[str], list[str]]:
    cflags: list[str] = []
    ldflags: list[str] = []
    include_dir_paths: list[str] = []
    lib_dir_paths: list[str] = []
    libs: list[str] = []
    passthrough_args: list[str] = []
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


def get_mmap_self_macro(use_mmap_weights: bool) -> list[str]:
    macros = []
    if use_mmap_weights:
        macros.append(" USE_MMAP_SELF")
    return macros


def get_cpp_torch_options(
    cpp_compiler: str,
    vec_isa: VecISA,
    include_pytorch: bool,
    aot_mode: bool,
    use_relative_path: bool,
    use_mmap_weights: bool,
) -> tuple[list[str], list[str], list[str], list[str], list[str], list[str], list[str]]:
    definitions: list[str] = []
    include_dirs: list[str] = []
    cflags: list[str] = []
    ldflags: list[str] = []
    libraries_dirs: list[str] = []
    libraries: list[str] = []
    passthrough_args: list[str] = []

    torch_cpp_wrapper_definitions = _get_torch_cpp_wrapper_definition()
    use_custom_generated_macros_definitions = _use_custom_generated_macros()

    (
        sys_libs_cflags,
        sys_libs_include_dirs,
        sys_libs_passthrough_args,
    ) = _setup_standard_sys_libs(cpp_compiler, aot_mode, use_relative_path)

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

    definitions = (
        torch_cpp_wrapper_definitions
        + use_custom_generated_macros_definitions
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
        definitions,
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
        use_relative_path: bool = False,
        use_mmap_weights: bool = False,
        shared: bool = True,
        extra_flags: Sequence[str] = (),
        compiler: str = "",
        min_optimize: bool = False,
        precompiling: bool = False,
        preprocessing: bool = False,
    ) -> None:
        super().__init__(
            compile_only=compile_only,
            warning_all=warning_all,
            extra_flags=extra_flags,
            use_relative_path=use_relative_path,
            compiler=compiler,
            min_optimize=min_optimize,
            precompiling=precompiling,
            preprocessing=preprocessing,
        )

        self._aot_mode = aot_mode

        (
            torch_definitions,
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
            use_relative_path=use_relative_path,
            use_mmap_weights=use_mmap_weights,
        )

        _append_list(self._definitions, torch_definitions)
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


@functools.lru_cache(8)
def _find_libcudart_static(path: str) -> Optional[Path]:
    lib_dirs = list(Path(path).rglob("libcudart_static.a"))
    if lib_dirs:
        return lib_dirs[0].resolve().parent
    log_msg = f'"libcudart_static.a" not found under {path}'
    log.info(log_msg)
    return None


def _transform_cuda_paths(lpaths: list[str]) -> None:
    # This handles two cases:
    # 1. Cases where libs are in (e.g.) lib/cuda-12 and lib/cuda-12/stubs
    # 2. Linux machines may have CUDA installed under either lib64/ or lib/
    for i, path in enumerate(lpaths):
        if "CUDA_HOME" in os.environ and path.startswith(os.environ["CUDA_HOME"]):
            lib_dir: Optional[Path] = _find_libcudart_static(path)
            if lib_dir is None:
                continue
            lpaths[i] = str(lib_dir)
            stub_dir = lib_dir / "stubs"
            if stub_dir.exists():
                lpaths.append(str(stub_dir))


def get_cpp_torch_device_options(
    device_type: str,
    aot_mode: bool = False,
    compile_only: bool = False,
) -> tuple[list[str], list[str], list[str], list[str], list[str], list[str], list[str]]:
    definitions: list[str] = []
    include_dirs: list[str] = []
    cflags: list[str] = []
    ldflags: list[str] = []
    libraries_dirs: list[str] = []
    libraries: list[str] = []
    passthrough_args: list[str] = []
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
        definitions.append(" USE_ROCM" if torch.version.hip else " USE_CUDA")

        if torch.version.hip is not None:
            if config.is_fbcode():
                libraries += ["amdhip64"]
            else:
                libraries += ["c10_hip", "torch_hip"]
            definitions.append(" __HIP_PLATFORM_AMD__")
        else:
            if config.is_fbcode():
                libraries += ["cuda"]
            else:
                libraries += ["c10_cuda", "cuda", "torch_cuda"]
            _transform_cuda_paths(libraries_dirs)

    if device_type == "xpu":
        definitions.append(" USE_XPU")
        # Suppress multi-line comment warnings in sycl headers
        cflags += ["Wno-comment"]
        libraries += ["c10_xpu", "sycl", "ze_loader", "torch_xpu"]
        if not find_library("ze_loader"):
            raise OSError(
                "Intel GPU driver is not properly installed, please follow the instruction "
                "in https://github.com/pytorch/pytorch?tab=readme-ov-file#intel-gpu-support."
            )

    if device_type == "mps":
        definitions.append(" USE_MPS")

    if config.is_fbcode():
        include_dirs.append(build_paths.sdk_include)

        if aot_mode and device_type == "cuda":
            if torch.version.hip is None:
                if not compile_only:
                    # Only add link args, when compile_only is false.
                    passthrough_args = ["-Wl,-Bstatic -lcudart_static -Wl,-Bdynamic"]

    if config.aot_inductor.custom_op_libs:
        libraries += config.aot_inductor.custom_op_libs

    return (
        definitions,
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
        use_relative_path: bool = False,
        use_mmap_weights: bool = False,
        shared: bool = True,
        extra_flags: Sequence[str] = (),
        min_optimize: bool = False,
        precompiling: bool = False,
        preprocessing: bool = False,
    ) -> None:
        super().__init__(
            vec_isa=vec_isa,
            include_pytorch=include_pytorch,
            aot_mode=aot_mode,
            compile_only=compile_only,
            use_relative_path=use_relative_path,
            use_mmap_weights=use_mmap_weights,
            extra_flags=extra_flags,
            min_optimize=min_optimize,
            precompiling=precompiling,
            preprocessing=preprocessing,
        )

        device_definitions: list[str] = []
        device_include_dirs: list[str] = []
        device_cflags: list[str] = []
        device_ldflags: list[str] = []
        device_libraries_dirs: list[str] = []
        device_libraries: list[str] = []
        device_passthrough_args: list[str] = []

        (
            device_definitions,
            device_include_dirs,
            device_cflags,
            device_ldflags,
            device_libraries_dirs,
            device_libraries,
            device_passthrough_args,
        ) = get_cpp_torch_device_options(
            device_type=device_type, aot_mode=aot_mode, compile_only=compile_only
        )
        _append_list(self._definitions, device_definitions)
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
) -> tuple[str, str]:
    """
    This function help prepare parameters to new cpp_builder.
    Example:
        input_code: /tmp/tmpof1n5g7t/5c/c5crkkcdvhdxpktrmjxbqkqyq5hmxpqsfza4pxcf3mwk42lphygc.cpp
        name, dir = get_name_and_dir_from_output_file_path(input_code)
    Run result:
        name = c5crkkcdvhdxpktrmjxbqkqyq5hmxpqsfza4pxcf3mwk42lphygc
        dir = /tmp/tmpof1n5g7t/5c/

    put 'name' and 'dir' to CppBuilder's 'name' and 'output_dir'.
    CppBuilder --> get_target_file_path will format output path according OS:
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
            2. Due to the CppBuilder is supports multiple OS, it will maintains ext for OS difference.
        sources:
            Source code file list to be built.
        BuildOption:
            Build options to the builder.
        output_dir:
            1. The output_dir the target file will output to.
            2. The default value is empty string, and then the use current dir as output dir.
            3. Final target file: output_dir/name.ext
    """

    @staticmethod
    def __get_python_module_flags() -> tuple[str, str]:
        extension = ".pyd" if _IS_WINDOWS else ".so"
        output_flags = "/Fe" if _IS_WINDOWS else "-o"
        return extension, output_flags

    @staticmethod
    def __get_object_flags() -> tuple[str, str]:
        extension = ".obj" if _IS_WINDOWS else ".o"
        output_flags = "/c /Fo" if _IS_WINDOWS else "-c -o"  # codespell:ignore
        return extension, output_flags

    @staticmethod
    def __get_precompiled_header_flags() -> tuple[str, str]:
        extension = ".pch" if _IS_WINDOWS or not is_gcc() else ".gch"
        output_flags = "/Fp" if _IS_WINDOWS else "-o"
        return extension, output_flags

    @staticmethod
    def __get_preprocessor_output_flags() -> tuple[str, str]:
        extension = ".i"
        output_flags = "/EP /P" if _IS_WINDOWS else "-E -P -o"
        return extension, output_flags

    def __init__(
        self,
        name: str,
        sources: Union[str, list[str]],
        BuildOption: BuildOptionsBase,
        output_dir: str = "",
    ) -> None:
        self._compiler = ""
        self._cflags_args = ""
        self._definitions_args = ""
        self._include_dirs_args = ""
        self._ldflags_args = ""
        self._libraries_dirs_args = ""
        self._libraries_args = ""
        self._passthrough_parameters_args = ""

        # When relative path is used, we need to maintain the source dir list.
        self._orig_source_paths = []
        self._output_dir = ""
        self._target_file = ""

        self._use_relative_path: bool = False
        self._aot_mode: bool = False

        self._name = name

        # Code start here, initial self internal variables firstly.
        self._build_option = BuildOption
        self._compiler = BuildOption.get_compiler()
        self._use_relative_path = BuildOption.get_use_relative_path()
        self._aot_mode = BuildOption.get_aot_mode()

        self._output_dir = output_dir

        self._compile_only = BuildOption.get_compile_only()
        self._precompiling = BuildOption.get_precompiling()
        self._preprocessing = BuildOption.get_preprocessing()
        # Only one of these options (if any) should be true at any given time.
        assert sum((self._compile_only, self._precompiling, self._preprocessing)) <= 1
        self._do_link = not (
            self._compile_only or self._precompiling or self._preprocessing
        )

        # MSVC produces two files when precompiling: the actual .pch file, as well as an
        # object file which must be linked into the final library.  This class assumes
        # only one output file of note, so for now we'll error out here.
        assert not _IS_WINDOWS or not self._precompiling, (
            "Cannot currently precompile headers on Windows!"
        )

        if self._compile_only:
            file_ext, output_flags = self.__get_object_flags()
        elif self._precompiling:
            file_ext, output_flags = self.__get_precompiled_header_flags()
        elif self._preprocessing:
            file_ext, output_flags = self.__get_preprocessor_output_flags()
        else:
            file_ext, output_flags = self.__get_python_module_flags()
        self._target_file = os.path.join(self._output_dir, f"{self._name}{file_ext}")

        relative_target_file = (
            os.path.basename(self._target_file)
            if self._use_relative_path
            else self._target_file
        )
        if _IS_WINDOWS:
            if self._preprocessing:
                # The target file name is automatically determined by MSVC.
                self._output = output_flags
            else:
                self._output = f"{output_flags}{relative_target_file}"
        else:
            self._output = f"{output_flags} {relative_target_file}"

        if isinstance(sources, str):
            sources = [sources]

        if config.is_fbcode() and (not self._aot_mode or self._use_relative_path):
            # Will create another temp directory for building, so do NOT use the
            # absolute path.
            self._orig_source_paths = list(sources)
            sources = [os.path.basename(i) for i in sources]

        if self._precompiling:
            assert len(sources) == 1
            # See above; we can currently assume this is not on MSVC.
            self._sources_args = f"-x c++-header {sources[0]}"
        else:
            self._sources_args = " ".join(sources)

        for cflag in BuildOption.get_cflags():
            if _IS_WINDOWS:
                self._cflags_args += f"/{cflag} "
            else:
                self._cflags_args += f"-{cflag} "

        for definition in BuildOption.get_definitions():
            if _IS_WINDOWS:
                self._definitions_args += f"/D {definition} "
            else:
                self._definitions_args += f"-D {definition} "

        if precompiled_header := BuildOption.precompiled_header:
            if _IS_WINDOWS:
                log.warning(
                    "Precompiled header support for MSVC is currently unavailable; ignoring %s",
                    precompiled_header,
                )
            else:
                self._include_dirs_args = f"-include {precompiled_header} "

        for inc_dir in BuildOption.get_include_dirs():
            if _IS_WINDOWS:
                self._include_dirs_args += f'/I "{inc_dir}" '
            else:
                self._include_dirs_args += f"-I{shlex.quote(inc_dir)} "

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
            definitions_args: str,
            cflags_args: str,
            ldflags_args: str,
            libraries_args: str,
            libraries_dirs_args: str,
            passthrough_args: str,
            output: str,
        ) -> str:
            if _IS_WINDOWS:
                # https://learn.microsoft.com/en-us/cpp/build/walkthrough-compile-a-c-program-on-the-command-line?view=msvc-1704
                # https://stackoverflow.com/a/31566153
                cmd = (
                    f"{compiler} {include_dirs_args} {definitions_args} {cflags_args} "
                    f"{sources} {passthrough_args} {output}"
                )
                if self._do_link:
                    cmd += f" /LD /link {libraries_dirs_args} {libraries_args} {ldflags_args}"
                cmd = normalize_path_separator(cmd)
            else:
                cmd = (
                    f"{compiler} {sources} {definitions_args} {cflags_args} "
                    f"{include_dirs_args} {passthrough_args} {output}"
                )
                if self._do_link:
                    cmd += f" {ldflags_args} {libraries_args} {libraries_dirs_args}"
            return cmd

        command_line = format_build_command(
            compiler=self._compiler,
            sources=self._sources_args,
            include_dirs_args=self._include_dirs_args,
            definitions_args=self._definitions_args,
            cflags_args=self._cflags_args,
            ldflags_args=self._ldflags_args,
            libraries_args=self._libraries_args,
            libraries_dirs_args=self._libraries_dirs_args,
            passthrough_args=self._passthrough_parameters_args,
            output=self._output,
        )
        return command_line

    def get_target_file_path(self) -> str:
        return normalize_path_separator(self._target_file)

    def build_fbcode_re(
        self,
    ) -> None:
        with dynamo_timed("compile_file"):
            command = self.get_command_line().split()
            try:
                output_path = self._target_file
                # When we build remotely, we need to make sure to carefully copy any files
                # that are required during the compilation process into our build directly.
                # This is where all of the ATen/c10/Torch includes come from.
                torch_includes_path = os.path.join(_TORCH_PATH, "include")
                with tempfile.TemporaryDirectory() as tmp_dir:
                    # Copy everything to tmp compilation folder
                    shutil.copy(_LINKER_SCRIPT, os.path.join(tmp_dir, "script.ld"))
                    for src in self._orig_source_paths:
                        shutil.copy(src, os.path.join(tmp_dir, os.path.basename(src)))
                    dest_include_path = os.path.join(tmp_dir, "include")
                    shutil.copytree(torch_includes_path, dest_include_path)
                    # Run the build
                    tmp_output_path = _run_build_command(
                        command, tmp_dir, os.path.basename(output_path)
                    )
                    # Copy output from the build
                    if os.path.exists(output_path):
                        os.remove(output_path)
                    shutil.copy(tmp_output_path, output_path)
                    if output_path.endswith(".o"):
                        os.chmod(output_path, 0o644)
                    elif output_path.endswith(".so"):
                        os.chmod(output_path, 0o755)
            except subprocess.CalledProcessError as e:
                output = e.output.decode("utf-8")
                raise exc.CppCompileError(command, output) from e

    def build(self) -> None:
        """
        It is must need a temporary directory to store object files in Windows.
        After build completed, delete the temporary directory to save disk space.
        """
        if self._use_relative_path:
            # remote build uses relative path
            return self.build_fbcode_re()
        _create_if_dir_not_exist(self._output_dir)
        _build_tmp_dir = os.path.join(
            self._output_dir, f"{self._name}_{_BUILD_TEMP_DIR}"
        )
        _create_if_dir_not_exist(_build_tmp_dir)

        build_cmd = self.get_command_line()
        run_compile_cmd(build_cmd, cwd=_build_tmp_dir)
        _remove_dir(_build_tmp_dir)

    def save_compile_cmd_to_cmake(
        self,
        cmake_path: str,
        device_type: str,
    ) -> None:
        """
        Save global cmake settings here, e.g. compiler options.
        If targeting CUDA, also emit a custom function to embed CUDA kernels.
        """

        definitions = " ".join(self._build_option.get_definitions())
        contents = textwrap.dedent(
            f"""
            cmake_minimum_required(VERSION 3.27 FATAL_ERROR)
            project(aoti_model LANGUAGES CXX)
            set(CMAKE_CXX_STANDARD 17)

            # May need to point CMAKE_PREFIX_PATH to the right torch location
            find_package(Torch REQUIRED)

            # Set a shared library target
            add_library(aoti_model SHARED)

            # Add macro definitions
            target_compile_definitions(aoti_model PRIVATE {definitions})

            # Add compile flags
            target_compile_options(aoti_model PRIVATE {self._cflags_args})
            # Backend specific flags
            target_compile_options(aoti_model PRIVATE {self._passthrough_parameters_args} -c)

            """
        )
        if device_type == "cuda" and torch.version.hip is None:
            from torch._inductor.codecache import _nvcc_arch_as_compile_option

            current_arch = _nvcc_arch_as_compile_option()
            contents += textwrap.dedent(
                f"""
                enable_language(CUDA)
                find_package(CUDAToolkit REQUIRED)

                find_program(OBJCOPY_EXECUTABLE objcopy)
                if(NOT OBJCOPY_EXECUTABLE)
                    message(FATAL_ERROR "objcopy not found. Cannot embed fatbin as object file")
                endif()

                set(KERNEL_TARGETS "")
                set(KERNEL_OBJECT_FILES "")
                # Function to embed a single kernel
                function(embed_gpu_kernel KERNEL_NAME PTX_FILE)
                    set(FATBIN_BASENAME ${{KERNEL_NAME}}.fatbin)
                    set(FATBIN_FILE ${{CMAKE_CURRENT_BINARY_DIR}}/${{FATBIN_BASENAME}})
                    set(OBJECT_BASENAME ${{KERNEL_NAME}}.fatbin.o)
                    set(OBJECT_FILE ${{CMAKE_CURRENT_BINARY_DIR}}/${{OBJECT_BASENAME}})

                    # --- Define UNIQUE C symbol names ---
                    set(SYMBOL_START __${{KERNEL_NAME}}_start)
                    set(SYMBOL_END __${{KERNEL_NAME}}_end)
                    set(SYMBOL_SIZE __${{KERNEL_NAME}}_size)
                    string(REGEX REPLACE "[^a-zA-Z0-9]" "_" MANGLED_BASENAME ${{FATBIN_FILE}})
                    set(OBJCOPY_START_SYM _binary_${{MANGLED_BASENAME}}_start)
                    set(OBJCOPY_END_SYM _binary_${{MANGLED_BASENAME}}_end)
                    set(OBJCOPY_SIZE_SYM _binary_${{MANGLED_BASENAME}}_size)

                    # --- PTX to FATBIN Command & Target ---
                    add_custom_command(
                        OUTPUT ${{FATBIN_FILE}}
                        COMMAND ${{CUDAToolkit_NVCC_EXECUTABLE}} --fatbin ${{PTX_FILE}} -o ${{FATBIN_FILE}} ${{NVCC_GENCODE_FLAGS}}
                                -gencode arch=compute_80,code=compute_80
                                -gencode arch=compute_{current_arch},code=sm_{current_arch}
                        DEPENDS ${{PTX_FILE}}
                    )

                    # --- FATBIN to Object File (.o) Command ---
                    add_custom_command(
                        OUTPUT ${{OBJECT_FILE}}
                        COMMAND ${{CMAKE_LINKER}} -r -b binary -z noexecstack -o ${{OBJECT_FILE}} ${{FATBIN_FILE}}
                        COMMAND ${{OBJCOPY_EXECUTABLE}} --rename-section .data=.rodata,alloc,load,readonly,data,contents
                                ${{OBJECT_FILE}}
                        COMMAND ${{OBJCOPY_EXECUTABLE}}
                                --redefine-sym ${{OBJCOPY_START_SYM}}=${{SYMBOL_START}}
                                --redefine-sym ${{OBJCOPY_END_SYM}}=${{SYMBOL_END}}
                                --redefine-sym ${{OBJCOPY_SIZE_SYM}}=${{SYMBOL_SIZE}}
                                ${{OBJECT_FILE}}
                        DEPENDS ${{FATBIN_FILE}}
                    )
                    add_custom_target(build_kernel_object_${{KERNEL_NAME}} DEPENDS ${{OBJECT_FILE}})

                    # --- Add to a list for linking later ---
                    set(KERNEL_TARGETS ${{KERNEL_TARGETS}} build_kernel_object_${{KERNEL_NAME}} PARENT_SCOPE)
                    set(KERNEL_OBJECT_FILES ${{KERNEL_OBJECT_FILES}} ${{OBJECT_FILE}} PARENT_SCOPE)
                endfunction()

                """
            )

        with open(cmake_path, "w") as f:
            f.write(contents)

    def save_src_to_cmake(self, cmake_path: str, src_path: str) -> None:
        # Remove the directory part of file_path
        src_path = "${CMAKE_CURRENT_SOURCE_DIR}/" + Path(src_path).name
        with open(cmake_path, "a") as f:
            f.write(f"target_sources(aoti_model PRIVATE {src_path})\n")

    def save_kernel_asm_to_cmake(self, cmake_path: str, asm_files: list[str]) -> None:
        # TODO: make this work beyond CUDA
        with open(cmake_path, "a") as f:
            for asm_file in asm_files:
                kernel_name = Path(asm_file).name.split(".")[0]
                asm_file = f"${{CMAKE_CURRENT_SOURCE_DIR}}/{Path(asm_file).name}"
                contents = textwrap.dedent(
                    f"""
                    embed_gpu_kernel({kernel_name} {asm_file})
                    """
                )
                f.write(contents)
            f.write("add_dependencies(aoti_model ${KERNEL_TARGETS})\n")
            f.write(
                "target_link_libraries(aoti_model PRIVATE ${KERNEL_OBJECT_FILES})\n"
            )

    def save_link_cmd_to_cmake(self, cmake_path: str) -> None:
        lflags = " ".join(self._build_option.get_ldflags())
        libs = " ".join(self._build_option.get_libraries())
        contents = textwrap.dedent(
            f"""
            # Add linker flags
            target_link_options(aoti_model PRIVATE {lflags})

            # Add libraries
            target_link_libraries(aoti_model PRIVATE {libs})
         """
        )

        assert os.path.exists(cmake_path), (
            f"save_link_cmd_to_cmakefile expects {cmake_path} to already exist"
        )
        with open(cmake_path, "a") as f:
            f.write(contents)
