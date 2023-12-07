import errno
import functools
import os
import platform
import re
import shlex
import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import List, Tuple

import torch
from torch._inductor import config

if config.is_fbcode():
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


def _get_cxx_compiler():
    if _IS_WINDOWS:
        compiler = os.environ.get("CXX", "cl")
    else:
        compiler = os.environ.get("CXX", "c++")
    return compiler


def _nonduplicate_append(dest_list: List[str], src_list: List[str]):
    for item in src_list:
        if item not in dest_list:
            dest_list.append(item)


def _create_if_dir_not_exist(path_dir):
    if not os.path.exists(path_dir):
        try:
            Path(path_dir).mkdir(parents=True, exist_ok=True)
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise RuntimeError(f"Fail to create path {path_dir}")


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
    status = subprocess.call(cmd, cwd=cwd, stderr=subprocess.STDOUT)

    return status


def is_gcc(cpp_compiler) -> bool:
    return bool(re.search(r"(gcc|g\+\+)", cpp_compiler))


def is_clang(cpp_compiler) -> bool:
    return bool(re.search(r"(clang|clang\+\+)", cpp_compiler))


@functools.lru_cache(None)
def is_apple_clang(cpp_compiler) -> bool:
    version_string = subprocess.check_output([cpp_compiler, "--version"]).decode("utf8")
    return "Apple" in version_string.splitlines()[0]


class BuildOptionsBase:
    """
    This is the Base class for store cxx build options, as a template.
    Acturally, to build a cxx shared library. We just need to select a compiler
    and maintains the suitable args.
    """

    _compiler = ""
    _definations: List[str] = []
    _include_dirs: List[str] = []
    _cflags: List[str] = []
    _ldlags: List[str] = []
    _libraries_dirs: List[str] = []
    _libraries: List[str] = []
    _passthough_args: List[str] = []

    def __init__(self) -> None:
        pass

    def get_compiler(self) -> str:
        return self._compiler

    def get_definations(self) -> List[str]:
        return self._definations

    def get_include_dirs(self) -> List[str]:
        return self._include_dirs

    def get_cflags(self) -> List[str]:
        return self._cflags

    def get_ldlags(self) -> List[str]:
        return self._ldlags

    def get_libraries_dirs(self) -> List[str]:
        return self._libraries_dirs

    def get_libraries(self) -> List[str]:
        return self._libraries

    def get_passthough_args(self) -> List[str]:
        return self._passthough_args


def get_warning_all_flag(warning_all: bool = False) -> List[str]:
    if not _IS_WINDOWS:
        return ["Wall"] if warning_all else []
    else:
        return []


def get_cxx_std(std_num: str = "c++17") -> List[str]:
    if _IS_WINDOWS:
        return [f"std:{std_num}"]
    else:
        return [f"std={std_num}"]


def get_linux_cpp_cflags(cpp_compiler) -> List[str]:
    if not _IS_WINDOWS:
        cflags = ["Wno-unused-variable", "Wno-unknown-pragmas"]
        if is_clang(cpp_compiler):
            cflags.append("Werror=ignored-optimization-argument")
        return cflags
    else:
        return []


def optimization_cflags() -> List[str]:
    if _IS_WINDOWS:
        return ["O2"]
    else:
        cflags = ["O0", "g"] if config.aot_inductor.debug_compile else ["O3", "DNDEBUG"]
        cflags.append("ffast-math")
        cflags.append("fno-finite-math-only")

        if not config.cpp.enable_unsafe_math_opt_flag:
            cflags.append("fno-unsafe-math-optimizations")

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


class CxxOptions(BuildOptionsBase):
    """
    This class is inherited from BuildOptionsBase, and as cxx build options.
    This option need contains basic cxx build option, which contains:
    1. OS related args.
    2. Toolchains related args.
    3. Cxx standard related args.
    Note:
    1. According to the base class __init__ function would be called when each
    child class instances created. We need use _nonduplicate_append to avoid
    duplicate args append.
    2. This Options is good for assist modules build, such as x86_isa_help.
    """

    def _get_shared_cflag(self) -> List[str]:
        SHARED_FLAG = ["DLL"] if _IS_WINDOWS else ["shared", "fPIC"]
        return SHARED_FLAG

    def __init__(self) -> None:
        super().__init__()
        self._compiler = _get_cxx_compiler()
        _nonduplicate_append(self._cflags, self._get_shared_cflag())

        _nonduplicate_append(self._cflags, optimization_cflags())
        _nonduplicate_append(self._cflags, get_warning_all_flag())
        _nonduplicate_append(self._cflags, get_cxx_std())

        _nonduplicate_append(self._cflags, get_linux_cpp_cflags(self._compiler))


def get_glibcxx_abi_build_flags() -> List[str]:
    return ["-D_GLIBCXX_USE_CXX11_ABI=" + str(int(torch._C._GLIBCXX_USE_CXX11_ABI))]


def get_torch_cpp_wrapper_defination() -> List[str]:
    return ["TORCH_INDUCTOR_CPP_WRAPPER"]


def use_custom_generated_macros() -> List[str]:
    return [" C10_USING_CUSTOM_GENERATED_MACROS"]


def use_fb_internal_macros() -> List[str]:
    if not _IS_WINDOWS:
        if config.is_fbcode():
            # openmp_lib = build_paths.openmp_lib()
            preprocessor_flags = " ".join(
                (
                    "-D C10_USE_GLOG",
                    "-D C10_USE_MINIMAL_GLOG",
                    "-D C10_DISABLE_TENSORIMPL_EXTENSIBILITY",
                )
            )
            # return [f"-Wp,-fopenmp {openmp_lib} {preprocessor_flags}"]
            return [f"{preprocessor_flags}"]
        else:
            return []
    else:
        return []


def use_standard_sys_dir_headers() -> List[str]:
    if _IS_WINDOWS:
        return []

    if config.is_fbcode():
        return ["nostdinc"]
    else:
        return []


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


def get_build_args_of_chosen_isa():
    from torch._inductor.codecache import chosen_isa

    cap = str(chosen_isa).upper()
    macros = [
        f"CPU_CAPABILITY={cap}",
        f"CPU_CAPABILITY_{cap}",
        f"HAVE_{cap}_CPU_DEFINITION",
    ]
    # Add Windows support later.
    build_flags = [chosen_isa.build_arch_flags()]

    return macros, build_flags


def get_torch_related_args():
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
    libraries = ["torch", "torch_cpu", "c10"]
    return include_dirs, libraries_dirs, libraries


def get_python_related_args():
    python_include_dirs = []
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

    return python_include_dirs, python_lib_path


"""
def get_openmp_args(cpp_compiler):
    ipaths = []
    lpaths = []
    if _IS_MACOS:
        from torch._inductor.codecache import (
            homebrew_libomp,
            is_conda_llvm_openmp_installed,
        )

        # only Apple builtin compilers (Apple Clang++) require openmp
        omp_available = not is_apple_clang(cpp_compiler)

        # check the `OMP_PREFIX` environment first
        if os.getenv("OMP_PREFIX") is not None:
            header_path = os.path.join(os.getenv("OMP_PREFIX"), "include", "omp.h")
            valid_env = os.path.exists(header_path)
            if valid_env:
                ipaths.append(os.path.join(os.getenv("OMP_PREFIX"), "include"))
                lpaths.append(os.path.join(os.getenv("OMP_PREFIX"), "lib"))
            else:
                warnings.warn("environment variable `OMP_PREFIX` is invalid.")
            omp_available = omp_available or valid_env

        libs = [] if omp_available else ["omp"]

        # prefer to use openmp from `conda install llvm-openmp`
        if not omp_available and os.getenv("CONDA_PREFIX") is not None:
            omp_available = is_conda_llvm_openmp_installed()
            if omp_available:
                conda_lib_path = os.path.join(os.getenv("CONDA_PREFIX"), "lib")
                ipaths.append(os.path.join(os.getenv("CONDA_PREFIX"), "include"))
                lpaths.append(conda_lib_path)
                # Prefer Intel OpenMP on x86 machine
                if os.uname().machine == "x86_64" and os.path.exists(
                    os.path.join(conda_lib_path, "libiomp5.dylib")
                ):
                    libs = ["iomp5"]

        # next, try to use openmp from `brew install libomp`
        if not omp_available:
            omp_available, libomp_path = homebrew_libomp()
            if omp_available:
                ipaths.append(os.path.join(libomp_path, "include"))
                lpaths.append(os.path.join(libomp_path, "lib"))

        # if openmp is still not available, we let the compiler to have a try,
        # and raise error together with instructions at compilation error later
    elif _IS_WINDOWS:
        todo = 0
    else:
        libs = ["omp"] if config.is_fbcode() else ["gomp"]

    return
"""


class CxxTorchOptions(CxxOptions):
    """
    This class is inherited from CxxTorchOptions, which automatic contains
    base cxx build options. And then it will maintains torch related build
    args.
    1. Torch include_directories, libraries, libraries_directories.
    2. Python include_directories, libraries, libraries_directories.
    3. OpenMP related.
    4. Torch MACROs.
    5. MISC
    """

    def __init__(self) -> None:
        super().__init__()
        _nonduplicate_append(self._definations, get_torch_cpp_wrapper_defination())
        _nonduplicate_append(self._definations, use_custom_generated_macros())

        _nonduplicate_append(self._cflags, use_standard_sys_dir_headers())

        macros, build_flags = get_build_args_of_chosen_isa()
        _nonduplicate_append(self._definations, macros)
        _nonduplicate_append(self._passthough_args, build_flags)

        (
            torch_include_dirs,
            torch_libraries_dirs,
            torch_libraries,
        ) = get_torch_related_args()
        _nonduplicate_append(self._include_dirs, torch_include_dirs)
        _nonduplicate_append(self._libraries_dirs, torch_libraries_dirs)
        _nonduplicate_append(self._libraries, torch_libraries)

        python_include_dirs, python_libraries_dirs = get_python_related_args()
        _nonduplicate_append(self._include_dirs, python_include_dirs)
        _nonduplicate_append(self._libraries_dirs, python_libraries_dirs)

        # cpp_prefix_dir = [f"{os.path.dirname(_cpp_prefix_path())}"]
        # _nonduplicate_append(self._include_dirs, cpp_prefix_dir)

        if not _IS_WINDOWS:
            # glibcxx is not available in Windows.
            _nonduplicate_append(self._passthough_args, get_glibcxx_abi_build_flags())
            _nonduplicate_append(self._passthough_args, use_fb_internal_macros())


class CxxTorchCudaOptions(CxxTorchOptions):
    """
    This class is inherited from CxxTorchOptions, which automatic contains
    base cxx build options and torch common build options. And then it will
    maintains cuda device related build args.
    """

    def __init__(self) -> None:
        super().__init__()
        # _nonduplicate_append(self._cflags, ["DCUDA"])


class CxxBuilder:
    _compiler = ""
    _cflags_args = ""
    _definations_args = ""
    _include_dirs_args = ""
    _ldlags_args = ""
    _libraries_dirs_args = ""
    _libraries_args = ""
    _passthough_parameters_args = ""

    _name = ""
    _sources_args = ""
    _output_dir = ""
    _target_file = ""

    def get_shared_lib_ext(self) -> str:
        SHARED_LIB_EXT = ".dll" if _IS_WINDOWS else ".so"
        return SHARED_LIB_EXT

    def __init__(
        self,
        name: str,
        sources: List[str],
        BuildOption: BuildOptionsBase,
        output_dir: str = "",
    ) -> None:
        self._name = name
        self._sources_args = " ".join(sources)

        if output_dir is None:
            self._output_dir = os.path.dirname(os.path.abspath(__file__))
        else:
            self._output_dir = output_dir

        self._target_file = os.path.join(
            self._output_dir, f"{self._name}{self.get_shared_lib_ext()}"
        )

        self._compiler = BuildOption.get_compiler()

        for cflag in BuildOption.get_cflags():
            if _IS_WINDOWS:
                self._cflags_args += f"/{cflag} "
            else:
                self._cflags_args += f"-{cflag} "

        for defination in BuildOption.get_definations():
            if _IS_WINDOWS:
                self._definations_args += f"/D {defination} "
            else:
                self._definations_args += f"-D{defination} "

        for inc_dir in BuildOption.get_include_dirs():
            if _IS_WINDOWS:
                self._include_dirs_args += f"/I {inc_dir} "
            else:
                self._include_dirs_args += f"-I{inc_dir} "

        for ldflag in BuildOption.get_ldlags():
            if _IS_WINDOWS:
                self._ldlags_args += f"/{ldflag} "
            else:
                self._ldlags_args += f"-{ldflag} "

        for lib_dir in BuildOption.get_libraries_dirs():
            if _IS_WINDOWS:
                self._libraries_dirs_args += f"/LIBPATH:{lib_dir} "
            else:
                self._libraries_dirs_args += f"-L{lib_dir} "

        for lib in BuildOption.get_libraries():
            if _IS_WINDOWS:
                self._libraries_args += f"lib{lib}.lib "
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
                    f"{compiler} {include_dirs_args} {definations_args} {cflags_args} {sources} {ldflags_args} "
                    f"{libraries_args} {libraries_dirs_args} {passthougn_args} /LD /Fe{target_file}"
                )
                cmd = cmd.replace("\\", "\\\\")
            else:
                cmd = (
                    f"{compiler} {sources} {include_dirs_args} {definations_args} {cflags_args} {ldflags_args} "
                    f"{libraries_args} {libraries_dirs_args} {passthougn_args} -o {target_file}"
                )
            return cmd

        command_line = format_build_command(
            compiler=self._compiler,
            sources=self._sources_args,
            include_dirs_args=self._include_dirs_args,
            definations_args=self._definations_args,
            cflags_args=self._cflags_args,
            ldflags_args=self._ldlags_args,
            libraries_args=self._libraries_args,
            libraries_dirs_args=self._libraries_dirs_args,
            passthougn_args=self._passthough_parameters_args,
            target_file=self._target_file,
        )
        return command_line

    def build(self) -> Tuple[int, str]:
        """
        It is must need a temperary directory to store object files in Windows.
        """
        _create_if_dir_not_exist(self._output_dir)
        _build_tmp_dir = os.path.join(
            self._output_dir, f"{self._name}_{_BUILD_TEMP_DIR}"
        )
        _create_if_dir_not_exist(_build_tmp_dir)

        build_cmd = self.get_command_line()
        print("!!! build_cmd: ", build_cmd)
        status = run_command_line(build_cmd, cwd=_build_tmp_dir)

        _remove_dir(_build_tmp_dir)
        return status, self._target_file
