import errno
import os
import shlex
import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import List

from torch._inductor import config, exc

if config.is_fbcode():
    from triton.fb import build_paths
    from triton.fb.build import _run_build_command

    from torch._inductor.fb.utils import (  # type: ignore[import]
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

# initialize variables for compilation
_IS_LINUX = sys.platform.startswith("linux")
_IS_MACOS = sys.platform.startswith("darwin")
_IS_WINDOWS = sys.platform == "win32"

_BUILD_TEMP_DIR = "CxxBuild"


def _get_cxx_compiler():
    if _IS_WINDOWS:
        compiler = os.environ.get("CXX", "cl")
    else:
        compiler = os.environ.get("CXX", "c++")
    return compiler


def _create_if_dir_not_exist(path_dir):
    if not os.path.exists(path_dir):
        try:
            Path(path_dir).mkdir(parents=True, exist_ok=True)
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise RuntimeError(f"Fail to create path {path_dir}")


def get_dir_name_from_path(file_path):
    dir_name = os.path.dirname(file_path)
    return dir_name


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


def _check_if_file_exist(file_path):
    check_file = os.path.isfile(file_path)
    return check_file


def _get_file_relative_path(project_root, src_file):
    relative_path = os.path.relpath(src_file, project_root)
    if relative_path is None:
        raise RuntimeError(
            f"source file {src_file} is not belong to project {project_root}"
        )
    return relative_path


def run_command_line(cmd_line, cwd=None):
    cmd = shlex.split(cmd_line)
    status = subprocess.call(cmd, cwd=cwd, stderr=subprocess.STDOUT)

    return status


def _get_windows_runtime_libs():
    return [
        "psapi",
        "shell32",
        "user32",
        "advapi32",
        "bcrypt",
        "kernel32",
        "user32",
        "gdi32",
        "winspool",
        "shell32",
        "ole32",
        "oleaut32",
        "uuid",
        "comdlg32",
        "advapi32",
    ]


class BuildTarget:
    __name = None
    __sources = []
    __definations = []
    __include_dirs = []
    __CFLAGS = []
    __LDFLAGS = []
    __lib_dirs = []
    __libraries = []
    __output_directory = None
    __is_shared = False

    # OS info
    def is_windows(self):
        return _IS_WINDOWS

    def is_linux(self):
        return _IS_LINUX

    def is_mac_os(self):
        return _IS_MACOS

    # File types
    def __get_shared_flag(self):
        SHARED_FLAG = "DLL" if _IS_WINDOWS else "shared"
        return SHARED_FLAG

    def get_shared_lib_ext(self):
        SHARED_LIB_EXT = ".dll" if _IS_WINDOWS else ".so"
        return SHARED_LIB_EXT

    def get_exec_ext(self):
        EXEC_EXT = ".exe" if _IS_WINDOWS else ""
        return EXEC_EXT

    def __init__(self) -> None:
        pass

    # Build
    def __prepare_build_parameters(self):
        cmd_include_dirs = ""
        cmd_libraries = ""
        cmd_lib_dirs = ""
        cmd_definations = ""
        cmd_cflags = ""
        cmd_ldflags = ""

        if len(self.__include_dirs) != 0:
            for inc in self.__include_dirs:
                if _IS_WINDOWS:
                    cmd_include_dirs += f"/I {inc} "
                else:
                    cmd_include_dirs += f"-I{inc} "

        if len(self.__libraries) != 0:
            for lib in self.__libraries:
                if _IS_WINDOWS:
                    cmd_libraries += f"{lib}.lib "
                else:
                    cmd_libraries += f"-l{lib} "
                    
        if len(self.__lib_dirs) != 0:
            for lib_dir in self.__lib_dirs:
                if _IS_WINDOWS:
                    cmd_lib_dirs += f"/LIBPATH:{lib_dir} "
                else:
                    cmd_lib_dirs += f"-L{lib_dir} "

        if len(self.__definations) != 0:
            for defs in self.__definations:
                if _IS_WINDOWS:
                    cmd_definations += f"/D {defs} "
                else:
                    cmd_definations += f"-D{defs} "

        if len(self.__CFLAGS) != 0:
            for cflag in self.__CFLAGS:
                if _IS_WINDOWS:
                    cmd_cflags += f"/{cflag} "
                else:
                    cmd_cflags += f"-{cflag} "

        if len(self.__LDFLAGS) != 0:
            for ldflag in self.__LDFLAGS:
                if _IS_WINDOWS:
                    cmd_ldflags += f"/{ldflag} "
                else:
                    cmd_ldflags += f"-{ldflag} "

        return cmd_include_dirs, cmd_libraries, cmd_definations, cmd_cflags, cmd_ldflags, cmd_lib_dirs

    # Config
    def add_sources(self, sources: List[str]):
        for i in sources:
            self.__sources.append(i)
            
    def add_includes(self, includes: List[str]):
        for i in includes:
            self.__include_dirs.append(i)
            
    def add_lib_dirs(self, lib_dirs: List[str]):
        for i in lib_dirs:
            self.__lib_dirs.append(i)            

    def add_libraries(self, libraries: List[str]):
        for i in libraries:
            self.__libraries.append(i)

    def add_definations(self, definations: List[str]):
        for i in definations:
            self.__definations.append(i)

    def add_defination(self, defination: str, value: str = ""):
        define = f"{defination}={value}" if value != "" else f"{defination}"
        self.__definations.append(define)

    def add_cflags(self, cflags: List[str]):
        for i in cflags:
            self.__CFLAGS.append(i)

    def add_ldflags(self, ldflags: List[str]):
        for i in ldflags:
            self.__LDFLAGS.append(i)

    # Major
    def target(
        self,
        name: str,
        sources: List[str],
        definations: List[str] = [],
        include_dirs: List[str] = [],
        cflags: List[str] = [],
        ldflags: List[str] = [],
        libraries: List[str] = [],
        output_directory: str = None,
        is_shared: bool = True,
    ) -> bool:
        self.__name = name
        self.__sources = sources
        self.__definations = definations
        self.__include_dirs = include_dirs
        self.__CFLAGS = cflags
        self.__LDFLAGS = ldflags
        self.__libraries = libraries
        self.__output_directory = output_directory
        self.__is_shared = is_shared

    def _get_build_root_dir(self):
        if self.__output_directory is None:
            build_root = os.path.dirname(os.path.abspath(__file__))
        else:
            build_root = self.__output_directory
        _create_if_dir_not_exist(build_root)            
        return build_root

    def get_target_file_path(self):
        build_root = self._get_build_root_dir()
        if self.__is_shared:
            file_ext = self.get_shared_lib_ext()
        else:
            file_ext = self.get_exec_ext()
            
        target_file = f"{self.__name}{file_ext}"
        target_file = os.path.join(build_root, target_file)   
        return target_file

    def get_build_cmd(self):
        if self.__name is None:
            raise RuntimeError("target name should not be None.")

        if self.__is_shared:
            self.add_ldflags([self.__get_shared_flag()])

        if _IS_WINDOWS:
            self.add_libraries(_get_windows_runtime_libs())
            
        self._config_include_and_linking_paths()
        
        target_file = self.get_target_file_path()

        compiler = _get_cxx_compiler()
        (
            cmd_include_dirs,
            cmd_libraries,
            cmd_definations,
            cmd_cflags,
            cmd_ldflags,
            cmd_lib_dirs,
        ) = self.__prepare_build_parameters()

        def format_build_command(
            compiler,
            src_file,
            cmd_include_dirs,
            cmd_definations,
            cmd_cflags,
            cmd_ldflags,
            cmd_libraries,
            cmd_lib_dirs,
            target_file,
        ):
            srcs = " ".join(src_file)
            if _IS_WINDOWS:
                # https://learn.microsoft.com/en-us/cpp/build/walkthrough-compile-a-c-program-on-the-command-line?view=msvc-1704
                # https://stackoverflow.com/a/31566153
                cmd = f"{compiler} {cmd_include_dirs} {cmd_definations} {cmd_cflags} {srcs} {cmd_ldflags} {cmd_libraries} {cmd_lib_dirs} /LD /Fe{target_file}"
                cmd = cmd.replace("\\", "\\\\")
            else:
                cmd = f"{compiler} {cmd_include_dirs} {srcs} {cmd_definations} {cmd_cflags} {cmd_ldflags} {cmd_libraries} {cmd_lib_dirs} -o {target_file}"
            return cmd

        build_cmd = format_build_command(
            compiler=compiler,
            src_file=self.__sources,
            cmd_include_dirs=cmd_include_dirs,
            cmd_definations=cmd_definations,
            cmd_cflags=cmd_cflags,
            cmd_ldflags=cmd_ldflags,
            cmd_libraries=cmd_libraries,
            cmd_lib_dirs=cmd_lib_dirs,
            target_file=target_file,
        )
        return build_cmd

    def build(self):
        build_root = self._get_build_root_dir()

        # Create a temprary dir to store object files, and delete it after build complete.
        build_temp_dir = os.path.join(build_root, _BUILD_TEMP_DIR)
        _create_if_dir_not_exist(build_temp_dir)

        build_cmd = self.get_build_cmd()
        run_command_line(build_cmd, cwd=build_temp_dir)
        _remove_dir(build_temp_dir)
        
    def _config_include_and_linking_paths(self,
    include_pytorch: bool = False,
    # vec_isa: VecISA = invalid_vec_isa,
    cuda: bool = False,
    aot_mode: bool = False,
    ):
        from .codecache import cpp_prefix_path
        if (
            config.is_fbcode()
            and "CUDA_HOME" not in os.environ
            and "CUDA_PATH" not in os.environ
        ):
            os.environ["CUDA_HOME"] = os.path.dirname(build_paths.cuda())
        from torch.utils import cpp_extension

        if _IS_LINUX and (
            include_pytorch
            # or vec_isa != invalid_vec_isa
            or cuda
            or config.cpp.enable_kernel_profile
        ):
            # Note - We include pytorch only on linux right now. There is more work
            # to do to enable OMP build on darwin where PyTorch is built with IOMP
            # and we need a way to link to what PyTorch links.
            ipaths = cpp_extension.include_paths(cuda) + [sysconfig.get_path("include")]
            lpaths = cpp_extension.library_paths(cuda) + [
                sysconfig.get_config_var("LIBDIR")
            ]
            self.add_includes(ipaths)
            self.add_lib_dirs([lpaths])

            # No need to manually specify libraries in fbcode.
            if not config.is_fbcode():
                # libs += ["torch", "torch_cpu"]
                # libs += ["gomp"]
                self.add_libraries(["torch", "torch_cpu", "gomp"])
                if not aot_mode:
                    # libs += ["torch_python"]
                    self.add_libraries(["torch_python"])
            else:
                # internal remote execution is able to find omp, but not gomp
                # libs += ["omp"]
                self.add_libraries(["omp"])
                if aot_mode:
                    ipaths = [os.path.dirname(cpp_prefix_path())]
                    self.add_includes(ipaths)
                    if cuda:
                        # This is a special treatment for Meta internal cuda-12 where all libs
                        # are in lib/cuda-12 and lib/cuda-12/stubs
                        for i, path in enumerate(lpaths):
                            if path.startswith(
                                os.environ["CUDA_HOME"]
                            ) and not os.path.exists(f"{path}/libcudart_static.a"):
                                for root, dirs, files in os.walk(path):
                                    if "libcudart_static.a" in files:
                                        lpaths[i] = os.path.join(path, root)
                                        # lpaths.append(os.path.join(lpaths[i], "stubs"))
                                        self.add_lib_dirs(os.path.join(lpaths[i], "stubs"))
                                        break
            '''
            macros = vec_isa.build_macro()
            if macros:
                if config.is_fbcode() and vec_isa != invalid_vec_isa:
                    cap = str(vec_isa).upper()
                    macros = " ".join(
                        [
                            vec_isa.build_arch_flags(),
                            f"-D CPU_CAPABILITY={cap}",
                            f"-D CPU_CAPABILITY_{cap}",
                            f"-D HAVE_{cap}_CPU_DEFINITION",
                        ]
                    )            
            '''

            if aot_mode and cuda:
                '''
                if macros is None:
                    macros = ""
                macros += " -D USE_CUDA"                
                '''
                self.add_defination("USE_CUDA")

            if cuda:
                if config.is_fbcode():
                    # libs += ["cuda"]
                    self.add_libraries(["cuda"])
                else:
                    # libs += ["c10_cuda", "cuda", "torch_cuda"]
                    self.add_libraries(["c10_cuda", "cuda", "torch_cuda"])
            # build_arch_flags = vec_isa.build_arch_flags()
        else:
            # Note - this is effectively a header only inclusion. Usage of some header files may result in
            # symbol not found, if those header files require a library.
            # For those cases, include the lpath and libs command as we do for pytorch above.
            # This approach allows us to only pay for what we use.
            ipaths = cpp_extension.include_paths(cuda) + [sysconfig.get_path("include")]
            self.add_includes(ipaths)
            if aot_mode:
                ipaths = [os.path.dirname(cpp_prefix_path())]
                self.add_includes(ipaths)
            lpaths = []
            if sys.platform == "darwin":
                # only Apple builtin compilers (Apple Clang++) require openmp
                omp_available = not is_apple_clang()

                # ToDo: xuhan
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
            else:
                # libs = ["omp"] if config.is_fbcode() else ["gomp"]
                self.add_libraries(["omp"] if config.is_fbcode() else ["gomp"])

        # Unconditionally import c10 for non-abi-compatible mode to use TORCH_CHECK - See PyTorch #108690
        if not config.aot_inductor.abi_compatible:
            # libs += ["c10"]
            self.add_libraries(["c10"])
            # lpaths += [cpp_extension.TORCH_LIB_PATH]
            self.add_lib_dirs([cpp_extension.TORCH_LIB_PATH])

        # third party libs
        if config.is_fbcode():
            # ipaths.append(build_paths.sleef())
            self.add_includes(build_paths.sleef())
            #ipaths.append(build_paths.openmp())
            self.add_includes(build_paths.openmp())            
            # ipaths.append(build_paths.cc_include())
            self.add_includes(build_paths.cc_include())            
            # ipaths.append(build_paths.libgcc())
            self.add_includes(build_paths.libgcc())            
            # ipaths.append(build_paths.libgcc_arch())
            self.add_includes(build_paths.libgcc_arch())            
            # ipaths.append(build_paths.libgcc_backward())
            self.add_includes(build_paths.libgcc_backward())            
            # ipaths.append(build_paths.glibc())
            self.add_includes(build_paths.glibc())            
            # ipaths.append(build_paths.linux_kernel())
            self.add_includes(build_paths.linux_kernel())            
            # ipaths.append(build_paths.cuda())
            self.add_includes(build_paths.cuda())
            
            # We also need to bundle includes with absolute paths into a remote directory
            # (later on, we copy the include paths from cpp_extensions into our remote dir)
            # ipaths.append("include")
            self.add_includes("include")

        # static_link_libs = []
        if aot_mode and cuda and config.is_fbcode():
            # For Meta internal cuda-12, it is recommended to static link cudart
            # static_link_libs = ["-Wl,-Bstatic", "-lcudart_static", "-Wl,-Bdynamic"]
            self.add_libraries("cudart_static")
            if not _IS_WINDOWS:
                self.add_ldflags("Wl,-Bstatic", "Wl,-Bdynamic")

        # lpaths_str = " ".join(["-L" + p for p in lpaths])
        # libs_str = " ".join(static_link_libs + ["-l" + p for p in libs])
        # return ipaths, lpaths_str, libs_str, macros, build_arch_flags
