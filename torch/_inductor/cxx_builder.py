import errno
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List

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
                    cmd_libraries += f"-L{lib} "

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

        return cmd_include_dirs, cmd_libraries, cmd_definations, cmd_cflags, cmd_ldflags

    # Config
    def add_sources(self, sources: List[str]):
        for i in sources:
            self.__sources.append(i)

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
            self.add_ldflags([self.__get_shared_flag()])
        else:
            file_ext = self.get_exec_ext()
            
        target_file = f"{self.__name}{file_ext}"
        target_file = os.path.join(build_root, target_file)            

    def get_build_cmd(self):
        if self.__name is None:
            raise RuntimeError("target name should not be None.")

        if self.__is_shared:
            self.add_ldflags([self.__get_shared_flag()])

        if _IS_WINDOWS:
            self.add_libraries(_get_windows_runtime_libs())
            
        target_file = self.get_target_file_path()

        compiler = _get_cxx_compiler()
        (
            cmd_include_dirs,
            cmd_libraries,
            cmd_definations,
            cmd_cflags,
            cmd_ldflags,
        ) = self.__prepare_build_parameters()

        def format_build_command(
            compiler,
            src_file,
            cmd_include_dirs,
            cmd_definations,
            cmd_cflags,
            cmd_ldflags,
            cmd_libraries,
            target_file,
        ):
            srcs = " ".join(src_file)
            if _IS_WINDOWS:
                # https://learn.microsoft.com/en-us/cpp/build/walkthrough-compile-a-c-program-on-the-command-line?view=msvc-1704
                # https://stackoverflow.com/a/31566153
                cmd = f"{compiler} {cmd_include_dirs} {cmd_definations} {cmd_cflags} {srcs} {cmd_ldflags} {cmd_libraries} /LD /Fe{target_file}"
                cmd = cmd.replace("\\", "\\\\")
            else:
                cmd = f"{compiler} {cmd_include_dirs} {srcs} {cmd_definations} {cmd_cflags} {cmd_ldflags} {cmd_libraries} -o {target_file}"
            return cmd

        build_cmd = format_build_command(
            compiler=compiler,
            src_file=self.__sources,
            cmd_include_dirs=cmd_include_dirs,
            cmd_definations=cmd_definations,
            cmd_cflags=cmd_cflags,
            cmd_ldflags=cmd_ldflags,
            cmd_libraries=cmd_libraries,
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
