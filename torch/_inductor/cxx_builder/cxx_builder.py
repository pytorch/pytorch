import sys
from typing import Dict, List
import os
import errno
from pathlib import Path
import shlex
import subprocess
import re
import torch

# Windows need setup a temp dir to store .obj files.
_BUILD_TEMP_DIR = "CxxBuild"

# initialize variables for compilation
_IS_LINUX = sys.platform.startswith('linux')
_IS_MACOS = sys.platform.startswith('darwin')
_IS_WINDOWS = sys.platform == 'win32'

def _get_cxx_compiler():
    if _IS_WINDOWS:
        compiler = os.environ.get('CXX', 'cl')
    else:
        compiler = os.environ.get('CXX', 'c++')
    return compiler

def _nonduplicate_append(dest_list: list, src_list: list):
    for i in src_list:
        if not i in dest_list:
            dest_list.append(i)

def _create_if_dir_not_exist(path_dir):
    if not os.path.exists(path_dir):
        try:
            Path(path_dir).mkdir(parents=True, exist_ok=True)
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise RuntimeError("Fail to create path {}".format(path_dir))
            
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

class BuildOptionsBase(object):
    '''
    This is the Base class for store cxx build options, as a template.
    Acturally, to build a cxx shared library. We just need to select a compiler
    and maintains the suitable args.
    '''
    _compiler = ""
    _definations = []
    _include_dirs = []
    _cflags = []
    _ldlags = []
    _libraries_dirs = []
    _libraries = []
    _passthough_args = []

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

class CxxOptions(BuildOptionsBase):
    '''
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
    '''
    def _get_shared_cflag(self) -> List[str]:
        SHARED_FLAG = ['DLL'] if _IS_WINDOWS else ['shared', 'fPIC']
        return SHARED_FLAG
    
    def __init__(self) -> None:
        super().__init__()
        self._compiler = _get_cxx_compiler()
        _nonduplicate_append(self._cflags, ["O2"])
        _nonduplicate_append(self._cflags, self._get_shared_cflag())
        _nonduplicate_append(self._cflags, get_warning_all_flag())
    
def get_glibcxx_abi_build_flags() ->  List[str]:
    return ["-D_GLIBCXX_USE_CXX11_ABI=" + str(int(torch._C._GLIBCXX_USE_CXX11_ABI))]

class CxxTorchOptions(CxxOptions):
    '''
    This class is inherited from CxxTorchOptions, which automatic contains
    base cxx build options. And then it will maintains torch related build
    args.
    1. Torch include directories.
    2. Torch libraries.
    3. Torch libraries directories.
    4. Torch MACROs.
    5. MISC
    '''
    def __init__(self) -> None:
        super().__init__()

        if not _IS_WINDOWS:
            # glibcxx is not available in Windows.
            _nonduplicate_append(self._passthough_args, get_glibcxx_abi_build_flags())
    
class CxxTorchCudaOptions(CxxTorchOptions):
    '''
    This class is inherited from CxxTorchOptions, which automatic contains
    base cxx build options and torch common build options. And then it will
    maintains cuda device related build args.
    '''
    def __init__(self) -> None:
        super().__init__()
        #_nonduplicate_append(self._cflags, ["DCUDA"])

class CxxBuilder(object):
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
        SHARED_LIB_EXT = '.dll' if _IS_WINDOWS else '.so'
        return SHARED_LIB_EXT    

    def __init__(self, name: str, sources: List[str], BuildOption: BuildOptionsBase, output_dir: str = None) -> None:
        self._name = name
        self._sources_args = " ".join(sources)
        
        if output_dir is None:
            self._output_dir = os.path.dirname(os.path.abspath(__file__))
        else:
            self._output_dir = output_dir

        self._target_file = os.path.join(self._output_dir, f"{self._name}{self.get_shared_lib_ext()}")

        self._compiler = BuildOption.get_compiler()

        for cflag in BuildOption.get_cflags():
            if _IS_WINDOWS:
                self._cflags_args += (f"/{cflag} ")
            else:
                self._cflags_args += (f"-{cflag} ")

        for defination in BuildOption.get_definations():
            if _IS_WINDOWS:
                self._definations_args +=  (f"/D {defination} ")
            else:
                self._definations_args +=  (f"-D{defination} ")

        for inc_dir in BuildOption.get_include_dirs():
            if _IS_WINDOWS:
                self._include_dirs_args += (f"/I {inc_dir} ")
            else:
                self._include_dirs_args += (f"-I{inc_dir} ")
        
        for ldflag in BuildOption.get_ldlags():
            if _IS_WINDOWS:
                self._ldlags_args += (f"/{ldflag} ")
            else:    
                self._ldlags_args += (f"-{ldflag} ")
        
        for lib_dir in BuildOption.get_libraries_dirs():
            if _IS_WINDOWS:
                self._libraries_dirs_args += (f"/LIBPATH:{lib_dir} ")
            else:
                self._libraries_dirs_args += (f"-L{lib_dir} ")

        for lib in BuildOption.get_libraries():
            if _IS_WINDOWS:
                self._libraries_args += (f"{lib}.lib ")
            else:
                self._libraries_args += (f"-l{lib} ")

        for passthough_arg in BuildOption.get_passthough_args():
            self._passthough_parameters_args += (f"{passthough_arg}")

    def get_command_line(self) -> str:
        def format_build_command(compiler, sources, include_dirs_args, definations_args, cflags_args, ldflags_args, libraries_args, libraries_dirs_args, passthougn_args, target_file):
            if _IS_WINDOWS:
                # https://learn.microsoft.com/en-us/cpp/build/walkthrough-compile-a-c-program-on-the-command-line?view=msvc-1704
                # https://stackoverflow.com/a/31566153
                cmd = f"{compiler} {include_dirs_args} {definations_args} {cflags_args} {sources} {ldflags_args} {libraries_args} {libraries_dirs_args} {passthougn_args} /LD /Fe{target_file}"
                cmd = cmd.replace("\\", "\\\\")
            else:
                cmd = f"{compiler} {include_dirs_args} {sources} {definations_args} {cflags_args} {ldflags_args} {libraries_args} {libraries_dirs_args} {passthougn_args} -o {target_file}"
            return cmd
        
        command_line = format_build_command(compiler=self._compiler, sources=self._sources_args, include_dirs_args=self._include_dirs_args, definations_args=self._definations_args,
                                            cflags_args=self._cflags_args, ldflags_args=self._ldlags_args, libraries_args=self._libraries_args, libraries_dirs_args=self._libraries_dirs_args,
                                            passthougn_args=self._passthough_parameters_args, target_file=self._target_file)
        return command_line
    
    def build(self):
        '''
        It is must need a temperary directory to store object files in Windows.
        '''
        _create_if_dir_not_exist(self._output_dir)
        _build_tmp_dir = os.path.join(self._output_dir, f"{self._name}_{_BUILD_TEMP_DIR}")
        _create_if_dir_not_exist(_build_tmp_dir)

        build_cmd = self.get_command_line()
        print("!!! build_cmd: ", build_cmd)
        status = run_command_line(build_cmd, cwd=_build_tmp_dir)

        _remove_dir(_build_tmp_dir)
        return status, self._target_file


cxx_build_options = CxxOptions()
torch_build_options = CxxTorchOptions()
cuda_build_options = CxxTorchCudaOptions()
