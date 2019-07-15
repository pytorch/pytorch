import os
import sys
from glob import glob
import shutil


def which(thefile):
    path = os.environ.get("PATH", os.defpath).split(os.pathsep)
    for dir in path:
        fname = os.path.join(dir, thefile)
        fnames = [fname]
        if IS_WINDOWS:
            exts = os.environ.get('PATHEXT', '').split(os.pathsep)
            fnames += [fname + ext for ext in exts]
        for name in fnames:
            if (os.path.exists(name) and os.access(name, os.F_OK | os.X_OK)
                    and not os.path.isdir(name)):
                return name
    return None


def cmake_version(cmd):
    for line in check_output([cmd, '--version']).decode('utf-8').split('\n'):
        if 'version' in line:
            return LooseVersion(line.strip().split(' ')[2])
    raise Exception('no version found')


def get_cmake_command():
    cmake_command = 'cmake'
    if IS_WINDOWS:
        return cmake_command
    cmake3 = which('cmake3')
    cmake_command = cmake3
    '''
    if cmake3 is not None:
        cmake = which('cmake')
        if cmake is not None:
            bare_version = cmake_version(cmake)
            if bare_version < LooseVersion("3.5.0") and cmake_version(cmake3) > bare_version:
                cmake_command = 'cmake3'
    '''
    return cmake_command


def cmake_defines(lst, **kwargs):
    for key in sorted(kwargs.keys()):
        value = kwargs[key]
        if value is not None:
            lst.append('-D{}={}'.format(key, value))


# Ninja
# Use ninja if it is on the PATH. Previous version of PyTorch required the
# ninja python package, but we no longer use it, so we do not have to import it
USE_NINJA = not check_negative_env_flag('USE_NINJA') and (which('ninja') is not None)

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
install_dir = base_dir + "/torch"
build_type = "Release"
if DEBUG:
    build_type = "Debug"
elif REL_WITH_DEB_INFO:
    build_type = "RelWithDebInfo"


def overlay_windows_vcvars(env):
    from distutils._msvccompiler import _get_vc_env
    vc_arch = 'x64' if IS_64BIT else 'x86'
    vc_env = _get_vc_env(vc_arch)
    for k, v in env.items():
        lk = k.lower()
        if lk not in vc_env:
            vc_env[lk] = v
    return vc_env

from .setup_helpers import escape_path
from .setup_helpers.env import IS_64BIT, IS_WINDOWS, check_negative_env_flag
from .setup_helpers.cmake import USE_NINJA
from .setup_helpers.cuda import USE_CUDA, CUDA_HOME
from .setup_helpers.cudnn import CUDNN_INCLUDE_DIR, CUDNN_LIBRARY, USE_CUDNN



def _overlay_windows_vcvars(env):
    if sys.version_info >= (3, 5):
        from distutils._msvccompiler import _get_vc_env
        vc_arch = 'x64' if IS_64BIT else 'x86'
        vc_env = _get_vc_env(vc_arch)
        # Keys in `_get_vc_env` are always lowercase.
        # We turn them into uppercase before overlaying vcvars
        # because OS environ keys are always uppercase on Windows.
        # https://stackoverflow.com/a/7797329
        vc_env = {k.upper(): v for k, v in vc_env.items()}
        for k, v in env.items():
            uk = k.upper()
            if uk not in vc_env:
                vc_env[uk] = v
        return vc_env
    else:
        return env


def _create_build_env():
    # XXX - our cmake file sometimes looks at the system environment
    # and not cmake flags!
    # you should NEVER add something to this list. It is bad practice to
    # have cmake read the environment
    my_env = os.environ.copy()
    if USE_CUDNN:
        my_env['CUDNN_LIBRARY'] = escape_path(CUDNN_LIBRARY)
        my_env['CUDNN_INCLUDE_DIR'] = escape_path(CUDNN_INCLUDE_DIR)
    if USE_CUDA:
        my_env['CUDA_BIN_PATH'] = escape_path(CUDA_HOME)

    if IS_WINDOWS and USE_NINJA:
        # When using Ninja under Windows, the gcc toolchain will be chosen as
        # default. But it should be set to MSVC as the user's first choice.
        my_env = _overlay_windows_vcvars(my_env)
        my_env.setdefault('CC', 'cl')
        my_env.setdefault('CXX', 'cl')
    return my_env


def build_caffe2(version, cmake_python_library, build_python, rerun_cmake, cmake_only, cmake):
    my_env = _create_build_env()
    build_test = not check_negative_env_flag('BUILD_TEST')
    cmake.generate(version,
                   cmake_python_library,
                   build_python,
                   build_test,
                   my_env,
                   rerun_cmake)
    if cmake_only:
        return
    cmake.build(my_env)
    if build_python:
        caffe2_proto_dir = os.path.join(cmake.build_dir, 'caffe2', 'proto')
        for proto_file in glob(os.path.join(caffe2_proto_dir, '*.py')):
            if proto_file != os.path.join(caffe2_proto_dir, '__init__.py'):
                shutil.copy(proto_file, os.path.join('caffe2', 'proto'))
