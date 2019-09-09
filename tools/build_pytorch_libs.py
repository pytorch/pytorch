import os
import sys
from glob import glob
import shutil

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
        my_env['CUDNN_LIBRARY'] = CUDNN_LIBRARY
        my_env['CUDNN_INCLUDE_DIR'] = CUDNN_INCLUDE_DIR
    if USE_CUDA:
        my_env['CUDA_BIN_PATH'] = CUDA_HOME

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
