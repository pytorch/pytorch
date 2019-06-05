import os
import sys
from subprocess import check_call
from glob import glob
import multiprocessing
import shutil

from .setup_helpers import escape_path
from .setup_helpers.env import IS_64BIT, IS_WINDOWS, check_negative_env_flag
from .setup_helpers import cmake
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


def build_caffe2(version,
                 cmake_python_library,
                 build_python,
                 rerun_cmake,
                 cmake_only,
                 build_dir):
    my_env = _create_build_env()
    build_test = not check_negative_env_flag('BUILD_TEST')
    max_jobs = os.getenv('MAX_JOBS', str(multiprocessing.cpu_count()))
    cmake_cache_file = os.path.join(build_dir, 'CMakeCache.txt')
    ninja_build_file = os.path.join(build_dir, 'build.ninja')
    if rerun_cmake and os.path.isfile(cmake_cache_file):
        os.remove(cmake_cache_file)
    if not os.path.exists(cmake_cache_file) or (
            USE_NINJA and not os.path.exists(ninja_build_file)):
        cmake.run(version,
                  cmake_python_library,
                  build_python,
                  build_test,
                  build_dir,
                  my_env)
    if cmake_only:
        return
    build_cmd = [cmake.get_command(), '--build', '.',
                 '--target', 'install', '--config', cmake.get_build_type()]
    # This ``if-else'' clause would be unnecessary when cmake 3.12 becomes
    # minimum, which provides a '-j' option: build_cmd += ['-j', max_jobs]
    # would be sufficient by then.
    if IS_WINDOWS and not USE_NINJA:  # We are likely using msbuild here
        build_cmd += ['--', '/maxcpucount:{}'.format(max_jobs)]
    else:
        build_cmd += ['--', '-j', max_jobs]
    check_call(build_cmd, cwd=build_dir, env=my_env)

    # in cmake, .cu compilation involves generating certain intermediates
    # such as .cu.o and .cu.depend, and these intermediates finally get compiled
    # into the final .so.
    # Ninja updates build.ninja's timestamp after all dependent files have been built,
    # and re-kicks cmake on incremental builds if any of the dependent files
    # have a timestamp newer than build.ninja's timestamp.
    # There is a cmake bug with the Ninja backend, where the .cu.depend files
    # are still compiling by the time the build.ninja timestamp is updated,
    # so the .cu.depend file's newer timestamp is screwing with ninja's incremental
    # build detector.
    # This line works around that bug by manually updating the build.ninja timestamp
    # after the entire build is finished.
    if os.path.exists(ninja_build_file):
        os.utime(ninja_build_file, None)

    if build_python:
        caffe2_proto_dir = os.path.join(build_dir, 'caffe2', 'proto')
        for proto_file in glob(os.path.join(caffe2_proto_dir, '*.py')):
            if proto_file != os.path.join(caffe2_proto_dir, '__init__.py'):
                shutil.copy(proto_file, os.path.join('caffe2', 'proto'))
