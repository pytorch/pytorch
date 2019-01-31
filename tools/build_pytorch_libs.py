from .setup_helpers.env import (IS_ARM, IS_DARWIN, IS_LINUX, IS_PPC, IS_WINDOWS,
                                DEBUG, REL_WITH_DEB_INFO, USE_MKLDNN,
                                check_env_flag, check_negative_env_flag, hotpatch_build_env_vars)

import os
import sys
import distutils
import distutils.sysconfig
from distutils.file_util import copy_file
from distutils.dir_util import copy_tree
from subprocess import check_call, call, check_output
from distutils.version import LooseVersion
from .setup_helpers.cuda import USE_CUDA, CUDA_HOME
from .setup_helpers.dist_check import USE_DISTRIBUTED, USE_GLOO_IBVERBS
from .setup_helpers.nccl import USE_SYSTEM_NCCL, NCCL_INCLUDE_DIR, NCCL_ROOT_DIR, NCCL_SYSTEM_LIB
from .setup_helpers.rocm import ROCM_HOME, ROCM_VERSION, USE_ROCM
from .setup_helpers.nnpack import USE_NNPACK
from .setup_helpers.qnnpack import USE_QNNPACK
from .setup_helpers.cudnn import CUDNN_INCLUDE_DIR, CUDNN_LIB_DIR, CUDNN_LIBRARY, USE_CUDNN


from pprint import pprint
from glob import glob
import multiprocessing
import shutil


def which(thefile):
    path = os.environ.get("PATH", os.defpath).split(os.pathsep)
    for dir in path:
        fname = os.path.join(dir, thefile)
        fnames = [fname]
        if IS_WINDOWS:
            exts = os.environ.get('PATHEXT', '').split(os.sep)
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
    if cmake3 is not None:
        cmake = which('cmake')
        if cmake is not None:
            bare_version = cmake_version(cmake)
            if bare_version < LooseVersion("3.5.0") and cmake_version(cmake3) > bare_version:
                cmake_command = 'cmake3'
    return cmake_command


def cmake_defines(lst, **kwargs):
    for key in sorted(kwargs.keys()):
        value = kwargs[key]
        if value is not None:
            lst.append('-D{}={}'.format(key, value))


# Ninja
try:
    import ninja
    USE_NINJA = True
except ImportError:
    USE_NINJA = False

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
install_dir = base_dir + "/torch"
build_type = "Release"
if DEBUG:
    build_type = "Debug"
elif REL_WITH_DEB_INFO:
    build_type = "RelWithDebInfo"


def mkdir_p(dir):
    try:
        os.makedirs(dir)
    except OSError:
        pass


def run_cmake(version,
              cmake_python_library,
              build_python,
              build_test,
              build_dir):
    cmake_args = [
        get_cmake_command(),
        base_dir
    ]
    if USE_NINJA:
        cmake_args.append('-GNinja')
    elif IS_WINDOWS:
        cmake_args.append('-GVisual Studio 15 2017 Win64')
    try:
        import numpy as np
        NUMPY_INCLUDE_DIR = np.get_include()
        USE_NUMPY = True
    except ImportError:
        USE_NUMPY = False
        NUMPY_INCLUDE_DIR = None

    cflags = os.getenv('CFLAGS') or ""
    ldflags = os.getenv('LDFLAGS') or ""
    if IS_WINDOWS:
        cflags += " /EHa"

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

    mkdir_p(install_dir)
    mkdir_p(build_dir)

    cmake_defines(
        cmake_args,
        PYTHON_EXECUTABLE=escape_path(sys.executable),
        PYTHON_LIBRARY=escape_path(cmake_python_library),
        PYTHON_INCLUDE_DIR=escape_path(distutils.sysconfig.get_python_inc()),
        BUILDING_WITH_TORCH_LIBS="ON",
        TORCH_BUILD_VERSION=version,
        CMAKE_BUILD_TYPE=build_type,
        BUILD_TORCH="ON",
        BUILD_PYTHON=build_python,
        BUILD_SHARED_LIBS=os.getenv("BUILD_SHARED_LIBS") or "ON",
        BUILD_BINARY=check_env_flag('BUILD_BINARY'),
        BUILD_TEST=build_test,
        INSTALL_TEST=build_test,
        BUILD_CAFFE2_OPS=not check_negative_env_flag('BUILD_CAFFE2_OPS'),
        ONNX_NAMESPACE=os.getenv("ONNX_NAMESPACE") or "onnx_torch",
        USE_CUDA=USE_CUDA,
        USE_DISTRIBUTED=USE_DISTRIBUTED,
        USE_FBGEMM=not (check_env_flag('NO_FBGEMM') or check_negative_env_flag('USE_FBGEMM')),
        USE_NUMPY=USE_NUMPY,
        NUMPY_INCLUDE_DIR=escape_path(NUMPY_INCLUDE_DIR),
        USE_SYSTEM_NCCL=USE_SYSTEM_NCCL,
        NCCL_INCLUDE_DIR=NCCL_INCLUDE_DIR,
        NCCL_ROOT_DIR=NCCL_ROOT_DIR,
        NCCL_SYSTEM_LIB=NCCL_SYSTEM_LIB,
        CAFFE2_STATIC_LINK_CUDA=check_env_flag('USE_CUDA_STATIC_LINK'),
        USE_ROCM=USE_ROCM,
        USE_NNPACK=USE_NNPACK,
        USE_LEVELDB=check_env_flag('USE_LEVELDB'),
        USE_LMDB=check_env_flag('USE_LMDB'),
        USE_OPENCV=check_env_flag('USE_OPENCV'),
        USE_QNNPACK=USE_QNNPACK,
        USE_TENSORRT=check_env_flag('USE_TENSORRT'),
        USE_FFMPEG=check_env_flag('USE_FFMPEG'),
        USE_SYSTEM_EIGEN_INSTALL="OFF",
        USE_MKLDNN=USE_MKLDNN,
        NCCL_EXTERNAL=USE_CUDA,
        CMAKE_INSTALL_PREFIX=install_dir,
        CMAKE_C_FLAGS=cflags,
        CMAKE_CXX_FLAGS=cflags,
        CMAKE_EXE_LINKER_FLAGS=ldflags,
        CMAKE_SHARED_LINKER_FLAGS=ldflags,
        THD_SO_VERSION="1",
        CMAKE_PREFIX_PATH=os.getenv('CMAKE_PREFIX_PATH') or distutils.sysconfig.get_python_lib(),
        BLAS=os.getenv('BLAS'),
        CUDA_NVCC_EXECUTABLE=escape_path(os.getenv('CUDA_NVCC_EXECUTABLE')),
        USE_REDIS=os.getenv('USE_REDIS'),
        USE_GLOG=os.getenv('USE_GLOG'),
        USE_GFLAGS=os.getenv('USE_GFLAGS'),
        WERROR=os.getenv('WERROR'))

    if USE_GLOO_IBVERBS:
        cmake_defines(cmake_args, USE_IBVERBS="1", USE_GLOO_IBVERBS="1")

    expected_wrapper = '/usr/local/opt/ccache/libexec'
    if IS_DARWIN and os.path.exists(expected_wrapper):
        cmake_defines(cmake_args,
                      CMAKE_C_COMPILER="{}/gcc".format(expected_wrapper),
                      CMAKE_CXX_COMPILER="{}/g++".format(expected_wrapper))
    pprint(cmake_args)
    pprint(os.environ.items())
    check_call(cmake_args, cwd=build_dir, env=my_env)


def build_caffe2(version,
                 cmake_python_library,
                 build_python,
                 rerun_cmake,
                 build_dir):
    build_test = not check_negative_env_flag('BUILD_TEST')
    cmake_cache_file = 'build/CMakeCache.txt'
    if rerun_cmake and os.path.isfile(cmake_cache_file):
        os.remove(cmake_cache_file)
    if not os.path.exists(cmake_cache_file) or (USE_NINJA and not os.path.exists('build/build.ninja')):
        run_cmake(version,
                  cmake_python_library,
                  build_python,
                  build_test,
                  build_dir)
    if IS_WINDOWS:
        if USE_NINJA:
            # sccache will fail if all cores are used for compiling
            j = max(1, multiprocessing.cpu_count() - 1)
            check_call(['cmake', '--build', '.', '--target', 'install', '--config', build_type, '--', '-j', str(j)],
                       cwd=build_dir)
        else:
            check_call(['msbuild', 'INSTALL.vcxproj', '/p:Configuration={}'.format(build_type)],
                       cwd=build_dir)
    else:
        if USE_NINJA:
            check_call(['ninja', 'install'], cwd=build_dir)
        else:
            max_jobs = os.getenv('MAX_JOBS', str(multiprocessing.cpu_count()))
            check_call(['make', '-j', str(max_jobs), 'install'], cwd=build_dir)

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
    if os.path.exists('build/build.ninja'):
        os.utime('build/build.ninja', None)

    if build_python:
        for proto_file in glob('build/caffe2/proto/*.py'):
            if os.path.sep != '/':
                proto_file = proto_file.replace(os.path.sep, '/')
            if proto_file != 'build/caffe2/proto/__init__.py':
                shutil.copyfile(proto_file, "caffe2/proto/" + os.path.basename(proto_file))


def escape_path(path):
    if os.path.sep != '/' and path is not None:
        return path.replace(os.path.sep, '/')
    return path
