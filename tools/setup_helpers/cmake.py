from __future__ import print_function

import os
from subprocess import check_call, check_output
import sys
from pprint import pprint
import distutils
import distutils.sysconfig
from distutils.version import LooseVersion

from . import escape_path
from .env import (IS_64BIT, IS_DARWIN, IS_WINDOWS,
                  DEBUG, REL_WITH_DEB_INFO, USE_MKLDNN,
                  check_env_flag, check_negative_env_flag)
from .cuda import USE_CUDA
from .dist_check import USE_DISTRIBUTED, USE_GLOO_IBVERBS
from .nccl import (USE_SYSTEM_NCCL, NCCL_INCLUDE_DIR, NCCL_ROOT_DIR,
                   NCCL_SYSTEM_LIB, USE_NCCL)
from .rocm import USE_ROCM
from .nnpack import USE_NNPACK
from .qnnpack import USE_QNNPACK


def _which(thefile):
    path = os.environ.get("PATH", os.defpath).split(os.pathsep)
    for d in path:
        fname = os.path.join(d, thefile)
        fnames = [fname]
        if IS_WINDOWS:
            exts = os.environ.get('PATHEXT', '').split(os.pathsep)
            fnames += [fname + ext for ext in exts]
        for name in fnames:
            if os.access(name, os.F_OK | os.X_OK) and not os.path.isdir(name):
                return name
    return None


def _mkdir_p(d):
    try:
        os.makedirs(d)
    except OSError:
        pass


# Ninja
# Use ninja if it is on the PATH. Previous version of PyTorch required the
# ninja python package, but we no longer use it, so we do not have to import it
USE_NINJA = (not check_negative_env_flag('USE_NINJA') and
             _which('ninja') is not None)


def get_version(cmd):
    "Returns cmake version."

    for line in check_output([cmd, '--version']).decode('utf-8').split('\n'):
        if 'version' in line:
            return LooseVersion(line.strip().split(' ')[2])
    raise Exception('no version found')


def get_command():
    "Returns cmake command."

    cmake_command = 'cmake'
    if IS_WINDOWS:
        return cmake_command
    cmake3 = _which('cmake3')
    if cmake3 is not None:
        cmake = _which('cmake')
        if cmake is not None:
            bare_version = get_version(cmake)
            if (bare_version < LooseVersion("3.5.0") and
                    get_version(cmake3) > bare_version):
                cmake_command = 'cmake3'
    return cmake_command


def defines(lst, **kwargs):
    "Add definitions to cmake."
    for key, value in sorted(kwargs.items()):
        if value is not None:
            lst.append('-D{}={}'.format(key, value))


def get_build_type():
    "Get the build type."
    if DEBUG:
        return "Debug"
    elif REL_WITH_DEB_INFO:
        return "RelWithDebInfo"
    else:
        return "Release"


def run(version,
        cmake_python_library,
        build_python,
        build_test,
        build_dir,
        my_env):
    "Run cmake."

    cmake_args = [
        get_command()
    ]
    if USE_NINJA:
        cmake_args.append('-GNinja')
    elif IS_WINDOWS:
        cmake_args.append('-GVisual Studio 15 2017')
        if IS_64BIT:
            cmake_args.append('-Ax64')
            cmake_args.append('-Thost=x64')
    try:
        import numpy as np
    except ImportError:
        USE_NUMPY = False
        NUMPY_INCLUDE_DIR = None
    else:
        NUMPY_INCLUDE_DIR = np.get_include()
        USE_NUMPY = True

    cflags = os.getenv('CFLAGS', "") + " " + os.getenv('CPPFLAGS', "")
    ldflags = os.getenv('LDFLAGS', "")
    if IS_WINDOWS:
        defines(cmake_args,
                MSVC_Z7_OVERRIDE=os.getenv('MSVC_Z7_OVERRIDE', "ON"))
        cflags += " /EHa"

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    install_dir = os.path.join(base_dir, "torch")

    _mkdir_p(install_dir)
    _mkdir_p(build_dir)

    defines(
        cmake_args,
        PYTHON_EXECUTABLE=escape_path(sys.executable),
        PYTHON_LIBRARY=escape_path(cmake_python_library),
        PYTHON_INCLUDE_DIR=escape_path(distutils.sysconfig.get_python_inc()),
        BUILDING_WITH_TORCH_LIBS=os.getenv("BUILDING_WITH_TORCH_LIBS", "ON"),
        TORCH_BUILD_VERSION=version,
        CMAKE_BUILD_TYPE=get_build_type(),
        BUILD_PYTHON=build_python,
        BUILD_SHARED_LIBS=os.getenv("BUILD_SHARED_LIBS", "ON"),
        BUILD_BINARY=check_env_flag('BUILD_BINARY'),
        BUILD_TEST=build_test,
        INSTALL_TEST=build_test,
        BUILD_CAFFE2_OPS=not check_negative_env_flag('BUILD_CAFFE2_OPS'),
        ONNX_NAMESPACE=os.getenv("ONNX_NAMESPACE", "onnx_torch"),
        ONNX_ML=not check_negative_env_flag("ONNX_ML"),
        USE_CUDA=USE_CUDA,
        USE_DISTRIBUTED=USE_DISTRIBUTED,
        USE_FBGEMM=not (check_env_flag('NO_FBGEMM') or
                        check_negative_env_flag('USE_FBGEMM')),
        NAMEDTENSOR_ENABLED=(check_env_flag('USE_NAMEDTENSOR') or
                             check_negative_env_flag('NO_NAMEDTENSOR')),
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
        USE_NCCL=USE_NCCL,
        NCCL_EXTERNAL=USE_NCCL,
        CMAKE_INSTALL_PREFIX=install_dir,
        CMAKE_C_FLAGS=cflags,
        CMAKE_CXX_FLAGS=cflags,
        CMAKE_EXE_LINKER_FLAGS=ldflags,
        CMAKE_SHARED_LINKER_FLAGS=ldflags,
        THD_SO_VERSION="1",
        CMAKE_PREFIX_PATH=(os.getenv('CMAKE_PREFIX_PATH') or
                           distutils.sysconfig.get_python_lib()),
        BLAS=os.getenv('BLAS'),
        CUDA_NVCC_EXECUTABLE=escape_path(os.getenv('CUDA_NVCC_EXECUTABLE')),
        USE_REDIS=os.getenv('USE_REDIS'),
        USE_GLOG=os.getenv('USE_GLOG'),
        USE_GFLAGS=os.getenv('USE_GFLAGS'),
        USE_ASAN=check_env_flag('USE_ASAN'),
        WERROR=os.getenv('WERROR'))

    if os.getenv('_GLIBCXX_USE_CXX11_ABI'):
        defines(cmake_args, GLIBCXX_USE_CXX11_ABI=os.getenv('_GLIBCXX_USE_CXX11_ABI'))

    if os.getenv('USE_OPENMP'):
        defines(cmake_args, USE_OPENMP=check_env_flag('USE_OPENMP'))

    if os.getenv('USE_TBB'):
        defines(cmake_args, USE_TBB=check_env_flag('USE_TBB'))

    if os.getenv('MKL_SEQ'):
        defines(cmake_args, INTEL_MKL_SEQUENTIAL=check_env_flag('MKL_SEQ'))

    if os.getenv('MKL_TBB'):
        defines(cmake_args, INTEL_MKL_TBB=check_env_flag('MKL_TBB'))

    mkldnn_threading = os.getenv('MKLDNN_THREADING')
    if mkldnn_threading:
        defines(cmake_args, MKLDNN_THREADING=mkldnn_threading)

    parallel_backend = os.getenv('PARALLEL_BACKEND')
    if parallel_backend:
        defines(cmake_args, PARALLEL_BACKEND=parallel_backend)

    if USE_GLOO_IBVERBS:
        defines(cmake_args, USE_IBVERBS="1", USE_GLOO_IBVERBS="1")

    if USE_MKLDNN:
        defines(cmake_args, MKLDNN_ENABLE_CONCURRENT_EXEC="ON")

    expected_wrapper = '/usr/local/opt/ccache/libexec'
    if IS_DARWIN and os.path.exists(expected_wrapper):
        defines(cmake_args,
                CMAKE_C_COMPILER="{}/gcc".format(expected_wrapper),
                CMAKE_CXX_COMPILER="{}/g++".format(expected_wrapper))
    for env_var_name in my_env:
        if env_var_name.startswith('gh'):
            # github env vars use utf-8, on windows, non-ascii code may
            # cause problem, so encode first
            try:
                my_env[env_var_name] = str(my_env[env_var_name].encode("utf-8"))
            except UnicodeDecodeError as e:
                shex = ':'.join('{:02x}'.format(ord(c)) for c in my_env[env_var_name])
                print('Invalid ENV[{}] = {}'.format(env_var_name, shex),
                      file=sys.stderr)
                print(e, file=sys.stderr)
    # According to the CMake manual, we should pass the arguments first,
    # and put the directory as the last element. Otherwise, these flags
    # may not be passed correctly.
    # Reference:
    # 1. https://cmake.org/cmake/help/latest/manual/cmake.1.html#synopsis
    # 2. https://stackoverflow.com/a/27169347
    cmake_args.append(base_dir)
    pprint(cmake_args)
    check_call(cmake_args, cwd=build_dir, env=my_env)
