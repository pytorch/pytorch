
from setup_helpers.env import (IS_ARM, IS_DARWIN, IS_LINUX, IS_PPC, IS_WINDOWS,
                               check_env_flag, check_negative_env_flag, hotpatch_build_env_vars)

import os
import sys
import distutils
import distutils.sysconfig
from subprocess import check_call, call, check_output
from distutils.version import StrictVersion
from setup_helpers.cuda import USE_CUDA
from setup_helpers.dist_check import USE_DISTRIBUTED, USE_GLOO_IBVERBS
from setup_helpers.nccl import USE_SYSTEM_NCCL, NCCL_INCLUDE_DIR, NCCL_ROOT_DIR, NCCL_SYSTEM_LIB
from setup_helpers.rocm import ROCM_HOME, ROCM_VERSION, USE_ROCM
from setup_helpers.nnpack import USE_NNPACK
from setup_helpers.qnnpack import USE_QNNPACK
from setup_helpers.cudnn import CUDNN_INCLUDE_DIR, CUDNN_LIB_DIR, CUDNN_LIBRARY, USE_CUDNN


from pprint import pprint
from glob import glob
import multiprocessing
import shutil

def which(thefile):
    path = os.environ.get("PATH", os.defpath).split(os.pathsep)
    for dir in path:
        name = os.path.join(dir, thefile)
        if (os.path.exists(name) and os.access(name, os.F_OK | os.X_OK)
                and not os.path.isdir(name)):
            return name
    return None

def cmake_version(cmd):
    for line in check_output([cmd, '--version']).split('\n'):
        if 'version' in line:
            return StrictVersion(line.strip().split(' ')[2])
    raise Exception('no version found')

def get_cmake_command():
    cmake_command = 'cmake'
    cmake3, cmake = which('cmake'), which('cmake')
    if cmake3 is not None and cmake is not None:
        bare_version = cmake_version(cmake)
        if bare_version < StrictVersion("3.5.0") and cmake_version(cmake3) > bare_version:
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
torch_lib_dir = base_dir + "/torch/lib"
install_dir = base_dir + "/torch/lib/tmp_install"

def mkdir_p(dir):
    try:
        os.makedirs(dir)
    except OSError:
        pass

def run_cmake(version,
              cmake_python_library,
              build_python,
              build_test):
    cmake_args = [
        get_cmake_command(),
        base_dir
    ]
    if USE_NINJA and not IS_WINDOWS:
        cmake_args.append('-GNinja')
    build_type = "Release"
    if check_env_flag('DEBUG'):
        build_type = "Debug"
    elif check_env_flag('REL_WITH_DEB_INFO'):
        build_type = "RelWithDebInfo"

    try:
        import numpy as np
        NUMPY_INCLUDE_DIR = np.get_include()
        USE_NUMPY = True
    except ImportError:
        USE_NUMPY = False
        NUMPY_INCLUDE_DIR=None

    cflags = os.getenv('CFLAGS') or ""
    ldflags = os.getenv('LDFLAGS') or ""
    if IS_DARWIN:
        ldflags += " -Wl,-rpath,@loader_path"
    elif USE_ROCM:
        ldflags += " -Wl,-rpath,\\\\\\$ORIGIN"
    else:
        ldflags += " -Wl,-rpath,$ORIGIN"


    # what joy! our cmake file sometimes looks at the system environment
    # and not cmake flags!
    if USE_CUDNN:
        my_env['CUDNN_LIBRARY'] = CUDNN_LIBRARY
        my_env['CUDNN_INCLUDE_DIR'] = CUDNN_INCLUDE_DIR

    mkdir_p(install_dir)
    mkdir_p('build')

    cmake_defines(cmake_args,
        PYTHON_EXECUTABLE=sys.executable,
        PYTHON_LIBRARY=cmake_python_library,
        PYTHON_INCLUDE_DIR= distutils.sysconfig.get_python_inc(),
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
        NUMPY_INCLUDE_DIR=NUMPY_INCLUDE_DIR,
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
        USE_MKLDNN=check_env_flag('USE_MKLDNN', 'OFF' if IS_PPC or IS_ARM else 'OFF'),
        NCCL_EXTERNAL=USE_CUDA,
        CMAKE_INSTALL_PREFIX=install_dir,
        CMAKE_C_FLAGS=cflags,
        CMAKE_CXX_FLAGS=cflags,
        CMAKE_EXE_LINKER_FLAGS=ldflags,
        CMAKE_SHARED_LINKER_FLAGS=ldflags,
        THD_SO_VERSION="1",
        CMAKE_PREFIX_PATH=os.getenv('CMAKE_PREFIX_PATH') or distutils.sysconfig.get_python_lib(),
        BLAS=os.getenv('BLAS'),
        CUDA_NVCC_EXECUTABLE=os.getenv('CUDA_NVCC_EXECUTABLE'),
        USE_REDIS=os.getenv('USE_REDIS'),
        USE_GLOG=os.getenv('USE_GLOG'),
        USE_GFLAGS=os.getenv('USE_GFLAGS'))

    if USE_GLOO_IBVERBS:
        cmake_args += cmake_defines(USE_IBVERBS="1", USE_GLOO_IBVERBS="1")

    expected_wrapper = '/usr/local/opt/ccache/libexec'
    if IS_DARWIN and os.path.exists(expected_wrapper):
        cmake_args += cmake_defines(CMAKE_C_COMPILER="{}/gcc".format(expected_wrapper),
                                    CMAKE_CXX_COMPILER="{}/g++".format(expected_wrapper))
    pprint(cmake_args)
    check_call(cmake_args, cwd='build')


def copy_files(build_test):
    if which("rsync") is not None:
        def sync(*args):
            check_call(('rsync', '-lptgoD') + args, cwd=torch_lib_dir)
    else:
        def sync(*args):
            check_call(('cp',) + args, cwd=torch_lib_dir)
    def sync_here(pattern):
        args = glob(pattern) + [ '-r', '.']
        sync(*args)
    shutil.rmtree(install_dir + '/lib/cmake', ignore_errors=True)
    shutil.rmtree(install_dir + '/lib/python', ignore_errors=True)
    sync_here(install_dir + '/lib/*')
    if os.path.exists(install_dir + '/lib64'):
        sync_here(install_dir + '/lib64/*')
    sync('../../aten/src/THNN/generic/THNN.h', '.')
    sync('../../aten/src/THCUNN/generic/THCUNN.h', '.')
    sync('-r', install_dir + '/include', '.')
    if os.path.exists(install_dir + '/bin/'):
        sync_here(install_dir + '/bin/*')

    if build_test:
        # Copy the test files to pytorch/caffe2 manually
        # They were built in pytorch/torch/lib/tmp_install/test
        # Why do we do this? So, setup.py has this section called 'package_data' which
        # you need to specify to include non-default files (usually .py files).
        # package_data takes a map from 'python package' to 'globs of files to
        # include'. By 'python package', it means a folder with an __init__.py file
        # that's not excluded in the find_packages call earlier in setup.py. So to
        # include our cpp_test into the site-packages folder in
        # site-packages/caffe2/cpp_test, we have to copy the cpp_test folder into the
        # root caffe2 folder and then tell setup.py to include them. Having another
        # folder like site-packages/caffe2_cpp_test would also be possible by adding a
        # caffe2_cpp_test folder to pytorch with an __init__.py in it.
        mkdir_p(base_dir + '/caffe2/cpp_test/')
        sync('-r', install_dir + '/test/*', base_dir + '/caffe2/cpp_test/')

def build_caffe2(version,
                 cmake_python_library,
                 build_python,
                 rerun_cmake):
    build_test = not check_negative_env_flag('BUILD_TEST')
    if rerun_cmake or not os.path.exists('build/CMakeCache.txt'):
        run_cmake(version,
                  cmake_python_library,
                  build_python,
                  build_test)

    if USE_NINJA:
        check_call(['ninja', 'install'], cwd='build')
    else:
        check_call(['make', '-j', multiprocessing.cpu_count(), 'install'], cwd='build')

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
            if proto_file != 'caffe2/proto/__init__.py':
                shutil.copyfile(proto_file, "caffe2/proto/" + os.path.basename(proto_file))

    copy_files(build_test)
