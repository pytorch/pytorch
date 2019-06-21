"Manages CMake."

from __future__ import print_function

import multiprocessing
import os
import re
from subprocess import check_call, check_output
import sys
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
from .numpy_ import USE_NUMPY, NUMPY_INCLUDE_DIR
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


class CMake:
    "Manages cmake."

    def __init__(self, build_dir):
        self._cmake_command = CMake._get_cmake_command()
        self.build_dir = build_dir

        if DEBUG:
            self._build_type = "Debug"
        elif REL_WITH_DEB_INFO:
            self._build_type = "RelWithDebInfo"
        else:
            self._build_type = "Release"

    @property
    def _cmake_cache_file(self):
        r"""Returns the path to CMakeCache.txt.

        Returns:
          string: The path to CMakeCache.txt.
        """
        return os.path.join(self.build_dir, 'CMakeCache.txt')

    @staticmethod
    def _get_cmake_command():
        "Returns cmake command."

        cmake_command = 'cmake'
        if IS_WINDOWS:
            return cmake_command
        cmake3 = _which('cmake3')
        if cmake3 is not None:
            cmake = _which('cmake')
            if cmake is not None:
                bare_version = CMake._get_version(cmake)
                if (bare_version < LooseVersion("3.5.0") and
                        CMake._get_version(cmake3) > bare_version):
                    cmake_command = 'cmake3'
        return cmake_command

    @staticmethod
    def _get_version(cmd):
        "Returns cmake version."

        for line in check_output([cmd, '--version']).decode('utf-8').split('\n'):
            if 'version' in line:
                return LooseVersion(line.strip().split(' ')[2])
        raise RuntimeError('no version found')

    def run(self, args, env):
        "Executes cmake with arguments and an environment."

        command = [self._cmake_command] + args
        print(' '.join(command))
        check_call(command, cwd=self.build_dir, env=env)

    @staticmethod
    def defines(args, **kwargs):
        "Adds definitions to a cmake argument list."
        for key, value in sorted(kwargs.items()):
            if value is not None:
                args.append('-D{}={}'.format(key, value))

    @staticmethod
    def convert_cmake_value_to_python_value(cmake_value, cmake_type):
        r"""Convert a CMake value in a string form to a Python value.

        Arguments:
          cmake_value (string): The CMake value in a string form (e.g., "ON", "OFF", "1").
          cmake_type (string): The CMake type of :attr:`cmake_value`.

        Returns:
          A Python value corresponding to :attr:`cmake_value` with type :attr:`cmake_type`.
        """

        cmake_type = cmake_type.upper()
        up_val = cmake_value.upper()
        if cmake_type == 'BOOL':
            # https://gitlab.kitware.com/cmake/community/wikis/doc/cmake/VariablesListsStrings#boolean-values-in-cmake
            return not (up_val in ('FALSE', 'OFF', 'N', 'NO', '0', '', 'NOTFOUND') or up_val.endswith('-NOTFOUND'))
        elif cmake_type == 'FILEPATH':
            if up_val.endswith('-NOTFOUND'):
                return None
            else:
                return cmake_value
        else:  # Directly return the cmake_value.
            return cmake_value

    @staticmethod
    def _get_cmake_cache_variables(cmake_cache_file):
        r"""Gets values in CMakeCache.txt into a dictionary.

        Arguments:
          cmake_cache_file: A CMakeCache.txt file object.
        Returns:
          dict: A ``dict`` containing the value of cached CMake variables.
        """

        results = dict()
        for line in cmake_cache_file:
            line = line.strip()
            if not line or line.startswith(('#', '//')):
                # Blank or comment line, skip
                continue

            # Space can also be part of variable name and value
            matched = re.match(r'(\S.*):\s*([a-zA-Z_-][a-zA-Z0-9_-]*)\s*=\s*(.*)', line)
            if matched is None:  # Illegal line
                raise ValueError('Unexpected line in {}: {}'.format(repr(cmake_cache_file), line))
            variable, type_, value = matched.groups()
            if type_.upper() in ('INTERNAL', 'STATIC'):
                # CMake internal variable, do not touch
                continue
            results[variable] = CMake.convert_cmake_value_to_python_value(value, type_)

        return results

    def get_cmake_cache_variables(self):
        r"""Gets values in CMakeCache.txt into a dictionary.
        Returns:
          dict: A ``dict`` containing the value of cached CMake variables.
        """
        with open(self._cmake_cache_file) as f:
            return CMake._get_cmake_cache_variables(f)

    def generate(self, version, cmake_python_library, build_python, build_test, my_env, rerun):
        "Runs cmake to generate native build files."

        if rerun and os.path.isfile(self._cmake_cache_file):
            os.remove(self._cmake_cache_file)
        ninja_build_file = os.path.join(self.build_dir, 'build.ninja')
        if os.path.exists(self._cmake_cache_file) and not (
                USE_NINJA and not os.path.exists(ninja_build_file)):
            # Everything's in place. Do not rerun.
            return

        args = []
        if USE_NINJA:
            args.append('-GNinja')
        elif IS_WINDOWS:
            generator = os.getenv('CMAKE_GENERATOR', 'Visual Studio 15 2017')
            supported = ['Visual Studio 15 2017', 'Visual Studio 16 2019']
            if generator not in supported:
                print('Unsupported `CMAKE_GENERATOR`: ' + generator)
                print('Please set it to one of the following values: ')
                print('\n'.join(supported))
                sys.exit(1)
            args.append('-G' + generator)
            toolset_dict = {}
            toolset_version = os.getenv('CMAKE_GENERATOR_TOOLSET_VERSION')
            if toolset_version is not None:
                toolset_dict['version'] = toolset_version
                curr_toolset = os.getenv('VCToolsVersion')
                if curr_toolset is None:
                    print('When you specify `CMAKE_GENERATOR_TOOLSET_VERSION`, you must also '
                          'activate the vs environment of this version. Please read the notes '
                          'in the build steps carefully.')
                    sys.exit(1)
            if IS_64BIT:
                args.append('-Ax64')
                toolset_dict['host'] = 'x64'
            if toolset_dict:
                toolset_expr = ','.join(["{}={}".format(k, v) for k, v in toolset_dict.items()])
                args.append('-T' + toolset_expr)

        cflags = os.getenv('CFLAGS', "") + " " + os.getenv('CPPFLAGS', "")
        ldflags = os.getenv('LDFLAGS', "")
        if IS_WINDOWS:
            CMake.defines(args, MSVC_Z7_OVERRIDE=not check_negative_env_flag(
                'MSVC_Z7_OVERRIDE'))
            cflags += " /EHa"

        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        install_dir = os.path.join(base_dir, "torch")

        _mkdir_p(install_dir)
        _mkdir_p(self.build_dir)

        CMake.defines(args,
                      PYTHON_EXECUTABLE=escape_path(sys.executable),
                      PYTHON_LIBRARY=escape_path(cmake_python_library),
                      PYTHON_INCLUDE_DIR=escape_path(distutils.sysconfig.get_python_inc()),
                      BUILDING_WITH_TORCH_LIBS=os.getenv("BUILDING_WITH_TORCH_LIBS", "ON"),
                      TORCH_BUILD_VERSION=version,
                      CMAKE_BUILD_TYPE=self._build_type,
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
            CMake.defines(args, GLIBCXX_USE_CXX11_ABI=os.getenv('_GLIBCXX_USE_CXX11_ABI'))

        if os.getenv('USE_OPENMP'):
            CMake.defines(args, USE_OPENMP=check_env_flag('USE_OPENMP'))

        if os.getenv('USE_TBB'):
            CMake.defines(args, USE_TBB=check_env_flag('USE_TBB'))

        if os.getenv('MKL_SEQ'):
            CMake.defines(args, USE_MKL_SEQ=check_env_flag('MKL_SEQ'))

        if os.getenv('MKL_TBB'):
            CMake.defines(args, USE_MKL_TBB=check_env_flag('MKL_TBB'))

        mkldnn_threading = os.getenv('MKLDNN_THREADING')
        if mkldnn_threading:
            CMake.defines(args, MKLDNN_THREADING=mkldnn_threading)

        parallel_backend = os.getenv('PARALLEL_BACKEND')
        if parallel_backend:
            CMake.defines(args, PARALLEL_BACKEND=parallel_backend)
        single_thread_pool = os.getenv('EXPERIMENTAL_SINGLE_THREAD_POOL')
        if single_thread_pool:
            CMake.defines(args, EXPERIMENTAL_SINGLE_THREAD_POOL=single_thread_pool)

        if USE_GLOO_IBVERBS:
            CMake.defines(args, USE_IBVERBS="1", USE_GLOO_IBVERBS="1")

        if USE_MKLDNN:
            CMake.defines(args, MKLDNN_ENABLE_CONCURRENT_EXEC="ON")

        expected_wrapper = '/usr/local/opt/ccache/libexec'
        if IS_DARWIN and os.path.exists(expected_wrapper):
            CMake.defines(args,
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
                    print('Invalid ENV[{}] = {}'.format(env_var_name, shex), file=sys.stderr)
                    print(e, file=sys.stderr)
        # According to the CMake manual, we should pass the arguments first,
        # and put the directory as the last element. Otherwise, these flags
        # may not be passed correctly.
        # Reference:
        # 1. https://cmake.org/cmake/help/latest/manual/cmake.1.html#synopsis
        # 2. https://stackoverflow.com/a/27169347
        args.append(base_dir)
        self.run(args, env=my_env)

    def build(self, my_env):
        "Runs cmake to build binaries."

        max_jobs = os.getenv('MAX_JOBS', str(multiprocessing.cpu_count()))
        build_args = ['--build', '.', '--target',
                      'install', '--config', self._build_type]
        # This ``if-else'' clause would be unnecessary when cmake 3.12 becomes
        # minimum, which provides a '-j' option: build_args += ['-j', max_jobs]
        # would be sufficient by then.
        if IS_WINDOWS and not USE_NINJA:  # We are likely using msbuild here
            build_args += ['--', '/maxcpucount:{}'.format(max_jobs)]
        else:
            build_args += ['--', '-j', max_jobs]
        self.run(build_args, my_env)

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
        ninja_build_file = os.path.join(self.build_dir, 'build.ninja')
        if os.path.exists(ninja_build_file):
            os.utime(ninja_build_file, None)
