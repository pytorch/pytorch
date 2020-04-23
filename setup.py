# Welcome to the PyTorch setup.py.
#
# Environment variables you are probably interested in:
#
#   DEBUG
#     build with -O0 and -g (debug symbols)
#
#   REL_WITH_DEB_INFO
#     build with optimizations and -g (debug symbols)
#
#   MAX_JOBS
#     maximum number of compile jobs we should use to compile your code
#
#   USE_CUDA=0
#     disables CUDA build
#
#   CFLAGS
#     flags to apply to both C and C++ files to be compiled (a quirk of setup.py
#     which we have faithfully adhered to in our build system is that CFLAGS
#     also applies to C++ files (unless CXXFLAGS is set), in contrast to the
#     default behavior of autogoo and cmake build systems.)
#
#   CC
#     the C/C++ compiler to use (NB: the CXX flag has no effect for distutils
#     compiles, because distutils always uses CC to compile, even for C++
#     files.
#
# Environment variables for feature toggles:
#
#   USE_CUDNN=0
#     disables the cuDNN build
#
#   USE_FBGEMM=0
#     disables the FBGEMM build
#
#   USE_NUMPY=0
#     disables the NumPy build
#
#   BUILD_TEST=0
#     disables the test build
#
#   USE_MKLDNN=0
#     disables use of MKLDNN
#
#   MKLDNN_CPU_RUNTIME
#     MKL-DNN threading mode: TBB or OMP (default)
#
#   USE_NNPACK=0
#     disables NNPACK build
#
#   USE_QNNPACK=0
#     disables QNNPACK build (quantized 8-bit operators)
#
#   USE_DISTRIBUTED=0
#     disables distributed (c10d, gloo, mpi, etc.) build
#
#   USE_SYSTEM_NCCL=0
#     disables use of system-wide nccl (we will use our submoduled
#     copy in third_party/nccl)
#
#   BUILD_CAFFE2_OPS=0
#     disable Caffe2 operators build
#
#   USE_IBVERBS
#     toggle features related to distributed support
#
#   USE_OPENCV
#     enables use of OpenCV for additional operators
#
#   USE_OPENMP=0
#     disables use of OpenMP for parallelization
#
#   USE_FFMPEG
#     enables use of ffmpeg for additional operators
#
#   USE_LEVELDB
#     enables use of LevelDB for storage
#
#   USE_LMDB
#     enables use of LMDB for storage
#
#   BUILD_BINARY
#     enables the additional binaries/ build
#
#   PYTORCH_BUILD_VERSION
#   PYTORCH_BUILD_NUMBER
#     specify the version of PyTorch, rather than the hard-coded version
#     in this file; used when we're building binaries for distribution
#
#   TORCH_CUDA_ARCH_LIST
#     specify which CUDA architectures to build for.
#     ie `TORCH_CUDA_ARCH_LIST="6.0;7.0"`
#     These are not CUDA versions, instead, they specify what
#     classes of NVIDIA hardware we should generate PTX for.
#
#   ONNX_NAMESPACE
#     specify a namespace for ONNX built here rather than the hard-coded
#     one in this file; needed to build with other frameworks that share ONNX.
#
#   BLAS
#     BLAS to be used by Caffe2. Can be MKL, Eigen, ATLAS, or OpenBLAS. If set
#     then the build will fail if the requested BLAS is not found, otherwise
#     the BLAS will be chosen based on what is found on your system.
#
#   MKL_THREADING
#     MKL threading mode: SEQ, TBB or OMP (default)
#
#   USE_REDIS
#     Whether to use Redis for distributed workflows (Linux only)
#
#   USE_ZSTD
#     Enables use of ZSTD, if the libraries are found
#
# Environment variables we respect (these environment variables are
# conventional and are often understood/set by other software.)
#
#   CUDA_HOME (Linux/OS X)
#   CUDA_PATH (Windows)
#     specify where CUDA is installed; usually /usr/local/cuda or
#     /usr/local/cuda-x.y
#   CUDAHOSTCXX
#     specify a different compiler than the system one to use as the CUDA
#     host compiler for nvcc.
#
#   CUDA_NVCC_EXECUTABLE
#     Specify a NVCC to use. This is used in our CI to point to a cached nvcc
#
#   CUDNN_LIB_DIR
#   CUDNN_INCLUDE_DIR
#   CUDNN_LIBRARY
#     specify where cuDNN is installed
#
#   MIOPEN_LIB_DIR
#   MIOPEN_INCLUDE_DIR
#   MIOPEN_LIBRARY
#     specify where MIOpen is installed
#
#   NCCL_ROOT
#   NCCL_LIB_DIR
#   NCCL_INCLUDE_DIR
#     specify where nccl is installed
#
#   NVTOOLSEXT_PATH (Windows only)
#     specify where nvtoolsext is installed
#
#   LIBRARY_PATH
#   LD_LIBRARY_PATH
#     we will search for libraries in these paths
#
#   ATEN_THREADING
#     ATen parallel backend to use for intra- and inter-op parallelism
#     possible values:
#       OMP - use OpenMP for intra-op and native backend for inter-op tasks
#       NATIVE - use native thread pool for both intra- and inter-op tasks
#       TBB - using TBB for intra- and native thread pool for inter-op parallelism
#
#   USE_TBB
#      enable TBB support
#

from __future__ import print_function

import sys
if sys.version_info < (3,):
    raise Exception("Python 2 has reached end-of-life and is no longer supported by PyTorch.")

from setuptools import setup, Extension, distutils, find_packages
from collections import defaultdict
from distutils import core
from distutils.core import Distribution
from distutils.errors import DistutilsArgError
import setuptools.command.build_ext
import setuptools.command.install
import distutils.command.clean
import distutils.sysconfig
import filecmp
import subprocess
import shutil
import os
import json
import glob
import importlib

from tools.build_pytorch_libs import build_caffe2
from tools.setup_helpers.env import (IS_WINDOWS, IS_DARWIN, IS_LINUX,
                                     check_env_flag, build_type)
from tools.setup_helpers.cmake import CMake

try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError  # Python 2.7 does not have FileNotFoundError

################################################################################
# Parameters parsed from environment
################################################################################

VERBOSE_SCRIPT = True
RUN_BUILD_DEPS = True
# see if the user passed a quiet flag to setup.py arguments and respect
# that in our parts of the build
EMIT_BUILD_WARNING = False
RERUN_CMAKE = False
CMAKE_ONLY = False
filtered_args = []
for i, arg in enumerate(sys.argv):
    if arg == '--cmake':
        RERUN_CMAKE = True
        continue
    if arg == '--cmake-only':
        # Stop once cmake terminates. Leave users a chance to adjust build
        # options.
        CMAKE_ONLY = True
        continue
    if arg == 'rebuild' or arg == 'build':
        arg = 'build'  # rebuild is gone, make it build
        EMIT_BUILD_WARNING = True
    if arg == "--":
        filtered_args += sys.argv[i:]
        break
    if arg == '-q' or arg == '--quiet':
        VERBOSE_SCRIPT = False
    if arg == 'clean' or arg == 'egg_info':
        RUN_BUILD_DEPS = False
    filtered_args.append(arg)
sys.argv = filtered_args

if VERBOSE_SCRIPT:
    def report(*args):
        print(*args)
else:
    def report(*args):
        pass

# Constant known variables used throughout this file
cwd = os.path.dirname(os.path.abspath(__file__))
lib_path = os.path.join(cwd, "torch", "lib")
third_party_path = os.path.join(cwd, "third_party")
caffe2_build_dir = os.path.join(cwd, "build")
# lib/pythonx.x/site-packages
rel_site_packages = distutils.sysconfig.get_python_lib(prefix='')
# full absolute path to the dir above
full_site_packages = distutils.sysconfig.get_python_lib()
# CMAKE: full path to python library
if IS_WINDOWS:
    cmake_python_library = "{}/libs/python{}.lib".format(
        distutils.sysconfig.get_config_var("prefix"),
        distutils.sysconfig.get_config_var("VERSION"))
    # Fix virtualenv builds
    # TODO: Fix for python < 3.3
    if not os.path.exists(cmake_python_library):
        cmake_python_library = "{}/libs/python{}.lib".format(
            sys.base_prefix,
            distutils.sysconfig.get_config_var("VERSION"))
else:
    cmake_python_library = "{}/{}".format(
        distutils.sysconfig.get_config_var("LIBDIR"),
        distutils.sysconfig.get_config_var("INSTSONAME"))
cmake_python_include_dir = distutils.sysconfig.get_python_inc()


################################################################################
# Version, create_version_file, and package_name
################################################################################
package_name = os.getenv('TORCH_PACKAGE_NAME', 'torch')
version = open('version.txt', 'r').read().strip()
sha = 'Unknown'

try:
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=cwd).decode('ascii').strip()
except Exception:
    pass

if os.getenv('PYTORCH_BUILD_VERSION'):
    assert os.getenv('PYTORCH_BUILD_NUMBER') is not None
    build_number = int(os.getenv('PYTORCH_BUILD_NUMBER'))
    version = os.getenv('PYTORCH_BUILD_VERSION')
    if build_number > 1:
        version += '.post' + str(build_number)
elif sha != 'Unknown':
    version += '+' + sha[:7]
report("Building wheel {}-{}".format(package_name, version))

cmake = CMake()

# all the work we need to do _before_ setup runs
def build_deps():
    report('-- Building version ' + version)

    def check_file(f):
        if not os.path.exists(f):
            report("Could not find {}".format(f))
            report("Did you run 'git submodule update --init --recursive'?")
            sys.exit(1)

    check_file(os.path.join(third_party_path, "gloo", "CMakeLists.txt"))
    check_file(os.path.join(third_party_path, "pybind11", "CMakeLists.txt"))
    check_file(os.path.join(third_party_path, 'cpuinfo', 'CMakeLists.txt'))
    check_file(os.path.join(third_party_path, 'tbb', 'Makefile'))
    check_file(os.path.join(third_party_path, 'onnx', 'CMakeLists.txt'))
    check_file(os.path.join(third_party_path, 'foxi', 'CMakeLists.txt'))
    check_file(os.path.join(third_party_path, 'QNNPACK', 'CMakeLists.txt'))
    check_file(os.path.join(third_party_path, 'fbgemm', 'CMakeLists.txt'))
    check_file(os.path.join(third_party_path, 'fbgemm', 'third_party',
                            'asmjit', 'CMakeLists.txt'))
    check_file(os.path.join(third_party_path, 'onnx', 'third_party',
                            'benchmark', 'CMakeLists.txt'))

    check_pydep('yaml', 'pyyaml')
    check_pydep('typing', 'typing')

    build_caffe2(version=version,
                 cmake_python_library=cmake_python_library,
                 build_python=True,
                 rerun_cmake=RERUN_CMAKE,
                 cmake_only=CMAKE_ONLY,
                 cmake=cmake)

    version_path = os.path.join(cwd, 'torch', 'version.py')
    with open(version_path, 'w') as f:
        f.write("__version__ = '{}'\n".format(version))
        # NB: This is not 100% accurate, because you could have built the
        # library code with DEBUG, but csrc without DEBUG (in which case
        # this would claim to be a release build when it's not.)
        f.write("debug = {}\n".format(repr(build_type.is_debug())))
        cmake_cache_vars = defaultdict(lambda: None, cmake.get_cmake_cache_variables())
        f.write("cuda = {}\n".format(repr(cmake_cache_vars['CUDA_VERSION'])))
        f.write("git_version = {}\n".format(repr(sha)))
        f.write("hip = {}\n".format(repr(cmake_cache_vars['HIP_VERSION'])))

    if CMAKE_ONLY:
        report('Finished running cmake. Run "ccmake build" or '
               '"cmake-gui build" to adjust build options and '
               '"python setup.py install" to build.')
        sys.exit()

    # Use copies instead of symbolic files.
    # Windows has very poor support for them.
    sym_files = ['tools/shared/cwrap_common.py', 'tools/shared/_utils_internal.py']
    orig_files = ['aten/src/ATen/common_with_cwrap.py', 'torch/_utils_internal.py']
    for sym_file, orig_file in zip(sym_files, orig_files):
        same = False
        if os.path.exists(sym_file):
            if filecmp.cmp(sym_file, orig_file):
                same = True
            else:
                os.remove(sym_file)
        if not same:
            shutil.copyfile(orig_file, sym_file)

################################################################################
# Building dependent libraries
################################################################################

# the list of runtime dependencies required by this built package
install_requires = ['future']

missing_pydep = '''
Missing build dependency: Unable to `import {importname}`.
Please install it via `conda install {module}` or `pip install {module}`
'''.strip()


def check_pydep(importname, module):
    try:
        importlib.import_module(importname)
    except ImportError:
        raise RuntimeError(missing_pydep.format(importname=importname, module=module))


class build_ext(setuptools.command.build_ext.build_ext):
    def run(self):
        # Report build options. This is run after the build completes so # `CMakeCache.txt` exists and we can get an
        # accurate report on what is used and what is not.
        cmake_cache_vars = defaultdict(lambda: False, cmake.get_cmake_cache_variables())
        if cmake_cache_vars['USE_NUMPY']:
            report('-- Building with NumPy bindings')
        else:
            report('-- NumPy not found')
        if cmake_cache_vars['USE_CUDNN']:
            report('-- Detected cuDNN at ' +
                   cmake_cache_vars['CUDNN_LIBRARY'] + ', ' + cmake_cache_vars['CUDNN_INCLUDE_DIR'])
        else:
            report('-- Not using cuDNN')
        if cmake_cache_vars['USE_CUDA']:
            report('-- Detected CUDA at ' + cmake_cache_vars['CUDA_TOOLKIT_ROOT_DIR'])
        else:
            report('-- Not using CUDA')
        if cmake_cache_vars['USE_MKLDNN']:
            report('-- Using MKLDNN')
            if cmake_cache_vars['USE_MKLDNN_CBLAS']:
                report('-- Using CBLAS in MKLDNN')
            else:
                report('-- Not using CBLAS in MKLDNN')
        else:
            report('-- Not using MKLDNN')
        if cmake_cache_vars['USE_NCCL'] and cmake_cache_vars['USE_SYSTEM_NCCL']:
            report('-- Using system provided NCCL library at {}, {}'.format(cmake_cache_vars['NCCL_LIBRARIES'],
                                                                            cmake_cache_vars['NCCL_INCLUDE_DIRS']))
        elif cmake_cache_vars['USE_NCCL']:
            report('-- Building NCCL library')
        else:
            report('-- Not using NCCL')
        if cmake_cache_vars['USE_DISTRIBUTED']:
            if IS_WINDOWS:
                report('-- Building without distributed package')
            else:
                report('-- Building with distributed package ')
        else:
            report('-- Building without distributed package')

        # Do not use clang to compile exensions if `-fstack-clash-protection` is defined
        # in system CFLAGS
        system_c_flags = distutils.sysconfig.get_config_var('CFLAGS')
        if IS_LINUX and '-fstack-clash-protection' in system_c_flags and 'clang' in os.environ.get('CC', ''):
            os.environ['CC'] = distutils.sysconfig.get_config_var('CC')

        # It's an old-style class in Python 2.7...
        setuptools.command.build_ext.build_ext.run(self)

        # Copy the essential export library to compile C++ extensions.
        if IS_WINDOWS:
            build_temp = self.build_temp

            ext_filename = self.get_ext_filename('_C')
            lib_filename = '.'.join(ext_filename.split('.')[:-1]) + '.lib'

            export_lib = os.path.join(
                build_temp, 'torch', 'csrc', lib_filename).replace('\\', '/')

            build_lib = self.build_lib

            target_lib = os.path.join(
                build_lib, 'torch', 'lib', '_C.lib').replace('\\', '/')

            # Create "torch/lib" directory if not exists.
            # (It is not created yet in "develop" mode.)
            target_dir = os.path.dirname(target_lib)
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)

            self.copy_file(export_lib, target_lib)

    def build_extensions(self):
        self.create_compile_commands()
        # The caffe2 extensions are created in
        # tmp_install/lib/pythonM.m/site-packages/caffe2/python/
        # and need to be copied to build/lib.linux.... , which will be a
        # platform dependent build folder created by the "build" command of
        # setuptools. Only the contents of this folder are installed in the
        # "install" command by default.
        # We only make this copy for Caffe2's pybind extensions
        caffe2_pybind_exts = [
            'caffe2.python.caffe2_pybind11_state',
            'caffe2.python.caffe2_pybind11_state_gpu',
            'caffe2.python.caffe2_pybind11_state_hip',
        ]
        i = 0
        while i < len(self.extensions):
            ext = self.extensions[i]
            if ext.name not in caffe2_pybind_exts:
                i += 1
                continue
            fullname = self.get_ext_fullname(ext.name)
            filename = self.get_ext_filename(fullname)
            report("\nCopying extension {}".format(ext.name))

            src = os.path.join("torch", rel_site_packages, filename)
            if not os.path.exists(src):
                report("{} does not exist".format(src))
                del self.extensions[i]
            else:
                dst = os.path.join(os.path.realpath(self.build_lib), filename)
                report("Copying {} from {} to {}".format(ext.name, src, dst))
                dst_dir = os.path.dirname(dst)
                if not os.path.exists(dst_dir):
                    os.makedirs(dst_dir)
                self.copy_file(src, dst)
                i += 1
        distutils.command.build_ext.build_ext.build_extensions(self)

    def get_outputs(self):
        outputs = distutils.command.build_ext.build_ext.get_outputs(self)
        outputs.append(os.path.join(self.build_lib, "caffe2"))
        report("setup.py::get_outputs returning {}".format(outputs))
        return outputs

    def create_compile_commands(self):
        def load(filename):
            with open(filename) as f:
                return json.load(f)
        ninja_files = glob.glob('build/*compile_commands.json')
        cmake_files = glob.glob('torch/lib/build/*/compile_commands.json')
        all_commands = [entry
                        for f in ninja_files + cmake_files
                        for entry in load(f)]

        # cquery does not like c++ compiles that start with gcc.
        # It forgets to include the c++ header directories.
        # We can work around this by replacing the gcc calls that python
        # setup.py generates with g++ calls instead
        for command in all_commands:
            if command['command'].startswith("gcc "):
                command['command'] = "g++ " + command['command'][4:]

        new_contents = json.dumps(all_commands, indent=2)
        contents = ''
        if os.path.exists('compile_commands.json'):
            with open('compile_commands.json', 'r') as f:
                contents = f.read()
        if contents != new_contents:
            with open('compile_commands.json', 'w') as f:
                f.write(new_contents)


class install(setuptools.command.install.install):
    def run(self):
        setuptools.command.install.install.run(self)


class clean(distutils.command.clean.clean):
    def run(self):
        import glob
        import re
        with open('.gitignore', 'r') as f:
            ignores = f.read()
            pat = re.compile(r'^#( BEGIN NOT-CLEAN-FILES )?')
            for wildcard in filter(None, ignores.split('\n')):
                match = pat.match(wildcard)
                if match:
                    if match.group(1):
                        # Marker is found and stop reading .gitignore.
                        break
                    # Ignore lines which begin with '#'.
                else:
                    for filename in glob.glob(wildcard):
                        try:
                            os.remove(filename)
                        except OSError:
                            shutil.rmtree(filename, ignore_errors=True)

        # It's an old-style class in Python 2.7...
        distutils.command.clean.clean.run(self)

def configure_extension_build():
    r"""Configures extension build options according to system environment and user's choice.

    Returns:
      The input to parameters ext_modules, cmdclass, packages, and entry_points as required in setuptools.setup.
    """

    try:
        cmake_cache_vars = defaultdict(lambda: False, cmake.get_cmake_cache_variables())
    except FileNotFoundError:
        # CMakeCache.txt does not exist. Probably running "python setup.py clean" over a clean directory.
        cmake_cache_vars = defaultdict(lambda: False)

    ################################################################################
    # Configure compile flags
    ################################################################################

    library_dirs = []
    extra_install_requires = []

    if IS_WINDOWS:
        # /NODEFAULTLIB makes sure we only link to DLL runtime
        # and matches the flags set for protobuf and ONNX
        extra_link_args = ['/NODEFAULTLIB:LIBCMT.LIB']
        # /MD links against DLL runtime
        # and matches the flags set for protobuf and ONNX
        # /Z7 turns on symbolic debugging information in .obj files
        # /EHa is about native C++ catch support for asynchronous
        # structured exception handling (SEH)
        # /DNOMINMAX removes builtin min/max functions
        # /wdXXXX disables warning no. XXXX
        extra_compile_args = ['/MD', '/Z7',
                              '/EHa', '/DNOMINMAX',
                              '/wd4267', '/wd4251', '/wd4522', '/wd4522', '/wd4838',
                              '/wd4305', '/wd4244', '/wd4190', '/wd4101', '/wd4996',
                              '/wd4275']
    else:
        extra_link_args = []
        extra_compile_args = [
            '-std=c++14',
            '-Wall',
            '-Wextra',
            '-Wno-strict-overflow',
            '-Wno-unused-parameter',
            '-Wno-missing-field-initializers',
            '-Wno-write-strings',
            '-Wno-unknown-pragmas',
            # This is required for Python 2 declarations that are deprecated in 3.
            '-Wno-deprecated-declarations',
            # Python 2.6 requires -fno-strict-aliasing, see
            # http://legacy.python.org/dev/peps/pep-3123/
            # We also depend on it in our code (even Python 3).
            '-fno-strict-aliasing',
            # Clang has an unfixed bug leading to spurious missing
            # braces warnings, see
            # https://bugs.llvm.org/show_bug.cgi?id=21629
            '-Wno-missing-braces',
        ]
        if check_env_flag('WERROR'):
            extra_compile_args.append('-Werror')

    library_dirs.append(lib_path)

    main_compile_args = []
    main_libraries = ['shm', 'torch_python']
    main_link_args = []
    main_sources = ["torch/csrc/stub.cpp"]

    if cmake_cache_vars['USE_CUDA']:
        library_dirs.append(
            os.path.dirname(cmake_cache_vars['CUDA_CUDA_LIB']))

    if cmake_cache_vars['USE_NUMPY']:
        extra_install_requires += ['numpy']

    if build_type.is_debug():
        if IS_WINDOWS:
            extra_link_args.append('/DEBUG:FULL')
        else:
            extra_compile_args += ['-O0', '-g']
            extra_link_args += ['-O0', '-g']

    if build_type.is_rel_with_deb_info():
        if IS_WINDOWS:
            extra_link_args.append('/DEBUG:FULL')
        else:
            extra_compile_args += ['-g']
            extra_link_args += ['-g']


    def make_relative_rpath(path):
        if IS_DARWIN:
            return '-Wl,-rpath,@loader_path/' + path
        elif IS_WINDOWS:
            return ''
        else:
            return '-Wl,-rpath,$ORIGIN/' + path

    ################################################################################
    # Declare extensions and package
    ################################################################################

    extensions = []
    packages = find_packages(exclude=('tools', 'tools.*'))
    C = Extension("torch._C",
                  libraries=main_libraries,
                  sources=main_sources,
                  language='c++',
                  extra_compile_args=main_compile_args + extra_compile_args,
                  include_dirs=[],
                  library_dirs=library_dirs,
                  extra_link_args=extra_link_args + main_link_args + [make_relative_rpath('lib')])
    extensions.append(C)

    if not IS_WINDOWS:
        DL = Extension("torch._dl",
                       sources=["torch/csrc/dl.c"],
                       language='c')
        extensions.append(DL)

    # These extensions are built by cmake and copied manually in build_extensions()
    # inside the build_ext implementaiton
    extensions.append(
        Extension(
            name=str('caffe2.python.caffe2_pybind11_state'),
            sources=[]),
    )
    if cmake_cache_vars['USE_CUDA']:
        extensions.append(
            Extension(
                name=str('caffe2.python.caffe2_pybind11_state_gpu'),
                sources=[]),
        )
    if cmake_cache_vars['USE_ROCM']:
        extensions.append(
            Extension(
                name=str('caffe2.python.caffe2_pybind11_state_hip'),
                sources=[]),
        )

    cmdclass = {
        'build_ext': build_ext,
        'clean': clean,
        'install': install,
    }

    entry_points = {
        'console_scripts': [
            'convert-caffe2-to-onnx = caffe2.python.onnx.bin.conversion:caffe2_to_onnx',
            'convert-onnx-to-caffe2 = caffe2.python.onnx.bin.conversion:onnx_to_caffe2',
        ]
    }

    return extensions, cmdclass, packages, entry_points, extra_install_requires

# post run, warnings, printed at the end to make them more visible
build_update_message = """
    It is no longer necessary to use the 'build' or 'rebuild' targets

    To install:
      $ python setup.py install
    To develop locally:
      $ python setup.py develop
    To force cmake to re-generate native build files (off by default):
      $ python setup.py develop --cmake
"""


def print_box(msg):
    lines = msg.split('\n')
    size = max(len(l) + 1 for l in lines)
    print('-' * (size + 2))
    for l in lines:
        print('|{}{}|'.format(l, ' ' * (size - len(l))))
    print('-' * (size + 2))

if __name__ == '__main__':
    # Parse the command line and check the arguments
    # before we proceed with building deps and setup
    dist = Distribution()
    dist.script_name = sys.argv[0]
    dist.script_args = sys.argv[1:]
    try:
        ok = dist.parse_command_line()
    except DistutilsArgError as msg:
        raise SystemExit(core.gen_usage(dist.script_name) + "\nerror: %s" % msg)
    if not ok:
        sys.exit()

    if RUN_BUILD_DEPS:
        build_deps()

    extensions, cmdclass, packages, entry_points, extra_install_requires = configure_extension_build()

    install_requires += extra_install_requires

    setup(
        name=package_name,
        version=version,
        description=("Tensors and Dynamic neural networks in "
                     "Python with strong GPU acceleration"),
        ext_modules=extensions,
        cmdclass=cmdclass,
        packages=packages,
        entry_points=entry_points,
        install_requires=install_requires,
        package_data={
            'torch': [
                'py.typed',
                'bin/*',
                'test/*',
                '__init__.pyi',
                'cuda/*.pyi',
                'optim/*.pyi',
                'autograd/*.pyi',
                'utils/data/*.pyi',
                'nn/*.pyi',
                'nn/modules/*.pyi',
                'nn/parallel/*.pyi',
                'lib/*.so*',
                'lib/*.dylib*',
                'lib/*.dll',
                'lib/*.lib',
                'lib/*.pdb',
                'lib/torch_shm_manager',
                'lib/*.h',
                'include/ATen/*.h',
                'include/ATen/cpu/*.h',
                'include/ATen/cpu/vec256/*.h',
                'include/ATen/core/*.h',
                'include/ATen/cuda/*.cuh',
                'include/ATen/cuda/*.h',
                'include/ATen/cuda/detail/*.cuh',
                'include/ATen/cuda/detail/*.h',
                'include/ATen/cudnn/*.h',
                'include/ATen/hip/*.cuh',
                'include/ATen/hip/*.h',
                'include/ATen/hip/detail/*.cuh',
                'include/ATen/hip/detail/*.h',
                'include/ATen/hip/impl/*.h',
                'include/ATen/detail/*.h',
                'include/ATen/native/*.h',
                'include/ATen/native/cpu/*.h',
                'include/ATen/native/quantized/*.h',
                'include/ATen/native/quantized/cpu/*.h',
                'include/ATen/quantized/*.h',
                'include/caffe2/utils/*.h',
                'include/caffe2/utils/**/*.h',
                'include/c10/*.h',
                'include/c10/macros/*.h',
                'include/c10/core/*.h',
                'include/c10/fmt/*.h',
                'include/ATen/core/boxing/*.h',
                'include/ATen/core/boxing/impl/*.h',
                'include/ATen/core/dispatch/*.h',
                'include/ATen/core/op_registration/*.h',
                'include/c10/core/impl/*.h',
                'include/c10/util/*.h',
                'include/c10/cuda/*.h',
                'include/c10/cuda/impl/*.h',
                'include/c10/hip/*.h',
                'include/c10/hip/impl/*.h',
                'include/c10d/*.hpp',
                'include/caffe2/**/*.h',
                'include/torch/*.h',
                'include/torch/csrc/*.h',
                'include/torch/csrc/api/include/torch/*.h',
                'include/torch/csrc/api/include/torch/data/*.h',
                'include/torch/csrc/api/include/torch/data/dataloader/*.h',
                'include/torch/csrc/api/include/torch/data/datasets/*.h',
                'include/torch/csrc/api/include/torch/data/detail/*.h',
                'include/torch/csrc/api/include/torch/data/samplers/*.h',
                'include/torch/csrc/api/include/torch/data/transforms/*.h',
                'include/torch/csrc/api/include/torch/detail/*.h',
                'include/torch/csrc/api/include/torch/detail/ordered_dict.h',
                'include/torch/csrc/api/include/torch/nn/*.h',
                'include/torch/csrc/api/include/torch/nn/functional/*.h',
                'include/torch/csrc/api/include/torch/nn/options/*.h',
                'include/torch/csrc/api/include/torch/nn/modules/*.h',
                'include/torch/csrc/api/include/torch/nn/modules/container/*.h',
                'include/torch/csrc/api/include/torch/nn/parallel/*.h',
                'include/torch/csrc/api/include/torch/nn/utils/*.h',
                'include/torch/csrc/api/include/torch/optim/*.h',
                'include/torch/csrc/api/include/torch/serialize/*.h',
                'include/torch/csrc/autograd/*.h',
                'include/torch/csrc/autograd/functions/*.h',
                'include/torch/csrc/autograd/generated/*.h',
                'include/torch/csrc/autograd/utils/*.h',
                'include/torch/csrc/cuda/*.h',
                'include/torch/csrc/jit/*.h',
                'include/torch/csrc/jit/generated/*.h',
                'include/torch/csrc/jit/passes/*.h',
                'include/torch/csrc/jit/passes/utils/*.h',
                'include/torch/csrc/jit/runtime/*.h',
                'include/torch/csrc/jit/ir/*.h',
                'include/torch/csrc/jit/frontend/*.h',
                'include/torch/csrc/jit/api/*.h',
                'include/torch/csrc/jit/serialization/*.h',
                'include/torch/csrc/jit/python/*.h',
                'include/torch/csrc/jit/testing/*.h',
                'include/torch/csrc/onnx/*.h',
                'include/torch/csrc/utils/*.h',
                'include/pybind11/*.h',
                'include/pybind11/detail/*.h',
                'include/TH/*.h*',
                'include/TH/generic/*.h*',
                'include/THC/*.cuh',
                'include/THC/*.h*',
                'include/THC/generic/*.h',
                'include/THCUNN/*.cuh',
                'include/THCUNN/generic/*.h',
                'include/THH/*.cuh',
                'include/THH/*.h*',
                'include/THH/generic/*.h',
                'share/cmake/ATen/*.cmake',
                'share/cmake/Caffe2/*.cmake',
                'share/cmake/Caffe2/public/*.cmake',
                'share/cmake/Caffe2/Modules_CUDA_fix/*.cmake',
                'share/cmake/Caffe2/Modules_CUDA_fix/upstream/*.cmake',
                'share/cmake/Caffe2/Modules_CUDA_fix/upstream/FindCUDA/*.cmake',
                'share/cmake/Gloo/*.cmake',
                'share/cmake/Torch/*.cmake',
            ],
            'caffe2': [
                'python/serialized_test/data/operator_test/*.zip',
            ],
        },
        url='https://pytorch.org/',
        download_url='https://github.com/pytorch/pytorch/tags',
        author='PyTorch Team',
        author_email='packages@pytorch.org',
        python_requires='>=3.6.1',
        # PyPI package information.
        classifiers=[
            'Development Status :: 5 - Production/Stable',
            'Intended Audience :: Developers',
            'Intended Audience :: Education',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: BSD License',
            'Programming Language :: C++',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Mathematics',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'Topic :: Software Development',
            'Topic :: Software Development :: Libraries',
            'Topic :: Software Development :: Libraries :: Python Modules',
        ],
        license='BSD-3',
        keywords='pytorch machine learning',
    )
    if EMIT_BUILD_WARNING:
        print_box(build_update_message)
