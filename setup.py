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
#   NO_CUDA
#     disables CUDA build
#
#   CFLAGS
#     flags to apply to both C and C++ files to be compiled (a quirk of setup.py
#     which we have faithfully adhered to in our build system is that CFLAGS
#     also applies to C++ files, in contrast to the default behavior of autogoo
#     and cmake build systems.)
#
#   CC
#     the C/C++ compiler to use (NB: the CXX flag has no effect for distutils
#     compiles, because distutils always uses CC to compile, even for C++
#     files.
#
# Environment variables for feature toggles:
#
#   NO_CUDNN
#     disables the cuDNN build
#
#   NO_FBGEMM
#     disables the FBGEMM build
#
#   NO_TEST
#     disables the test build
#
#   NO_MIOPEN
#     disables the MIOpen build
#
#   NO_MKLDNN
#     disables use of MKLDNN
#
#   NO_NNPACK
#     disables NNPACK build
#
#   NO_QNNPACK
#     disables QNNPACK build (quantized 8-bit operators)
#
#   NO_DISTRIBUTED
#     disables distributed (c10d, gloo, mpi, etc.) build
#
#   NO_SYSTEM_NCCL
#     disables use of system-wide nccl (we will use our submoduled
#     copy in third_party/nccl)
#
#   NO_CAFFE2_OPS
#     disable Caffe2 operators build
#
#   USE_GLOO_IBVERBS
#     toggle features related to distributed support
#
#   USE_OPENCV
#     enables use of OpenCV for additional operators
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
#   USE_FBGEMM
#     Enables use of FBGEMM
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
#   NCCL_ROOT_DIR
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

from __future__ import print_function
from setuptools import setup, Extension, distutils, Command, find_packages
import setuptools.command.build_ext
import setuptools.command.install
import setuptools.command.develop
import setuptools.command.build_py
import distutils.unixccompiler
import distutils.command.build
import distutils.command.clean
import distutils.sysconfig
import filecmp
import platform
import subprocess
import shutil
import multiprocessing
import sys
import os
import json
import glob
import importlib

# If you want to modify flags or environmental variables that is set when
# building torch, you should do it in tools/setup_helpers/configure.py.
# Please don't add it here unless it's only used in PyTorch.
from tools.setup_helpers.configure import *
from tools.setup_helpers.generate_code import generate_code
from tools.setup_helpers.ninja_builder import NinjaBuilder, ninja_build_ext
import tools.setup_helpers.configure

################################################################################
# Parameters parsed from environment
################################################################################

VERBOSE_SCRIPT = True
# see if the user passed a quiet flag to setup.py arguments and respect
# that in our parts of the build
for arg in sys.argv:
    if arg == "--":
        break
    if arg == '-q' or arg == '--quiet':
        VERBOSE_SCRIPT = False

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
tmp_install_path = lib_path + "/tmp_install"
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
else:
    cmake_python_library = "{}/{}".format(
        distutils.sysconfig.get_config_var("LIBDIR"),
        distutils.sysconfig.get_config_var("INSTSONAME"))
cmake_python_include_dir = distutils.sysconfig.get_python_inc()


class PytorchCommand(setuptools.Command):
    """
    Base Pytorch command to avoid implementing initialize/finalize_options in
    every subclass
    """
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass


################################################################################
# Version, create_version_file, and package_name
################################################################################
package_name = os.getenv('TORCH_PACKAGE_NAME', 'torch')
version = '1.1.0a0'
if os.getenv('PYTORCH_BUILD_VERSION'):
    assert os.getenv('PYTORCH_BUILD_NUMBER') is not None
    build_number = int(os.getenv('PYTORCH_BUILD_NUMBER'))
    version = os.getenv('PYTORCH_BUILD_VERSION')
    if build_number > 1:
        version += '.post' + str(build_number)
else:
    try:
        sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=cwd).decode('ascii').strip()
        version += '+' + sha[:7]
    except Exception:
        pass
report("Building wheel {}-{}".format(package_name, version))


class create_version_file(PytorchCommand):
    def run(self):
        global version, cwd
        report('-- Building version ' + version)
        version_path = os.path.join(cwd, 'torch', 'version.py')
        with open(version_path, 'w') as f:
            f.write("__version__ = '{}'\n".format(version))
            # NB: This is not 100% accurate, because you could have built the
            # library code with DEBUG, but csrc without DEBUG (in which case
            # this would claim to be a release build when it's not.)
            f.write("debug = {}\n".format(repr(DEBUG)))
            f.write("cuda = {}\n".format(repr(CUDA_VERSION)))


################################################################################
# Building dependent libraries
################################################################################

# All libraries that torch could depend on
dep_libs = ['caffe2']

missing_pydep = '''
Missing build dependency: Unable to `import {importname}`.
Please install it via `conda install {module}` or `pip install {module}`
'''.strip()


def check_pydep(importname, module):
    try:
        importlib.import_module(importname)
    except ImportError:
        raise RuntimeError(missing_pydep.format(importname=importname, module=module))


# Calls build_pytorch_libs.sh/bat with the correct env variables
def build_libs(libs):
    for lib in libs:
        assert lib in dep_libs, 'invalid lib: {}'.format(lib)
    if IS_WINDOWS:
        build_libs_cmd = ['tools\\build_pytorch_libs.bat']
    else:
        build_libs_cmd = ['bash', os.path.join('..', 'tools', 'build_pytorch_libs.sh')]

    my_env, extra_flags = get_pytorch_env_with_flags()
    build_libs_cmd.extend(extra_flags)
    my_env["PYTORCH_PYTHON_LIBRARY"] = cmake_python_library
    my_env["PYTORCH_PYTHON_INCLUDE_DIR"] = cmake_python_include_dir
    my_env["PYTORCH_BUILD_VERSION"] = version

    cmake_prefix_path = full_site_packages
    if "CMAKE_PREFIX_PATH" in my_env:
        cmake_prefix_path = my_env["CMAKE_PREFIX_PATH"] + ";" + cmake_prefix_path
    my_env["CMAKE_PREFIX_PATH"] = cmake_prefix_path

    if VERBOSE_SCRIPT:
        my_env['VERBOSE_SCRIPT'] = '1'
    try:
        os.mkdir('build')
    except OSError:
        pass

    kwargs = {'cwd': 'build'} if not IS_WINDOWS else {}

    if subprocess.call(build_libs_cmd + libs, env=my_env, **kwargs) != 0:
        report("Failed to run '{}'".format(' '.join(build_libs_cmd + libs)))
        sys.exit(1)


# Build all dependent libraries
class build_deps(PytorchCommand):
    def run(self):
        report('setup.py::build_deps::run()')
        # Check if you remembered to check out submodules

        def check_file(f):
            if not os.path.exists(f):
                report("Could not find {}".format(f))
                report("Did you run 'git submodule update --init --recursive'?")
                sys.exit(1)

        check_file(os.path.join(third_party_path, "gloo", "CMakeLists.txt"))
        check_file(os.path.join(third_party_path, "pybind11", "CMakeLists.txt"))
        check_file(os.path.join(third_party_path, 'cpuinfo', 'CMakeLists.txt'))
        check_file(os.path.join(third_party_path, 'onnx', 'CMakeLists.txt'))
        check_file(os.path.join(third_party_path, 'QNNPACK', 'CMakeLists.txt'))
        check_file(os.path.join(third_party_path, 'fbgemm', 'CMakeLists.txt'))

        check_pydep('yaml', 'pyyaml')
        check_pydep('typing', 'typing')

        libs = []
        libs += ['caffe2']
        build_libs(libs)

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

        self.copy_tree('torch/lib/tmp_install/share', 'torch/share')
        self.copy_tree('third_party/pybind11/include/pybind11/',
                       'torch/lib/include/pybind11')


build_dep_cmds = {}
rebuild_dep_cmds = {}

for lib in dep_libs:
    # wrap in function to capture lib
    class build_dep(build_deps):
        description = 'Build {} external library'.format(lib)

        def run(self):
            build_libs([self.lib])
    build_dep.lib = lib
    build_dep_cmds['build_' + lib.lower()] = build_dep

    class rebuild_dep(build_deps):
        description = 'Rebuild {} external library'.format(lib)

        def run(self):
            tools.setup_helpers.configure.RERUN_CMAKE = False
            build_libs([self.lib])
    rebuild_dep.lib = lib
    rebuild_dep_cmds['rebuild_' + lib.lower()] = rebuild_dep


class build_module(PytorchCommand):
    def run(self):
        report('setup.py::build_module::run()')
        self.run_command('build_py')
        self.run_command('build_ext')


class build_py(setuptools.command.build_py.build_py):

    def run(self):
        report('setup.py::build_py::run()')
        self.run_command('create_version_file')
        setuptools.command.build_py.build_py.run(self)


class develop(setuptools.command.develop.develop):

    def run(self):
        report('setup.py::develop::run()')
        self.run_command('create_version_file')
        setuptools.command.develop.develop.run(self)
        self.create_compile_commands()

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

        if not USE_NINJA:
            report("WARNING: 'develop' is not building C++ code incrementally")
            report("because ninja is not installed. Run this to enable it:")
            report(" > pip install ninja")


build_ext_parent = ninja_build_ext if USE_NINJA \
    else setuptools.command.build_ext.build_ext


class build_ext(build_ext_parent):

    def run(self):
        # report build options
        if USE_NUMPY:
            report('-- Building with NumPy bindings')
        else:
            report('-- NumPy not found')
        if USE_CUDNN:
            report('-- Detected cuDNN at ' + CUDNN_LIBRARY + ', ' + CUDNN_INCLUDE_DIR)
        else:
            report('-- Not using cuDNN')
        if USE_MIOPEN:
            report('-- Detected MIOpen at ' + MIOPEN_LIBRARY + ', ' + MIOPEN_INCLUDE_DIR)
        else:
            report('-- Not using MIOpen')
        if USE_CUDA:
            report('-- Detected CUDA at ' + CUDA_HOME)
        else:
            report('-- Not using CUDA')
        if USE_MKLDNN:
            report('-- Using MKLDNN')
        else:
            report('-- Not using MKLDNN')
        if USE_NCCL and USE_SYSTEM_NCCL:
            report('-- Using system provided NCCL library at ' + NCCL_SYSTEM_LIB + ', ' + NCCL_INCLUDE_DIR)
        elif USE_NCCL:
            report('-- Building NCCL library')
        else:
            report('-- Not using NCCL')
        if USE_DISTRIBUTED:
            report('-- Building with THD distributed package ')
            if IS_LINUX:
                report('-- Building with c10d distributed package ')
            else:
                report('-- Building without c10d distributed package')
        else:
            report('-- Building without distributed package')

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

            self.copy_file(export_lib, target_lib)

    def build_extensions(self):
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

            src = os.path.join(tmp_install_path, rel_site_packages, filename)
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


class build(distutils.command.build.build):
    sub_commands = [
        ('build_deps', lambda self: True),
    ] + distutils.command.build.build.sub_commands


class rebuild(distutils.command.build.build):
    sub_commands = [
        ('build_deps', lambda self: True),
    ] + distutils.command.build.build.sub_commands

    def run(self):
        tools.setup_helpers.configure.RERUN_CMAKE = False
        distutils.command.build.build.run(self)


class install(setuptools.command.install.install):

    def run(self):
        report('setup.py::run()')
        if not self.skip_build:
            self.run_command('build_deps')

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


################################################################################
# Configure compile flags
################################################################################

library_dirs = []

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
    if sys.version_info[0] == 2:
        if not check_env_flag('FORCE_PY27_BUILD'):
            report('The support for PyTorch with Python 2.7 on Windows is very experimental.')
            report('Please set the flag `FORCE_PY27_BUILD` to 1 to continue build.')
            sys.exit(1)
        # /bigobj increases number of sections in .obj file, which is needed to link
        # against libaries in Python 2.7 under Windows
        extra_compile_args.append('/bigobj')
else:
    extra_link_args = []
    extra_compile_args = [
        '-std=c++11',
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

# we specify exact lib names to avoid conflict with lua-torch installs
CAFFE2_LIBS = []
if USE_CUDA:
    CAFFE2_LIBS.extend(['-Wl,--no-as-needed', os.path.join(lib_path, 'libcaffe2_gpu.so'), '-Wl,--as-needed'])
if USE_ROCM:
    CAFFE2_LIBS.extend(['-Wl,--no-as-needed', os.path.join(lib_path, 'libcaffe2_hip.so'), '-Wl,--as-needed'])

# static library only
if IS_DARWIN:
    CAFFE2_LIBS = []
    if USE_CUDA:
        CAFFE2_LIBS.append(os.path.join(lib_path, 'libcaffe2_gpu.dylib'))
    if USE_ROCM:
        CAFFE2_LIBS.append(os.path.join(lib_path, 'libcaffe2_hip.dylib'))

if IS_WINDOWS:
    CAFFE2_LIBS = []
    if USE_CUDA:
        CAFFE2_LIBS.append(os.path.join(lib_path, 'caffe2_gpu.lib'))
    if USE_ROCM:
        CAFFE2_LIBS.append(os.path.join(lib_path, 'caffe2_hip.lib'))

main_compile_args = ['-D_THP_CORE', '-DONNX_NAMESPACE=' + ONNX_NAMESPACE]
main_libraries = ['shm', 'torch_python']
main_link_args = []
main_sources = ["torch/csrc/stub.cpp"]

# Before the introduction of stub.cpp, _C.so and libcaffe2.so defined
# some of the same symbols, and it was important for _C.so to be
# loaded before libcaffe2.so so that the versions in _C.so got
# used. This happened automatically because we loaded _C.so directly,
# and libcaffe2.so was brought in as a dependency (though I suspect it
# may have been possible to break by importing caffe2 first in the
# same process).
#
# Now, libtorch_python.so and libcaffe2.so define some of the same
# symbols. We directly load the _C.so stub, which brings both of these
# in as dependencies. We have to make sure that symbols continue to be
# looked up in libtorch_python.so first, by making sure it comes
# before libcaffe2.so in the linker command.
main_link_args.extend(CAFFE2_LIBS)

try:
    import numpy as np
    NUMPY_INCLUDE_DIR = np.get_include()
    USE_NUMPY = True
except ImportError:
    USE_NUMPY = False

if USE_CUDA:
    if IS_WINDOWS:
        cuda_lib_path = CUDA_HOME + '/lib/x64/'
    else:
        cuda_lib_dirs = ['lib64', 'lib']
        for lib_dir in cuda_lib_dirs:
            cuda_lib_path = os.path.join(CUDA_HOME, lib_dir)
            if os.path.exists(cuda_lib_path):
                break
    library_dirs.append(cuda_lib_path)

if DEBUG:
    if IS_WINDOWS:
        extra_link_args.append('/DEBUG:FULL')
    else:
        extra_compile_args += ['-O0', '-g']
        extra_link_args += ['-O0', '-g']

if REL_WITH_DEB_INFO:
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
              extra_link_args=extra_link_args + main_link_args + [make_relative_rpath('lib')],
              )
extensions.append(C)

if not IS_WINDOWS:
    DL = Extension("torch._dl",
                   sources=["torch/csrc/dl.c"],
                   language='c'
                   )
    extensions.append(DL)


if USE_CUDA:
    thnvrtc_link_flags = extra_link_args + [make_relative_rpath('lib')]
    if IS_LINUX:
        thnvrtc_link_flags = thnvrtc_link_flags + ['-Wl,--no-as-needed']
    # these have to be specified as -lcuda in link_flags because they
    # have to come right after the `no-as-needed` option
    if IS_WINDOWS:
        thnvrtc_link_flags += ['cuda.lib', 'nvrtc.lib']
    else:
        thnvrtc_link_flags += ['-lcuda', '-lnvrtc']
    cuda_stub_path = [cuda_lib_path + '/stubs']
    if IS_DARWIN:
        # on macOS this is where the CUDA stub is installed according to the manual
        cuda_stub_path = ["/usr/local/cuda/lib"]
    THNVRTC = Extension("torch._nvrtc",
                        sources=['torch/csrc/nvrtc.cpp'],
                        language='c++',
                        extra_compile_args=main_compile_args + extra_compile_args,
                        include_dirs=[cwd],
                        library_dirs=library_dirs + cuda_stub_path,
                        extra_link_args=thnvrtc_link_flags,
                        )
    extensions.append(THNVRTC)

# These extensions are built by cmake and copied manually in build_extensions()
# inside the build_ext implementaiton
extensions.append(
    Extension(
        name=str('caffe2.python.caffe2_pybind11_state'),
        sources=[]),
)
if USE_CUDA:
    extensions.append(
        Extension(
            name=str('caffe2.python.caffe2_pybind11_state_gpu'),
            sources=[]),
    )
if USE_ROCM:
    extensions.append(
        Extension(
            name=str('caffe2.python.caffe2_pybind11_state_hip'),
            sources=[]),
    )

cmdclass = {
    'create_version_file': create_version_file,
    'build': build,
    'build_py': build_py,
    'build_ext': build_ext,
    'build_deps': build_deps,
    'build_module': build_module,
    'rebuild': rebuild,
    'develop': develop,
    'install': install,
    'clean': clean,
}
cmdclass.update(build_dep_cmds)
cmdclass.update(rebuild_dep_cmds)

entry_points = {
    'console_scripts': [
        'convert-caffe2-to-onnx = caffe2.python.onnx.bin.conversion:caffe2_to_onnx',
        'convert-onnx-to-caffe2 = caffe2.python.onnx.bin.conversion:onnx_to_caffe2',
    ]
}

if __name__ == '__main__':
    setup(
        name=package_name,
        version=version,
        description=("Tensors and Dynamic neural networks in "
                     "Python with strong GPU acceleration"),
        ext_modules=extensions,
        cmdclass=cmdclass,
        packages=packages,
        entry_points=entry_points,
        package_data={
            'torch': [
                'lib/*.so*',
                'lib/*.dylib*',
                'lib/*.dll',
                'lib/*.lib',
                'lib/*.pdb',
                'lib/torch_shm_manager',
                'lib/*.h',
                'lib/include/ATen/*.h',
                'lib/include/ATen/cpu/*.h',
                'lib/include/ATen/core/*.h',
                'lib/include/ATen/cuda/*.cuh',
                'lib/include/ATen/cuda/*.h',
                'lib/include/ATen/cuda/detail/*.cuh',
                'lib/include/ATen/cuda/detail/*.h',
                'lib/include/ATen/cudnn/*.h',
                'lib/include/ATen/detail/*.h',
                'lib/include/caffe2/utils/*.h',
                'lib/include/c10/*.h',
                'lib/include/c10/macros/*.h',
                'lib/include/c10/core/*.h',
                'lib/include/ATen/core/dispatch/*.h',
                'lib/include/c10/core/impl/*.h',
                'lib/include/ATen/core/opschema/*.h',
                'lib/include/c10/util/*.h',
                'lib/include/c10/cuda/*.h',
                'lib/include/c10/cuda/impl/*.h',
                'lib/include/c10/hip/*.h',
                'lib/include/c10/hip/impl/*.h',
                'lib/include/caffe2/**/*.h',
                'lib/include/torch/*.h',
                'lib/include/torch/csrc/*.h',
                'lib/include/torch/csrc/api/include/torch/*.h',
                'lib/include/torch/csrc/api/include/torch/data/*.h',
                'lib/include/torch/csrc/api/include/torch/data/dataloader/*.h',
                'lib/include/torch/csrc/api/include/torch/data/datasets/*.h',
                'lib/include/torch/csrc/api/include/torch/data/detail/*.h',
                'lib/include/torch/csrc/api/include/torch/data/samplers/*.h',
                'lib/include/torch/csrc/api/include/torch/data/transforms/*.h',
                'lib/include/torch/csrc/api/include/torch/detail/*.h',
                'lib/include/torch/csrc/api/include/torch/detail/ordered_dict.h',
                'lib/include/torch/csrc/api/include/torch/nn/*.h',
                'lib/include/torch/csrc/api/include/torch/nn/modules/*.h',
                'lib/include/torch/csrc/api/include/torch/nn/parallel/*.h',
                'lib/include/torch/csrc/api/include/torch/optim/*.h',
                'lib/include/torch/csrc/api/include/torch/serialize/*.h',
                'lib/include/torch/csrc/autograd/*.h',
                'lib/include/torch/csrc/autograd/generated/*.h',
                'lib/include/torch/csrc/cuda/*.h',
                'lib/include/torch/csrc/jit/*.h',
                'lib/include/torch/csrc/jit/generated/*.h',
                'lib/include/torch/csrc/jit/passes/*.h',
                'lib/include/torch/csrc/jit/script/*.h',
                'lib/include/torch/csrc/utils/*.h',
                'lib/include/pybind11/*.h',
                'lib/include/pybind11/detail/*.h',
                'lib/include/TH/*.h*',
                'lib/include/TH/generic/*.h*',
                'lib/include/THC/*.cuh',
                'lib/include/THC/*.h*',
                'lib/include/THC/generic/*.h',
                'lib/include/THCUNN/*.cuh',
                'lib/include/THNN/*.h',
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
                'cpp_test/*',
                'python/serialized_test/data/operator_test/*.zip',
            ]
        },
    )
