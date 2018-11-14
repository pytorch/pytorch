# Welcome to the PyTorch setup.py.
#
# Environment variables you are probably interested in:
#
#   DEBUG
#     build with -O0 and -g (debug symbols)
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

from tools.setup_helpers.env import check_env_flag, check_negative_env_flag


def hotpatch_var(var, prefix='USE_'):
    if check_env_flag('NO_' + var):
        os.environ[prefix + var] = '0'
    elif check_negative_env_flag('NO_' + var):
        os.environ[prefix + var] = '1'
    elif check_env_flag('WITH_' + var):
        os.environ[prefix + var] = '1'
    elif check_negative_env_flag('WITH_' + var):
        os.environ[prefix + var] = '0'

# Before we run the setup_helpers, let's look for NO_* and WITH_*
# variables and hotpatch environment with the USE_* equivalent
use_env_vars = ['CUDA', 'CUDNN', 'FBGEMM', 'MIOPEN', 'MKLDNN', 'NNPACK', 'DISTRIBUTED',
                'OPENCV', 'QNNPACK', 'FFMPEG', 'SYSTEM_NCCL', 'GLOO_IBVERBS']
list(map(hotpatch_var, use_env_vars))

# Also hotpatch a few with BUILD_* equivalent
build_env_vars = ['BINARY', 'TEST', 'CAFFE2_OPS']
[hotpatch_var(v, 'BUILD_') for v in build_env_vars]

from tools.setup_helpers.cuda import USE_CUDA, CUDA_HOME, CUDA_VERSION
from tools.setup_helpers.build import (BUILD_BINARY, BUILD_TEST,
                                       BUILD_CAFFE2_OPS, USE_LEVELDB,
                                       USE_LMDB, USE_OPENCV, USE_FFMPEG)
from tools.setup_helpers.rocm import USE_ROCM, ROCM_HOME, ROCM_VERSION
from tools.setup_helpers.cudnn import (USE_CUDNN, CUDNN_LIBRARY,
                                       CUDNN_LIB_DIR, CUDNN_INCLUDE_DIR)
from tools.setup_helpers.fbgemm import USE_FBGEMM
from tools.setup_helpers.miopen import (USE_MIOPEN, MIOPEN_LIBRARY,
                                        MIOPEN_LIB_DIR, MIOPEN_INCLUDE_DIR)
from tools.setup_helpers.nccl import USE_NCCL, USE_SYSTEM_NCCL, NCCL_LIB_DIR, \
    NCCL_INCLUDE_DIR, NCCL_ROOT_DIR, NCCL_SYSTEM_LIB
from tools.setup_helpers.nnpack import USE_NNPACK
from tools.setup_helpers.qnnpack import USE_QNNPACK
from tools.setup_helpers.nvtoolext import NVTOOLEXT_HOME
from tools.setup_helpers.generate_code import generate_code
from tools.setup_helpers.ninja_builder import NinjaBuilder, ninja_build_ext
from tools.setup_helpers.dist_check import USE_DISTRIBUTED, \
    USE_GLOO_IBVERBS

################################################################################
# Parameters parsed from environment
################################################################################

DEBUG = check_env_flag('DEBUG')
IS_WINDOWS = (platform.system() == 'Windows')
IS_DARWIN = (platform.system() == 'Darwin')
IS_LINUX = (platform.system() == 'Linux')
IS_PPC = (platform.machine() == 'ppc64le')

BUILD_PYTORCH = check_env_flag('BUILD_PYTORCH')
# ppc64le does not support MKLDNN
if IS_PPC:
    USE_MKLDNN = check_env_flag('USE_MKLDNN', 'OFF')
else:
    USE_MKLDNN = check_env_flag('USE_MKLDNN', 'ON')

USE_CUDA_STATIC_LINK = check_env_flag('USE_CUDA_STATIC_LINK')
RERUN_CMAKE = True

NUM_JOBS = multiprocessing.cpu_count()
max_jobs = os.getenv("MAX_JOBS")
if max_jobs is not None:
    NUM_JOBS = min(NUM_JOBS, int(max_jobs))

ONNX_NAMESPACE = os.getenv("ONNX_NAMESPACE")
if not ONNX_NAMESPACE:
    ONNX_NAMESPACE = "onnx_torch"

# Ninja
try:
    import ninja
    USE_NINJA = True
except ImportError:
    USE_NINJA = False

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
# Patches and workarounds
################################################################################
# Monkey-patch setuptools to compile in parallel
if not USE_NINJA:
    def parallelCCompile(self, sources, output_dir=None, macros=None,
                         include_dirs=None, debug=0, extra_preargs=None,
                         extra_postargs=None, depends=None):
        # those lines are copied from distutils.ccompiler.CCompiler directly
        macros, objects, extra_postargs, pp_opts, build = self._setup_compile(
            output_dir, macros, include_dirs, sources, depends, extra_postargs)
        cc_args = self._get_cc_args(pp_opts, debug, extra_preargs)

        # compile using a thread pool
        import multiprocessing.pool

        def _single_compile(obj):
            src, ext = build[obj]
            self._compile(obj, src, ext, cc_args, extra_postargs, pp_opts)
        multiprocessing.pool.ThreadPool(NUM_JOBS).map(_single_compile, objects)

        return objects
    distutils.ccompiler.CCompiler.compile = parallelCCompile

# Patch for linking with ccache
original_link = distutils.unixccompiler.UnixCCompiler.link


def patched_link(self, *args, **kwargs):
    _cxx = self.compiler_cxx
    self.compiler_cxx = None
    result = original_link(self, *args, **kwargs)
    self.compiler_cxx = _cxx
    return result

distutils.unixccompiler.UnixCCompiler.link = patched_link

# Workaround setuptools -Wstrict-prototypes warnings
# I lifted this code from https://stackoverflow.com/a/29634231/23845
cfg_vars = distutils.sysconfig.get_config_vars()
for key, value in cfg_vars.items():
    if type(value) == str:
        cfg_vars[key] = value.replace("-Wstrict-prototypes", "")


################################################################################
# Version, create_version_file, and package_name
################################################################################
package_name = os.getenv('TORCH_PACKAGE_NAME', 'torch')
version = '1.0.0a0'
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
print("Building wheel {}-{}".format(package_name, version))


class create_version_file(PytorchCommand):
    def run(self):
        global version, cwd
        print('-- Building version ' + version)
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
    my_env = os.environ.copy()
    my_env["PYTORCH_PYTHON"] = sys.executable
    my_env["PYTORCH_PYTHON_LIBRARY"] = cmake_python_library
    my_env["PYTORCH_PYTHON_INCLUDE_DIR"] = cmake_python_include_dir
    my_env["PYTORCH_BUILD_VERSION"] = version

    cmake_prefix_path = full_site_packages
    if "CMAKE_PREFIX_PATH" in my_env:
        cmake_prefix_path = my_env["CMAKE_PREFIX_PATH"] + ";" + cmake_prefix_path
    my_env["CMAKE_PREFIX_PATH"] = cmake_prefix_path

    my_env["NUM_JOBS"] = str(NUM_JOBS)
    my_env["ONNX_NAMESPACE"] = ONNX_NAMESPACE
    if not IS_WINDOWS:
        if USE_NINJA:
            my_env["CMAKE_GENERATOR"] = '-GNinja'
            my_env["CMAKE_INSTALL"] = 'ninja install'
        else:
            my_env['CMAKE_GENERATOR'] = ''
            my_env['CMAKE_INSTALL'] = 'make install'
    if USE_SYSTEM_NCCL:
        my_env["NCCL_ROOT_DIR"] = NCCL_ROOT_DIR
    if USE_CUDA:
        my_env["CUDA_BIN_PATH"] = CUDA_HOME
        build_libs_cmd += ['--use-cuda']
        if IS_WINDOWS:
            my_env["NVTOOLEXT_HOME"] = NVTOOLEXT_HOME
    if USE_CUDA_STATIC_LINK:
        build_libs_cmd += ['--cuda-static-link']
    if USE_FBGEMM:
        build_libs_cmd += ['--use-fbgemm']
    if USE_ROCM:
        build_libs_cmd += ['--use-rocm']
    if USE_NNPACK:
        build_libs_cmd += ['--use-nnpack']
    if USE_CUDNN:
        my_env["CUDNN_LIB_DIR"] = CUDNN_LIB_DIR
        my_env["CUDNN_LIBRARY"] = CUDNN_LIBRARY
        my_env["CUDNN_INCLUDE_DIR"] = CUDNN_INCLUDE_DIR
    if USE_MIOPEN:
        my_env["MIOPEN_LIB_DIR"] = MIOPEN_LIB_DIR
        my_env["MIOPEN_LIBRARY"] = MIOPEN_LIBRARY
        my_env["MIOPEN_INCLUDE_DIR"] = MIOPEN_INCLUDE_DIR
    if USE_MKLDNN:
        build_libs_cmd += ['--use-mkldnn']
    if USE_QNNPACK:
        build_libs_cmd += ['--use-qnnpack']
    if USE_GLOO_IBVERBS:
        build_libs_cmd += ['--use-gloo-ibverbs']
    if not RERUN_CMAKE:
        build_libs_cmd += ['--dont-rerun-cmake']

    my_env["BUILD_TORCH"] = "ON"
    my_env["BUILD_PYTHON"] = "ON"
    my_env["BUILD_BINARY"] = "ON" if BUILD_BINARY else "OFF"
    my_env["BUILD_TEST"] = "ON" if BUILD_TEST else "OFF"
    my_env["BUILD_CAFFE2_OPS"] = "ON" if BUILD_CAFFE2_OPS else "OFF"
    my_env["INSTALL_TEST"] = "ON" if BUILD_TEST else "OFF"
    my_env["USE_LEVELDB"] = "ON" if USE_LEVELDB else "OFF"
    my_env["USE_LMDB"] = "ON" if USE_LMDB else "OFF"
    my_env["USE_OPENCV"] = "ON" if USE_OPENCV else "OFF"
    my_env["USE_FFMPEG"] = "ON" if USE_FFMPEG else "OFF"
    my_env["USE_DISTRIBUTED"] = "ON" if USE_DISTRIBUTED else "OFF"

    try:
        os.mkdir('build')
    except OSError:
        pass

    kwargs = {'cwd': 'build'} if not IS_WINDOWS else {}

    if subprocess.call(build_libs_cmd + libs, env=my_env, **kwargs) != 0:
        print("Failed to run '{}'".format(' '.join(build_libs_cmd + libs)))
        sys.exit(1)


# Copy Caffe2's Python proto files (generated during the build with the
# protobuf python compiler) from the build folder to the root folder
# cp root/build/caffe2/proto/proto.py root/caffe2/proto/proto.py
def copy_protos():
    print('setup.py::copy_protos()')
    for src in glob.glob(
            os.path.join(caffe2_build_dir, 'caffe2', 'proto', '*.py')):
        dst = os.path.join(
            cwd, os.path.relpath(src, caffe2_build_dir))
        shutil.copyfile(src, dst)


# Build all dependent libraries
class build_deps(PytorchCommand):
    def run(self):
        print('setup.py::build_deps::run()')
        # Check if you remembered to check out submodules

        def check_file(f):
            if not os.path.exists(f):
                print("Could not find {}".format(f))
                print("Did you run 'git submodule update --init --recursive'?")
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
            global RERUN_CMAKE
            RERUN_CMAKE = False
            build_libs([self.lib])
    rebuild_dep.lib = lib
    rebuild_dep_cmds['rebuild_' + lib.lower()] = rebuild_dep


class build_module(PytorchCommand):
    def run(self):
        print('setup.py::build_module::run()')
        self.run_command('build_py')
        self.run_command('build_ext')


class build_py(setuptools.command.build_py.build_py):

    def run(self):
        print('setup.py::build_py::run()')
        self.run_command('create_version_file')
        setuptools.command.build_py.build_py.run(self)


class develop(setuptools.command.develop.develop):

    def run(self):
        print('setup.py::develop::run()')
        self.run_command('create_version_file')
        setuptools.command.develop.develop.run(self)
        self.create_compile_commands()
        copy_protos()

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
            print("WARNING: 'develop' is not building C++ code incrementally")
            print("because ninja is not installed. Run this to enable it:")
            print(" > pip install ninja")


def monkey_patch_C10D_inc_flags():
    '''
    C10D's include deps are not determined until after build c10d is run, so
    we need to monkey-patch it.
    '''
    mpi_include_path_file = tmp_install_path + "/include/c10d/mpi_include_path"
    if os.path.exists(mpi_include_path_file):
        with open(mpi_include_path_file, 'r') as f:
            mpi_include_paths = f.readlines()
        mpi_include_paths = [p.strip() for p in mpi_include_paths]
        C.include_dirs += mpi_include_paths
        print("-- For c10d, will include MPI paths: {}".format(mpi_include_paths))


build_ext_parent = ninja_build_ext if USE_NINJA \
    else setuptools.command.build_ext.build_ext


class build_ext(build_ext_parent):

    def run(self):
        # Print build options
        if USE_NUMPY:
            print('-- Building with NumPy bindings')
        else:
            print('-- NumPy not found')
        if USE_CUDNN:
            print('-- Detected cuDNN at ' + CUDNN_LIBRARY + ', ' + CUDNN_INCLUDE_DIR)
        else:
            print('-- Not using cuDNN')
        if USE_MIOPEN:
            print('-- Detected MIOpen at ' + MIOPEN_LIBRARY + ', ' + MIOPEN_INCLUDE_DIR)
        else:
            print('-- Not using MIOpen')
        if USE_CUDA:
            print('-- Detected CUDA at ' + CUDA_HOME)
        else:
            print('-- Not using CUDA')
        if USE_MKLDNN:
            print('-- Using MKLDNN')
        else:
            print('-- Not using MKLDNN')
        if USE_NCCL and USE_SYSTEM_NCCL:
            print('-- Using system provided NCCL library at ' +
                  NCCL_SYSTEM_LIB + ', ' + NCCL_INCLUDE_DIR)
        elif USE_NCCL:
            print('-- Building NCCL library')
        else:
            print('-- Not using NCCL')
        if USE_DISTRIBUTED:
            print('-- Building with THD distributed package ')
            if IS_LINUX:
                print('-- Building with c10d distributed package ')
                monkey_patch_C10D_inc_flags()
            else:
                print('-- Building without c10d distributed package')
        else:
            print('-- Building without distributed package')

        if USE_NINJA:
            ninja_builder = NinjaBuilder('global')

            generate_code(ninja_builder)

            # before we start the normal build make sure all generated code
            # gets built
            ninja_builder.run()
        else:
            generate_code(None)

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
            print("\nCopying extension {}".format(ext.name))

            src = os.path.join(tmp_install_path, rel_site_packages, filename)
            if not os.path.exists(src):
                print("{} does not exist".format(src))
                del self.extensions[i]
            else:
                dst = os.path.join(os.path.realpath(self.build_lib), filename)
                print("Copying {} from {} to {}".format(ext.name, src, dst))
                dst_dir = os.path.dirname(dst)
                if not os.path.exists(dst_dir):
                    os.makedirs(dst_dir)
                self.copy_file(src, dst)
                i += 1
        distutils.command.build_ext.build_ext.build_extensions(self)

    def get_outputs(self):
        outputs = distutils.command.build_ext.build_ext.get_outputs(self)
        outputs.append(os.path.join(self.build_lib, "caffe2"))
        print("setup.py::get_outputs returning {}".format(outputs))
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
        global RERUN_CMAKE
        RERUN_CMAKE = False
        distutils.command.build.build.run(self)


class install(setuptools.command.install.install):

    def run(self):
        print('setup.py::run()')
        if not self.skip_build:
            self.run_command('build_deps')

        setuptools.command.install.install.run(self)


class rebuild_libtorch(distutils.command.build.build):
    def run(self):
        if subprocess.call(['ninja', 'install'], cwd='build') != 0:
            print("Failed to run `ninja install` for the `rebuild_libtorch` command")
            sys.exit(1)


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

include_dirs = []
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
            print('The support for PyTorch with Python 2.7 on Windows is very experimental.')
            print('Please set the flag `FORCE_PY27_BUILD` to 1 to continue build.')
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
        '-Wno-zero-length-array',
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
        # gcc7 seems to report spurious warnings with this enabled
        "-Wno-stringop-overflow",
        # gcc7 also reports spurious warnings with this enabled
        "-Wno-maybe-uninitialized",
    ]
    if check_env_flag('WERROR'):
        extra_compile_args.append('-Werror')

include_dirs += [
    cwd,
    tmp_install_path + "/include",
    tmp_install_path + "/include/TH",
    tmp_install_path + "/include/THNN",
    tmp_install_path + "/include/ATen",
    third_party_path + "/pybind11/include",
    os.path.join(cwd, "torch", "csrc"),
    "build/third_party",
]

library_dirs.append(lib_path)

# we specify exact lib names to avoid conflict with lua-torch installs
CAFFE2_LIBS = [
    os.path.join(lib_path, 'libcaffe2.so'),
    os.path.join(lib_path, 'libc10.so')]
if USE_CUDA:
    CAFFE2_LIBS.extend(['-Wl,--no-as-needed', os.path.join(lib_path, 'libcaffe2_gpu.so'), '-Wl,--as-needed'])
if USE_ROCM:
    CAFFE2_LIBS.extend(['-Wl,--no-as-needed', os.path.join(lib_path, 'libcaffe2_hip.so'), '-Wl,--as-needed'])
THD_LIB = os.path.join(lib_path, 'libTHD.a')
NCCL_LIB = os.path.join(lib_path, 'libnccl.so.2')
C10D_LIB = os.path.join(lib_path, 'libc10d.a')
GLOO_LIB = os.path.join(lib_path, 'libgloo.a')
GLOO_CUDA_LIB = os.path.join(lib_path, 'libgloo_cuda.a')

# static library only
if IS_DARWIN:
    CAFFE2_LIBS = [os.path.join(lib_path, 'libcaffe2.dylib')]
    if USE_CUDA:
        CAFFE2_LIBS.append(os.path.join(lib_path, 'libcaffe2_gpu.dylib'))
    if USE_ROCM:
        CAFFE2_LIBS.append(os.path.join(lib_path, 'libcaffe2_hip.dylib'))
    NCCL_LIB = os.path.join(lib_path, 'libnccl.2.dylib')

if IS_WINDOWS:
    CAFFE2_LIBS = [
        os.path.join(lib_path, 'caffe2.lib'),
        os.path.join(lib_path, 'c10.lib')
    ]
    if USE_CUDA:
        CAFFE2_LIBS.append(os.path.join(lib_path, 'caffe2_gpu.lib'))
    if USE_ROCM:
        CAFFE2_LIBS.append(os.path.join(lib_path, 'caffe2_hip.lib'))

main_compile_args = ['-D_THP_CORE', '-DONNX_NAMESPACE=' + ONNX_NAMESPACE]
main_libraries = ['shm']
main_link_args = CAFFE2_LIBS
if IS_WINDOWS:
    main_link_args.append(os.path.join(lib_path, 'torch.lib'))
elif IS_DARWIN:
    main_link_args.append(os.path.join(lib_path, 'libtorch.dylib'))
else:
    main_link_args.append(os.path.join(lib_path, 'libtorch.so'))
main_sources = [
    "torch/csrc/DataLoader.cpp",
    "torch/csrc/Device.cpp",
    "torch/csrc/Dtype.cpp",
    "torch/csrc/DynamicTypes.cpp",
    "torch/csrc/Exceptions.cpp",
    "torch/csrc/TypeInfo.cpp",
    "torch/csrc/Generator.cpp",
    "torch/csrc/Layout.cpp",
    "torch/csrc/Module.cpp",
    "torch/csrc/PtrWrapper.cpp",
    "torch/csrc/Size.cpp",
    "torch/csrc/Storage.cpp",
    "torch/csrc/autograd/functions/init.cpp",
    "torch/csrc/autograd/generated/python_functions.cpp",
    "torch/csrc/autograd/generated/python_nn_functions.cpp",
    "torch/csrc/autograd/generated/python_torch_functions.cpp",
    "torch/csrc/autograd/generated/python_variable_methods.cpp",
    "torch/csrc/autograd/init.cpp",
    "torch/csrc/autograd/python_anomaly_mode.cpp",
    "torch/csrc/autograd/python_cpp_function.cpp",
    "torch/csrc/autograd/python_engine.cpp",
    "torch/csrc/autograd/python_function.cpp",
    "torch/csrc/autograd/python_hook.cpp",
    "torch/csrc/autograd/python_legacy_variable.cpp",
    "torch/csrc/autograd/python_variable.cpp",
    "torch/csrc/autograd/python_variable_indexing.cpp",
    "torch/csrc/byte_order.cpp",
    "torch/csrc/jit/batched/BatchTensor.cpp",
    "torch/csrc/jit/init.cpp",
    "torch/csrc/jit/passes/onnx.cpp",
    "torch/csrc/jit/passes/onnx/fixup_onnx_loop.cpp",
    "torch/csrc/jit/passes/onnx/prepare_division_for_onnx.cpp",
    "torch/csrc/jit/passes/onnx/peephole.cpp",
    "torch/csrc/jit/passes/to_batch.cpp",
    "torch/csrc/jit/python_arg_flatten.cpp",
    "torch/csrc/jit/python_interpreter.cpp",
    "torch/csrc/jit/python_ir.cpp",
    "torch/csrc/jit/python_tracer.cpp",
    "torch/csrc/jit/script/init.cpp",
    "torch/csrc/jit/script/lexer.cpp",
    "torch/csrc/jit/script/module.cpp",
    "torch/csrc/jit/script/python_tree_views.cpp",
    "torch/csrc/nn/THNN.cpp",
    "torch/csrc/onnx/init.cpp",
    "torch/csrc/serialization.cpp",
    "torch/csrc/tensor/python_tensor.cpp",
    "torch/csrc/utils.cpp",
    "torch/csrc/utils/cuda_lazy_init.cpp",
    "torch/csrc/utils/invalid_arguments.cpp",
    "torch/csrc/utils/object_ptr.cpp",
    "torch/csrc/utils/python_arg_parser.cpp",
    "torch/csrc/utils/tensor_apply.cpp",
    "torch/csrc/utils/tensor_conversion_dispatch.cpp",
    "torch/csrc/utils/tensor_dtypes.cpp",
    "torch/csrc/utils/tensor_flatten.cpp",
    "torch/csrc/utils/tensor_layouts.cpp",
    "torch/csrc/utils/tensor_list.cpp",
    "torch/csrc/utils/tensor_new.cpp",
    "torch/csrc/utils/tensor_numpy.cpp",
    "torch/csrc/utils/tensor_types.cpp",
    "torch/csrc/utils/tuple_parser.cpp",
]

try:
    import numpy as np
    include_dirs.append(np.get_include())
    extra_compile_args.append('-DUSE_NUMPY')
    USE_NUMPY = True
except ImportError:
    USE_NUMPY = False

if USE_DISTRIBUTED:
    extra_compile_args += ['-DUSE_DISTRIBUTED']
    main_sources += [
        "torch/csrc/distributed/Module.cpp",
    ]
    include_dirs += [tmp_install_path + "/include/THD"]
    main_link_args += [THD_LIB]
    if IS_LINUX:
        extra_compile_args.append('-DUSE_C10D')
        main_sources.append('torch/csrc/distributed/c10d/init.cpp')
        main_link_args.append(C10D_LIB)
        main_link_args.append(GLOO_LIB)
        if USE_CUDA:
            main_sources.append('torch/csrc/distributed/c10d/ddp.cpp')
            main_link_args.append(GLOO_CUDA_LIB)

if USE_CUDA:
    nvtoolext_lib_name = None
    if IS_WINDOWS:
        cuda_lib_path = CUDA_HOME + '/lib/x64/'
        nvtoolext_lib_path = NVTOOLEXT_HOME + '/lib/x64/'
        nvtoolext_include_path = os.path.join(NVTOOLEXT_HOME, 'include')

        library_dirs.append(nvtoolext_lib_path)
        include_dirs.append(nvtoolext_include_path)

        nvtoolext_lib_name = 'nvToolsExt64_1'

        # MSVC doesn't support runtime symbol resolving, `nvrtc` and `cuda` should be linked
        main_libraries += ['nvrtc', 'cuda']
    else:
        cuda_lib_dirs = ['lib64', 'lib']

        for lib_dir in cuda_lib_dirs:
            cuda_lib_path = os.path.join(CUDA_HOME, lib_dir)
            if os.path.exists(cuda_lib_path):
                break
        extra_link_args.append('-Wl,-rpath,' + cuda_lib_path)

        nvtoolext_lib_name = 'nvToolsExt'

    library_dirs.append(cuda_lib_path)
    cuda_include_path = os.path.join(CUDA_HOME, 'include')
    include_dirs.append(cuda_include_path)
    include_dirs.append(tmp_install_path + "/include/THCUNN")
    extra_compile_args += ['-DUSE_CUDA']
    extra_compile_args += ['-DCUDA_LIB_PATH=' + cuda_lib_path]
    main_libraries += ['cudart', nvtoolext_lib_name]
    main_sources += [
        "torch/csrc/cuda/Module.cpp",
        "torch/csrc/cuda/Storage.cpp",
        "torch/csrc/cuda/Stream.cpp",
        "torch/csrc/cuda/utils.cpp",
        "torch/csrc/cuda/comm.cpp",
        "torch/csrc/cuda/python_comm.cpp",
        "torch/csrc/cuda/serialization.cpp",
        "torch/csrc/nn/THCUNN.cpp",
    ]

if USE_ROCM:
    rocm_include_path = '/opt/rocm/include'
    hcc_include_path = '/opt/rocm/hcc/include'
    rocblas_include_path = '/opt/rocm/rocblas/include'
    hipsparse_include_path = '/opt/rocm/hipsparse/include'
    rocfft_include_path = '/opt/rocm/rocfft/include'
    hiprand_include_path = '/opt/rocm/hiprand/include'
    rocrand_include_path = '/opt/rocm/rocrand/include'
    thrust_include_path = '/opt/rocm/include/'
    hip_lib_path = '/opt/rocm/hip/lib'
    hcc_lib_path = '/opt/rocm/hcc/lib'
    include_dirs.append(rocm_include_path)
    include_dirs.append(hcc_include_path)
    include_dirs.append(rocblas_include_path)
    include_dirs.append(rocfft_include_path)
    include_dirs.append(hipsparse_include_path)
    include_dirs.append(hiprand_include_path)
    include_dirs.append(rocrand_include_path)
    include_dirs.append(thrust_include_path)
    include_dirs.append(tmp_install_path + "/include/THCUNN")
    extra_link_args.append('-L' + hip_lib_path)
    extra_link_args.append('-Wl,-rpath,' + hip_lib_path)
    extra_compile_args += ['-DUSE_ROCM']
    extra_compile_args += ['-D__HIP_PLATFORM_HCC__']

    main_sources += [
        "torch/csrc/cuda/Module.cpp",
        "torch/csrc/cuda/Storage.cpp",
        "torch/csrc/cuda/Stream.cpp",
        "torch/csrc/cuda/utils.cpp",
        "torch/csrc/cuda/comm.cpp",
        "torch/csrc/cuda/python_comm.cpp",
        "torch/csrc/cuda/serialization.cpp",
        "torch/csrc/nn/THCUNN.cpp",
    ]

if USE_NCCL:
    if USE_SYSTEM_NCCL:
        include_dirs.append(NCCL_INCLUDE_DIR)
    else:
        include_dirs.append("build/nccl/include")
    extra_compile_args += ['-DUSE_NCCL']
    main_sources += [
        "torch/csrc/cuda/nccl.cpp",
        "torch/csrc/cuda/python_nccl.cpp",
    ]
if USE_CUDNN:
    main_libraries += [CUDNN_LIBRARY]
    # NOTE: these are at the front, in case there's another cuDNN in CUDA path
    include_dirs.insert(0, CUDNN_INCLUDE_DIR)
    if not IS_WINDOWS:
        extra_link_args.insert(0, '-Wl,-rpath,' + CUDNN_LIB_DIR)
    extra_compile_args += ['-DUSE_CUDNN']

if USE_MIOPEN:
    main_libraries += [MIOPEN_LIBRARY]
    include_dirs.insert(0, MIOPEN_INCLUDE_DIR)
    extra_link_args.append('-L' + MIOPEN_LIB_DIR)
    if not IS_WINDOWS:
        extra_link_args.insert(0, '-Wl,-rpath,' + MIOPEN_LIB_DIR)
    extra_compile_args += ['-DWITH_MIOPEN']

if DEBUG:
    if IS_WINDOWS:
        extra_link_args.append('/DEBUG:FULL')
    else:
        extra_compile_args += ['-O0', '-g']
        extra_link_args += ['-O0', '-g']


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
              include_dirs=include_dirs,
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
                        include_dirs=include_dirs,
                        library_dirs=library_dirs + cuda_stub_path,
                        extra_link_args=thnvrtc_link_flags,
                        )
    extensions.append(THNVRTC)

# These extensions are built by cmake and copied manually in build_extensions()
# inside the build_ext implementaiton
extensions.append(
    setuptools.Extension(
        name=str('caffe2.python.caffe2_pybind11_state'),
        sources=[]),
)
if USE_CUDA:
    extensions.append(
        setuptools.Extension(
            name=str('caffe2.python.caffe2_pybind11_state_gpu'),
            sources=[]),
    )

cmdclass = {
    'create_version_file': create_version_file,
    'build': build,
    'build_py': build_py,
    'build_ext': build_ext,
    'build_deps': build_deps,
    'build_module': build_module,
    'rebuild_libtorch': rebuild_libtorch,
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
                'lib/include/c10/util/*.h',
                'lib/include/c10/detail/*.h',
                'lib/include/caffe2/core/*.h',
                'lib/include/caffe2/proto/*.h',
                'lib/include/torch/*.h',
                'lib/include/torch/csrc/*.h',
                'lib/include/torch/csrc/api/include/torch/*.h',
                'lib/include/torch/csrc/api/include/torch/data/*.h',
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
                rel_site_packages + '/caffe2/**/*.py'
            ]
        },
    )
