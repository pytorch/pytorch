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
#   NO_MKLDNN
#     disables the MKLDNN build
#
#   NO_NNPACK
#     disables NNPACK build
#
#   NO_DISTRIBUTED
#     disables THD (distributed) build
#
#   NO_SYSTEM_NCCL
#     disables use of system-wide nccl (we will use our submoduled
#     copy in third_party/nccl)
#
#   USE_GLOO_IBVERBS
#     toggle features related to distributed support
#
#   PYTORCH_BUILD_VERSION
#   PYTORCH_BUILD_NUMBER
#     specify the version of PyTorch, rather than the hard-coded version
#     in this file; used when we're building binaries for distribution
#
#   TORCH_CUDA_ARCH_LIST
#     specify which CUDA architectures to build for.
#     ie `TORCH_CUDA_ARCH_LIST="6.0;7.0"`
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
#
#   CUDNN_LIB_DIR
#   CUDNN_INCLUDE_DIR
#   CUDNN_LIBRARY
#     specify where cuDNN is installed
#
#   NCCL_ROOT_DIR
#   NCCL_LIB_DIR
#   NCCL_INCLUDE_DIR
#     specify where nccl is installed
#
#   MKLDNN_LIB_DIR
#   MKLDNN_LIBRARY
#   MKLDNN_INCLUDE_DIR
#     specify where MKLDNN is installed
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

# Before we run the setup_helpers, let's look for NO_* and WITH_*
# variables and hotpatch the environment with the USE_* equivalent
config_env_vars = ['CUDA', 'CUDNN', 'MKLDNN', 'NNPACK', 'DISTRIBUTED', 'DISTRIBUTED_MW',
                   'SYSTEM_NCCL', 'GLOO_IBVERBS']


def hotpatch_var(var):
    if check_env_flag('NO_' + var):
        os.environ['USE_' + var] = '0'
    elif check_negative_env_flag('NO_' + var):
        os.environ['USE_' + var] = '1'
    elif check_env_flag('WITH_' + var):
        os.environ['USE_' + var] = '1'
    elif check_negative_env_flag('WITH_' + var):
        os.environ['USE_' + var] = '0'

list(map(hotpatch_var, config_env_vars))

from tools.setup_helpers.cuda import USE_CUDA, CUDA_HOME, CUDA_VERSION
from tools.setup_helpers.rocm import USE_ROCM, ROCM_HOME, ROCM_VERSION
from tools.setup_helpers.cudnn import (USE_CUDNN, CUDNN_LIBRARY,
                                       CUDNN_LIB_DIR, CUDNN_INCLUDE_DIR)
from tools.setup_helpers.nccl import USE_NCCL, USE_SYSTEM_NCCL, NCCL_LIB_DIR, \
    NCCL_INCLUDE_DIR, NCCL_ROOT_DIR, NCCL_SYSTEM_LIB
from tools.setup_helpers.mkldnn import (USE_MKLDNN, MKLDNN_LIBRARY,
                                        MKLDNN_LIB_DIR, MKLDNN_INCLUDE_DIR)
from tools.setup_helpers.nnpack import USE_NNPACK
from tools.setup_helpers.nvtoolext import NVTOOLEXT_HOME
from tools.setup_helpers.generate_code import generate_code
from tools.setup_helpers.ninja_builder import NinjaBuilder, ninja_build_ext
from tools.setup_helpers.dist_check import USE_DISTRIBUTED, \
    USE_DISTRIBUTED_MW, USE_GLOO_IBVERBS, USE_C10D

################################################################################
# Parameters parsed from environment
################################################################################

DEBUG = check_env_flag('DEBUG')
IS_WINDOWS = (platform.system() == 'Windows')
IS_DARWIN = (platform.system() == 'Darwin')
IS_LINUX = (platform.system() == 'Linux')

FULL_CAFFE2 = check_env_flag('FULL_CAFFE2')
BUILD_PYTORCH = check_env_flag('BUILD_PYTORCH')

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
    ninja_global = NinjaBuilder('global')
except ImportError:
    USE_NINJA = False
    ninja_global = None

# Constant known variables used throughout this file
cwd = os.path.dirname(os.path.abspath(__file__))
lib_path = os.path.join(cwd, "torch", "lib")
third_party_path = os.path.join(cwd, "third_party")
tmp_install_path = lib_path + "/tmp_install"
rel_site_packages = distutils.sysconfig.get_python_lib(prefix='')
full_site_packages = distutils.sysconfig.get_python_lib()


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
# Version and create_version_file
################################################################################
version = '0.5.0a0'
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
dep_libs = [
    'nccl', 'caffe2',
    'libshm', 'libshm_windows', 'gloo', 'THD', 'nanopb', 'c10d',
]

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
        build_libs_cmd = ['bash', 'tools/build_pytorch_libs.sh']
    my_env = os.environ.copy()
    my_env["PYTORCH_PYTHON"] = sys.executable
    my_env["CMAKE_PREFIX_PATH"] = full_site_packages
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
    if USE_ROCM:
        build_libs_cmd += ['--use-rocm']
    if USE_NNPACK:
        build_libs_cmd += ['--use-nnpack']
    if USE_CUDNN:
        my_env["CUDNN_LIB_DIR"] = CUDNN_LIB_DIR
        my_env["CUDNN_LIBRARY"] = CUDNN_LIBRARY
        my_env["CUDNN_INCLUDE_DIR"] = CUDNN_INCLUDE_DIR
    if USE_MKLDNN:
        my_env["MKLDNN_LIB_DIR"] = MKLDNN_LIB_DIR
        my_env["MKLDNN_LIBRARY"] = MKLDNN_LIBRARY
        my_env["MKLDNN_INCLUDE_DIR"] = MKLDNN_INCLUDE_DIR
        build_libs_cmd += ['--use-mkldnn']
    if USE_GLOO_IBVERBS:
        build_libs_cmd += ['--use-gloo-ibverbs']
    if USE_DISTRIBUTED_MW:
        build_libs_cmd += ['--use-distributed-mw']
    if FULL_CAFFE2:
        build_libs_cmd += ['--full-caffe2']

    my_env["BUILD_TORCH"] = "ON"

    if subprocess.call(build_libs_cmd + libs, env=my_env) != 0:
        print("Failed to run '{}'".format(' '.join(build_libs_cmd + libs)))
        sys.exit(1)


# Build all dependent libraries
class build_deps(PytorchCommand):
    def run(self):
        # Check if you remembered to check out submodules
        def check_file(f):
            if not os.path.exists(f):
                print("Could not find {}".format(f))
                print("Did you run 'git submodule update --init'?")
                sys.exit(1)
        check_file(os.path.join(third_party_path, "gloo", "CMakeLists.txt"))
        check_file(os.path.join(third_party_path, "nanopb", "CMakeLists.txt"))
        check_file(os.path.join(third_party_path, "pybind11", "CMakeLists.txt"))
        check_file(os.path.join(third_party_path, 'cpuinfo', 'CMakeLists.txt'))
        check_file(os.path.join(third_party_path, 'catch', 'CMakeLists.txt'))
        check_file(os.path.join(third_party_path, 'onnx', 'CMakeLists.txt'))

        check_pydep('yaml', 'pyyaml')
        check_pydep('typing', 'typing')

        libs = []
        if USE_NCCL and not USE_SYSTEM_NCCL:
            libs += ['nccl']
        libs += ['caffe2', 'nanopb']
        if IS_WINDOWS:
            libs += ['libshm_windows']
        else:
            libs += ['libshm']
        if USE_DISTRIBUTED:
            if sys.platform.startswith('linux'):
                libs += ['gloo']
            libs += ['THD']
        if USE_C10D:
            libs += ['c10d']
        build_libs(libs)

        # Use copies instead of symbolic files.
        # Windows has very poor support for them.
        sym_files = ['tools/shared/cwrap_common.py', 'tools/shared/_utils_internal.py']
        orig_files = ['aten/src/ATen/common_with_cwrap.py', 'torch/_utils_internal.py']
        for sym_file, orig_file in zip(sym_files, orig_files):
            if os.path.exists(sym_file):
                os.remove(sym_file)
            shutil.copyfile(orig_file, sym_file)

        # Copy headers necessary to compile C++ extensions.
        #
        # This is not perfect solution as build does not depend on any of
        # the auto-generated code and auto-generated files will not be
        # included in this copy. If we want to use auto-generated files,
        # we need to find a better way to do this.
        # More information can be found in conversation thread of PR #5772

        self.copy_tree('torch/csrc', 'torch/lib/include/torch/csrc/')
        self.copy_tree('third_party/pybind11/include/pybind11/',
                       'torch/lib/include/pybind11')
        self.copy_file('torch/csrc/torch.h', 'torch/lib/include/torch/torch.h')


build_dep_cmds = {}

for lib in dep_libs:
    # wrap in function to capture lib
    class build_dep(build_deps):
        description = 'Build {} external library'.format(lib)

        def run(self):
            build_libs([self.lib])
    build_dep.lib = lib
    build_dep_cmds['build_' + lib.lower()] = build_dep


class build_module(PytorchCommand):
    def run(self):
        self.run_command('build_py')
        self.run_command('build_ext')


class build_py(setuptools.command.build_py.build_py):

    def run(self):
        self.run_command('create_version_file')
        setuptools.command.build_py.build_py.run(self)


class develop(setuptools.command.develop.develop):

    def run(self):
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
        with open('compile_commands.json', 'w') as f:
            json.dump(all_commands, f, indent=2)
        if not USE_NINJA:
            print("WARNING: 'develop' is not building C++ code incrementally")
            print("because ninja is not installed. Run this to enable it:")
            print(" > pip install ninja")


def monkey_patch_THD_link_flags():
    '''
    THD's dynamic link deps are not determined until after build_deps is run
    So, we need to monkey-patch them in later
    '''
    # read tmp_install_path/THD_deps.txt for THD's dynamic linkage deps
    with open(tmp_install_path + '/THD_deps.txt', 'r') as f:
        thd_deps_ = f.read()
    thd_deps = []
    # remove empty lines
    for l in thd_deps_.split(';'):
        if l != '':
            thd_deps.append(l)

    C.extra_link_args += thd_deps


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
        if USE_CUDA:
            print('-- Detected CUDA at ' + CUDA_HOME)
        else:
            print('-- Not using CUDA')
        if USE_MKLDNN:
            print('-- Detected MKLDNN at ' + MKLDNN_LIBRARY + ', ' + MKLDNN_INCLUDE_DIR)
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
            print('-- Building with distributed package ')
            monkey_patch_THD_link_flags()
        else:
            print('-- Building without distributed package')

        generate_code(ninja_global)

        if USE_NINJA:
            # before we start the normal build make sure all generated code
            # gets built
            ninja_global.run()

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
        if FULL_CAFFE2:
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

                src = os.path.join(tmp_install_path, rel_site_packages, filename)
                if not os.path.exists(src):
                    print("{} does not exist".format(src))
                    del self.extensions[i]
                else:
                    dst = os.path.join(os.path.realpath(self.build_lib), filename)
                    dst_dir = os.path.dirname(dst)
                    if not os.path.exists(dst_dir):
                        os.makedirs(dst_dir)
                    self.copy_file(src, dst)
                    i += 1
        distutils.command.build_ext.build_ext.build_extensions(self)

    def get_outputs(self):
        outputs = distutils.command.build_ext.build_ext.get_outputs(self)
        if FULL_CAFFE2:
            outputs += [os.path.join(self.build_lib, d) for d in ['caffe', 'caffe2']]
        return outputs


class build(distutils.command.build.build):
    sub_commands = [
        ('build_deps', lambda self: True),
    ] + distutils.command.build.build.sub_commands


class install(setuptools.command.install.install):

    def run(self):
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
CAFFE2_LIBS = [os.path.join(lib_path, 'libcaffe2.so')]
if USE_CUDA:
    CAFFE2_LIBS.extend(['-Wl,--no-as-needed', os.path.join(lib_path, 'libcaffe2_gpu.so'), '-Wl,--as-needed'])
if USE_ROCM:
    CAFFE2_LIBS.extend(['-Wl,--no-as-needed', os.path.join(lib_path, 'libcaffe2_hip.so'), '-Wl,--as-needed'])
THD_LIB = os.path.join(lib_path, 'libTHD.a')
NCCL_LIB = os.path.join(lib_path, 'libnccl.so.1')
C10D_LIB = os.path.join(lib_path, 'libc10d.a')

# static library only
NANOPB_STATIC_LIB = os.path.join(lib_path, 'libprotobuf-nanopb.a')
if DEBUG:
    PROTOBUF_STATIC_LIB = os.path.join(lib_path, 'libprotobufd.a')
else:
    PROTOBUF_STATIC_LIB = os.path.join(lib_path, 'libprotobuf.a')

if IS_DARWIN:
    CAFFE2_LIBS = [os.path.join(lib_path, 'libcaffe2.dylib')]
    if USE_CUDA:
        CAFFE2_LIBS.append(os.path.join(lib_path, 'libcaffe2_gpu.dylib'))
    if USE_ROCM:
        CAFFE2_LIBS.append(os.path.join(lib_path, 'libcaffe2_hip.dylib'))
    NCCL_LIB = os.path.join(lib_path, 'libnccl.1.dylib')

if IS_WINDOWS:
    CAFFE2_LIBS = [os.path.join(lib_path, 'caffe2.lib')]
    if USE_CUDA:
        CAFFE2_LIBS.append(os.path.join(lib_path, 'caffe2_gpu.lib'))
    if USE_ROCM:
        CAFFE2_LIBS.append(os.path.join(lib_path, 'caffe2_hip.lib'))
    # Windows needs direct access to ONNX libraries as well
    # as through Caffe2 library
    CAFFE2_LIBS += [
        os.path.join(lib_path, 'onnx.lib'),
        os.path.join(lib_path, 'onnx_proto.lib'),
    ]
    if DEBUG:
        NANOPB_STATIC_LIB = os.path.join(lib_path, 'protobuf-nanopbd.lib')
        PROTOBUF_STATIC_LIB = os.path.join(lib_path, 'libprotobufd.lib')
    else:
        NANOPB_STATIC_LIB = os.path.join(lib_path, 'protobuf-nanopb.lib')
        PROTOBUF_STATIC_LIB = os.path.join(lib_path, 'libprotobuf.lib')

main_compile_args = ['-D_THP_CORE', '-DONNX_NAMESPACE=' + ONNX_NAMESPACE]
main_libraries = ['shm']
main_link_args = CAFFE2_LIBS + [NANOPB_STATIC_LIB, PROTOBUF_STATIC_LIB]
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
    "torch/csrc/finalizer.cpp",
    "torch/csrc/jit/batched/BatchTensor.cpp",
    "torch/csrc/jit/init.cpp",
    "torch/csrc/jit/ivalue.cpp",
    "torch/csrc/jit/passes/onnx.cpp",
    "torch/csrc/jit/passes/onnx/fixup_onnx_loop.cpp",
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
    if USE_DISTRIBUTED_MW:
        main_sources += [
            "torch/csrc/distributed/Tensor.cpp",
            "torch/csrc/distributed/Storage.cpp",
        ]
        extra_compile_args += ['-DUSE_DISTRIBUTED_MW']
    include_dirs += [tmp_install_path + "/include/THD"]
    main_link_args += [THD_LIB]

if USE_C10D:
    extra_compile_args += ['-DUSE_C10D']
    main_sources += ['torch/csrc/distributed/c10d/init.cpp']
    main_link_args += [C10D_LIB]

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
    hipblas_include_path = '/opt/rocm/hipblas/include'
    hipsparse_include_path = '/opt/rocm/hcsparse/include'
    hiprand_include_path = '/opt/rocm/hiprand/include'
    rocrand_include_path = '/opt/rocm/rocrand/include'
    hip_lib_path = '/opt/rocm/hip/lib'
    hcc_lib_path = '/opt/rocm/hcc/lib'
    include_dirs.append(rocm_include_path)
    include_dirs.append(hcc_include_path)
    include_dirs.append(hipblas_include_path)
    include_dirs.append(hipsparse_include_path)
    include_dirs.append(hiprand_include_path)
    include_dirs.append(rocrand_include_path)
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
        main_link_args += [NCCL_SYSTEM_LIB]
        include_dirs.append(NCCL_INCLUDE_DIR)
    else:
        main_link_args += [NCCL_LIB]
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
if FULL_CAFFE2:
    packages = find_packages(exclude=('tools', 'tools.*'))
else:
    packages = find_packages(exclude=('tools', 'tools.*', 'caffe2', 'caffe2.*', 'caffe', 'caffe.*'))
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

if FULL_CAFFE2:
    # If building Caffe2 python as well, these extensions are built by cmake
    # copied manually in build_extensions() inside the build_ext implementaiton
    extensions.append(
        setuptools.Extension(
            name=str('caffe2.python.caffe2_pybind11_state'),
            sources=[]),
    )
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
    'develop': develop,
    'install': install,
    'clean': clean,
}
cmdclass.update(build_dep_cmds)

if __name__ == '__main__':
    setup(
        name="torch",
        version=version,
        description=("Tensors and Dynamic neural networks in "
                     "Python with strong GPU acceleration"),
        ext_modules=extensions,
        cmdclass=cmdclass,
        packages=packages,
        package_data={
            'torch': [
                'lib/*.so*',
                'lib/*.dylib*',
                'lib/*.dll',
                'lib/*.lib',
                'lib/torch_shm_manager',
                'lib/*.h',
                'lib/include/ATen/*.h',
                'lib/include/ATen/core/*.h',
                'lib/include/ATen/detail/*.h',
                'lib/include/ATen/cuda/*.h',
                'lib/include/ATen/cuda/*.cuh',
                'lib/include/ATen/cuda/detail/*.h',
                'lib/include/ATen/cudnn/*.h',
                'lib/include/ATen/cuda/detail/*.cuh',
                'lib/include/pybind11/*.h',
                'lib/include/pybind11/detail/*.h',
                'lib/include/TH/*.h*',
                'lib/include/TH/generic/*.h*',
                'lib/include/THC/*.h*',
                'lib/include/THC/*.cuh',
                'lib/include/THC/generic/*.h',
                'lib/include/THCUNN/*.cuh',
                'lib/include/THNN/*.h',
                'lib/include/torch/csrc/*.h',
                'lib/include/torch/csrc/autograd/*.h',
                'lib/include/torch/csrc/jit/*.h',
                'lib/include/torch/csrc/utils/*.h',
                'lib/include/torch/csrc/cuda/*.h',
                'lib/include/torch/torch.h',
            ]
        })
