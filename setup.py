from setuptools import setup, Extension, distutils, Command, find_packages
import setuptools.command.build_ext
import setuptools.command.install
import setuptools.command.develop
import setuptools.command.build_py
import distutils.unixccompiler
import distutils.command.build
import distutils.command.clean
import platform
import subprocess
import shutil
import sys
import os

from tools.setup_helpers.env import check_env_flag
from tools.setup_helpers.cuda import WITH_CUDA, CUDA_HOME
from tools.setup_helpers.cudnn import WITH_CUDNN, CUDNN_LIB_DIR, CUDNN_INCLUDE_DIR
from tools.setup_helpers.split_types import split_types
DEBUG = check_env_flag('DEBUG')
WITH_DISTRIBUTED = not check_env_flag('NO_DISTRIBUTED')
WITH_DISTRIBUTED_MW = WITH_DISTRIBUTED and check_env_flag('WITH_DISTRIBUTED_MW')
WITH_NCCL = WITH_CUDA and platform.system() != 'Darwin'
SYSTEM_NCCL = False


################################################################################
# Workaround setuptools -Wstrict-prototypes warnings
# I lifted this code from https://stackoverflow.com/a/29634231/23845
################################################################################
import distutils.sysconfig
cfg_vars = distutils.sysconfig.get_config_vars()
for key, value in cfg_vars.items():
    if type(value) == str:
            cfg_vars[key] = value.replace("-Wstrict-prototypes", "")

################################################################################
# Monkey-patch setuptools to compile in parallel
################################################################################
original_link = distutils.unixccompiler.UnixCCompiler.link


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
    num_jobs = multiprocessing.cpu_count()
    max_jobs = os.getenv("MAX_JOBS")
    if max_jobs is not None:
        num_jobs = min(num_jobs, int(max_jobs))
    multiprocessing.pool.ThreadPool(num_jobs).map(_single_compile, objects)

    return objects


def patched_link(self, *args, **kwargs):
    _cxx = self.compiler_cxx
    self.compiler_cxx = None
    result = original_link(self, *args, **kwargs)
    self.compiler_cxx = _cxx
    return result


distutils.ccompiler.CCompiler.compile = parallelCCompile
distutils.unixccompiler.UnixCCompiler.link = patched_link

################################################################################
# Custom build commands
################################################################################


class build_deps(Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        from tools.nnwrap import generate_wrappers as generate_nn_wrappers
        build_all_cmd = ['bash', 'torch/lib/build_all.sh']
        if WITH_CUDA:
            build_all_cmd += ['--with-cuda']
        if WITH_NCCL and not SYSTEM_NCCL:
            build_all_cmd += ['--with-nccl']
        if WITH_DISTRIBUTED:
            build_all_cmd += ['--with-distributed']
        if subprocess.call(build_all_cmd) != 0:
            sys.exit(1)
        generate_nn_wrappers()


class build_module(Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        self.run_command('build_py')
        self.run_command('build_ext')


class build_py(setuptools.command.build_py.build_py):

    def run(self):
        self.create_version_file()
        setuptools.command.build_py.build_py.run(self)

    @staticmethod
    def create_version_file():
        global version, cwd
        print('-- Building version ' + version)
        version_path = os.path.join(cwd, 'torch', 'version.py')
        with open(version_path, 'w') as f:
            f.write("__version__ = '{}'\n".format(version))


class develop(setuptools.command.develop.develop):

    def run(self):
        build_py.create_version_file()
        setuptools.command.develop.develop.run(self)


class build_ext(setuptools.command.build_ext.build_ext):

    def run(self):
        # Print build options
        if WITH_NUMPY:
            print('-- Building with NumPy bindings')
        else:
            print('-- NumPy not found')
        if WITH_CUDNN:
            print('-- Detected cuDNN at ' + CUDNN_LIB_DIR + ', ' + CUDNN_INCLUDE_DIR)
        else:
            print('-- Not using cuDNN')
        if WITH_CUDA:
            print('-- Detected CUDA at ' + CUDA_HOME)
        else:
            print('-- Not using CUDA')
        if WITH_NCCL and SYSTEM_NCCL:
            print('-- Using system provided NCCL library')
        elif WITH_NCCL:
            print('-- Building NCCL library')
        else:
            print('-- Not using NCCL')
        if WITH_DISTRIBUTED:
            print('-- Building with distributed package ')
        else:
            print('-- Building without distributed package')

        # cwrap depends on pyyaml, so we can't import it earlier
        from tools.cwrap import cwrap
        from tools.cwrap.plugins.THPPlugin import THPPlugin
        from tools.cwrap.plugins.ArgcountSortPlugin import ArgcountSortPlugin
        from tools.cwrap.plugins.AutoGPU import AutoGPU
        from tools.cwrap.plugins.BoolOption import BoolOption
        from tools.cwrap.plugins.KwargsPlugin import KwargsPlugin
        from tools.cwrap.plugins.NullableArguments import NullableArguments
        from tools.cwrap.plugins.CuDNNPlugin import CuDNNPlugin
        from tools.cwrap.plugins.WrapDim import WrapDim
        from tools.cwrap.plugins.AssertNDim import AssertNDim
        from tools.cwrap.plugins.Broadcast import Broadcast
        from tools.cwrap.plugins.ProcessorSpecificPlugin import ProcessorSpecificPlugin
        thp_plugin = THPPlugin()
        cwrap('torch/csrc/generic/TensorMethods.cwrap', plugins=[
            ProcessorSpecificPlugin(), BoolOption(), thp_plugin,
            AutoGPU(condition='IS_CUDA'), ArgcountSortPlugin(), KwargsPlugin(),
            AssertNDim(), WrapDim(), Broadcast()
        ])
        cwrap('torch/csrc/cudnn/cuDNN.cwrap', plugins=[
            CuDNNPlugin(), NullableArguments()
        ])
        # It's an old-style class in Python 2.7...
        setuptools.command.build_ext.build_ext.run(self)


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
        with open('.gitignore', 'r') as f:
            ignores = f.read()
            for wildcard in filter(bool, ignores.split('\n')):
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
extra_link_args = []
extra_compile_args = ['-std=c++11', '-Wno-write-strings',
                      # Python 2.6 requires -fno-strict-aliasing, see
                      # http://legacy.python.org/dev/peps/pep-3123/
                      '-fno-strict-aliasing']
if os.getenv('PYTORCH_BINARY_BUILD') and platform.system() == 'Linux':
    print('PYTORCH_BINARY_BUILD found. Static linking libstdc++ on Linux')
    # get path of libstdc++ and link manually.
    # for reasons unknown, -static-libstdc++ doesn't fully link some symbols
    CXXNAME = os.getenv('CXX', 'g++')
    path = subprocess.check_output([CXXNAME, '-print-file-name=libstdc++.a'])
    path = path[:-1]
    if type(path) != str:  # python 3
        path = path.decode(sys.stdout.encoding)
    extra_link_args += [path]

cwd = os.path.dirname(os.path.abspath(__file__))
lib_path = os.path.join(cwd, "torch", "lib")

tmp_install_path = lib_path + "/tmp_install"
include_dirs += [
    cwd,
    os.path.join(cwd, "torch", "csrc"),
    tmp_install_path + "/include",
    tmp_install_path + "/include/TH",
    tmp_install_path + "/include/THPP",
    tmp_install_path + "/include/THNN",
    tmp_install_path + "/include/ATen",
]

library_dirs.append(lib_path)

# we specify exact lib names to avoid conflict with lua-torch installs
TH_LIB = os.path.join(lib_path, 'libTH.so.1')
THS_LIB = os.path.join(lib_path, 'libTHS.so.1')
THC_LIB = os.path.join(lib_path, 'libTHC.so.1')
THCS_LIB = os.path.join(lib_path, 'libTHCS.so.1')
THNN_LIB = os.path.join(lib_path, 'libTHNN.so.1')
THCUNN_LIB = os.path.join(lib_path, 'libTHCUNN.so.1')
THPP_LIB = os.path.join(lib_path, 'libTHPP.so.1')
ATEN_LIB = os.path.join(lib_path, 'libATen.so.1')
THD_LIB = os.path.join(lib_path, 'libTHD.so.1')
NCCL_LIB = os.path.join(lib_path, 'libnccl.so.1')
if platform.system() == 'Darwin':
    TH_LIB = os.path.join(lib_path, 'libTH.1.dylib')
    THS_LIB = os.path.join(lib_path, 'libTHS.1.dylib')
    THC_LIB = os.path.join(lib_path, 'libTHC.1.dylib')
    THCS_LIB = os.path.join(lib_path, 'libTHCS.1.dylib')
    THNN_LIB = os.path.join(lib_path, 'libTHNN.1.dylib')
    THCUNN_LIB = os.path.join(lib_path, 'libTHCUNN.1.dylib')
    THPP_LIB = os.path.join(lib_path, 'libTHPP.1.dylib')
    ATEN_LIB = os.path.join(lib_path, 'libATen.1.dylib')
    THD_LIB = os.path.join(lib_path, 'libTHD.1.dylib')
    NCCL_LIB = os.path.join(lib_path, 'libnccl.1.dylib')

if WITH_NCCL and (subprocess.call('ldconfig -p | grep libnccl >/dev/null', shell=True) == 0 or
                  subprocess.call('/sbin/ldconfig -p | grep libnccl >/dev/null', shell=True) == 0):
        SYSTEM_NCCL = True

main_compile_args = ['-D_THP_CORE']
main_libraries = ['shm']
main_link_args = [TH_LIB, THS_LIB, THPP_LIB, THNN_LIB, ATEN_LIB]
main_sources = [
    "torch/csrc/PtrWrapper.cpp",
    "torch/csrc/Module.cpp",
    "torch/csrc/Generator.cpp",
    "torch/csrc/Size.cpp",
    "torch/csrc/Exceptions.cpp",
    "torch/csrc/Storage.cpp",
    "torch/csrc/DynamicTypes.cpp",
    "torch/csrc/byte_order.cpp",
    "torch/csrc/utils.cpp",
    "torch/csrc/expand_utils.cpp",
    "torch/csrc/utils/object_ptr.cpp",
    "torch/csrc/utils/tuple_parser.cpp",
    "torch/csrc/allocators.cpp",
    "torch/csrc/serialization.cpp",
    "torch/csrc/autograd/init.cpp",
    "torch/csrc/autograd/engine.cpp",
    "torch/csrc/autograd/function.cpp",
    "torch/csrc/autograd/variable.cpp",
    "torch/csrc/autograd/input_buffer.cpp",
    "torch/csrc/autograd/python_function.cpp",
    "torch/csrc/autograd/python_cpp_function.cpp",
    "torch/csrc/autograd/python_variable.cpp",
    "torch/csrc/autograd/python_engine.cpp",
    "torch/csrc/autograd/python_hook.cpp",
    "torch/csrc/autograd/functions/batch_normalization.cpp",
    "torch/csrc/autograd/functions/convolution.cpp",
    "torch/csrc/autograd/functions/basic_ops.cpp",
    "torch/csrc/autograd/functions/tensor.cpp",
    "torch/csrc/autograd/functions/accumulate_grad.cpp",
    "torch/csrc/autograd/functions/utils.cpp",
    "torch/csrc/autograd/functions/init.cpp",
]
main_sources += split_types("torch/csrc/Tensor.cpp")

try:
    import numpy as np
    include_dirs += [np.get_include()]
    extra_compile_args += ['-DWITH_NUMPY']
    WITH_NUMPY = True
except ImportError:
    WITH_NUMPY = False

if WITH_DISTRIBUTED:
    extra_compile_args += ['-DWITH_DISTRIBUTED']
    main_sources += [
        "torch/csrc/distributed/Module.cpp",
        "torch/csrc/distributed/utils.cpp",
    ]
    if WITH_DISTRIBUTED_MW:
        main_sources += [
            "torch/csrc/distributed/Tensor.cpp",
            "torch/csrc/distributed/Storage.cpp",
        ]
        extra_compile_args += ['-DWITH_DISTRIBUTED_MW']
    include_dirs += [tmp_install_path + "/include/THD"]
    main_link_args += [THD_LIB]

if WITH_CUDA:
    cuda_lib_dirs = ['lib64', 'lib']
    cuda_include_path = os.path.join(CUDA_HOME, 'include')
    for lib_dir in cuda_lib_dirs:
        cuda_lib_path = os.path.join(CUDA_HOME, lib_dir)
        if os.path.exists(cuda_lib_path):
            break
    include_dirs.append(cuda_include_path)
    include_dirs.append(tmp_install_path + "/include/THCUNN")
    library_dirs.append(cuda_lib_path)
    extra_link_args.append('-Wl,-rpath,' + cuda_lib_path)
    extra_compile_args += ['-DWITH_CUDA']
    extra_compile_args += ['-DCUDA_LIB_PATH=' + cuda_lib_path]
    main_libraries += ['cudart', 'nvToolsExt']
    main_link_args += [THC_LIB, THCS_LIB, THCUNN_LIB]
    main_sources += [
        "torch/csrc/cuda/Module.cpp",
        "torch/csrc/cuda/Storage.cpp",
        "torch/csrc/cuda/Stream.cpp",
        "torch/csrc/cuda/AutoGPU.cpp",
        "torch/csrc/cuda/utils.cpp",
        "torch/csrc/cuda/expand_utils.cpp",
        "torch/csrc/cuda/serialization.cpp",
    ]
    main_sources += split_types("torch/csrc/cuda/Tensor.cpp")

if WITH_NCCL:
    if SYSTEM_NCCL:
        main_libraries += ['nccl']
    else:
        main_link_args += [NCCL_LIB]
    extra_compile_args += ['-DWITH_NCCL']

if WITH_CUDNN:
    main_libraries += ['cudnn']
    include_dirs.append(CUDNN_INCLUDE_DIR)
    library_dirs.append(CUDNN_LIB_DIR)
    main_sources += [
        "torch/csrc/cudnn/BatchNorm.cpp",
        "torch/csrc/cudnn/Conv.cpp",
        "torch/csrc/cudnn/cuDNN.cpp",
        "torch/csrc/cudnn/GridSampler.cpp",
        "torch/csrc/cudnn/AffineGridGenerator.cpp",
        "torch/csrc/cudnn/Types.cpp",
        "torch/csrc/cudnn/Handles.cpp",
    ]
    extra_compile_args += ['-DWITH_CUDNN']

if DEBUG:
    extra_compile_args += ['-O0', '-g']
    extra_link_args += ['-O0', '-g']


def make_relative_rpath(path):
    if platform.system() == 'Darwin':
        return '-Wl,-rpath,@loader_path/' + path
    else:
        return '-Wl,-rpath,$ORIGIN/' + path

################################################################################
# Declare extensions and package
################################################################################

extensions = []
packages = find_packages(exclude=('tools', 'tools.*',))

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

DL = Extension("torch._dl",
               sources=["torch/csrc/dl.c"],
               language='c',
               )
extensions.append(DL)

THNN = Extension("torch._thnn._THNN",
                 sources=['torch/csrc/nn/THNN.cpp'],
                 language='c++',
                 extra_compile_args=extra_compile_args,
                 include_dirs=include_dirs,
                 extra_link_args=extra_link_args + [
                     TH_LIB,
                     THNN_LIB,
                     make_relative_rpath('../lib'),
                 ]
                 )
extensions.append(THNN)

if WITH_CUDA:
    THCUNN = Extension("torch._thnn._THCUNN",
                       sources=['torch/csrc/nn/THCUNN.cpp'],
                       language='c++',
                       extra_compile_args=extra_compile_args,
                       include_dirs=include_dirs,
                       extra_link_args=extra_link_args + [
                           TH_LIB,
                           THC_LIB,
                           THCUNN_LIB,
                           make_relative_rpath('../lib'),
                       ]
                       )
    extensions.append(THCUNN)

version = '0.2.0'
if os.getenv('PYTORCH_BUILD_VERSION'):
    assert os.getenv('PYTORCH_BUILD_NUMBER') is not None
    version = os.getenv('PYTORCH_BUILD_VERSION') \
        + '_' + os.getenv('PYTORCH_BUILD_NUMBER')
else:
    try:
        sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=cwd).decode('ascii').strip()
        version += '+' + sha[:7]
    except subprocess.CalledProcessError:
        pass


setup(name="torch", version=version,
      description="Tensors and Dynamic neural networks in Python with strong GPU acceleration",
      ext_modules=extensions,
      cmdclass={
          'build': build,
          'build_py': build_py,
          'build_ext': build_ext,
          'build_deps': build_deps,
          'build_module': build_module,
          'develop': develop,
          'install': install,
          'clean': clean,
      },
      packages=packages,
      package_data={'torch': [
          'lib/*.so*', 'lib/*.dylib*',
          'lib/torch_shm_manager',
          'lib/*.h',
          'lib/include/TH/*.h', 'lib/include/TH/generic/*.h',
          'lib/include/THC/*.h', 'lib/include/THC/generic/*.h']},
      install_requires=['pyyaml', 'numpy'],
      )
