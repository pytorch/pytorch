from setuptools import setup, Extension, distutils, Command, find_packages
import setuptools.command.build_ext
import setuptools.command.install
import distutils.command.build
import distutils.command.clean
import platform
import subprocess
import shutil
import sys
import os

# TODO: make this more robust
WITH_CUDA = os.path.exists('/Developer/NVIDIA/CUDA-7.5/include') or os.path.exists('/usr/local/cuda/include')
DEBUG = False

################################################################################
# Monkey-patch setuptools to compile in parallel
################################################################################

def parallelCCompile(self, sources, output_dir=None, macros=None, include_dirs=None, debug=0, extra_preargs=None, extra_postargs=None, depends=None):
    # those lines are copied from distutils.ccompiler.CCompiler directly
    macros, objects, extra_postargs, pp_opts, build = self._setup_compile(output_dir, macros, include_dirs, sources, depends, extra_postargs)
    cc_args = self._get_cc_args(pp_opts, debug, extra_preargs)

    # compile using a thread pool
    import multiprocessing.pool
    def _single_compile(obj):
        src, ext = build[obj]
        self._compile(obj, src, ext, cc_args, extra_postargs, pp_opts)
    num_jobs = multiprocessing.cpu_count()
    multiprocessing.pool.ThreadPool(num_jobs).map(_single_compile, objects)

    return objects

distutils.ccompiler.CCompiler.compile = parallelCCompile

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


class build_ext(setuptools.command.build_ext.build_ext):
    def run(self):
        # cwrap depends on pyyaml, so we can't import it earlier
        from tools.cwrap import cwrap
        from tools.cwrap.plugins.THPPlugin import THPPlugin
        from tools.cwrap.plugins.THPLongArgsPlugin import THPLongArgsPlugin
        from tools.cwrap.plugins.ArgcountSortPlugin import ArgcountSortPlugin
        from tools.cwrap.plugins.AutoGPU import AutoGPU
        cwrap('torch/csrc/generic/TensorMethods.cwrap', plugins=[
            THPLongArgsPlugin(), THPPlugin(), ArgcountSortPlugin(), AutoGPU()
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
        with open('.gitignore', 'r') as f:
            ignores = f.read()
            for glob in filter(bool, ignores.split('\n')):
                shutil.rmtree(glob, ignore_errors=True)
        # It's an old-style class in Python 2.7...
        distutils.command.clean.clean.run(self)



################################################################################
# Configure compile flags
################################################################################

include_dirs = []
extra_link_args = []
extra_compile_args = ['-std=c++11', '-Wno-write-strings']
if os.getenv('PYTORCH_BINARY_BUILD') and platform.system() == 'Linux':
    print('PYTORCH_BINARY_BUILD found. Static linking libstdc++ on Linux')
    extra_compile_args += ['-static-libstdc++']
    extra_link_args += ['-static-libstdc++']

cwd = os.path.dirname(os.path.abspath(__file__))
lib_path = os.path.join(cwd, "torch", "lib")

tmp_install_path = lib_path + "/tmp_install"
include_dirs += [
    cwd,
    os.path.join(cwd, "torch", "csrc"),
    tmp_install_path + "/include",
    tmp_install_path + "/include/TH",
]

extra_link_args.append('-L' + lib_path)

main_libraries = ['TH', 'shm']
main_sources = [
    "torch/csrc/Module.cpp",
    "torch/csrc/Generator.cpp",
    "torch/csrc/Tensor.cpp",
    "torch/csrc/Storage.cpp",
    "torch/csrc/byte_order.cpp",
    "torch/csrc/utils.cpp",
    "torch/csrc/allocators.cpp",
    "torch/csrc/serialization.cpp",
]

try:
    import numpy as np
    include_dirs += [np.get_include()]
    extra_compile_args += ['-DWITH_NUMPY']
except ImportError:
    pass

if WITH_CUDA:
    if platform.system() == 'Darwin':
        cuda_path = '/Developer/NVIDIA/CUDA-7.5'
        cuda_include_path = cuda_path + '/include'
        cuda_lib_path = cuda_path + '/lib'
    else:
        cuda_path = '/usr/local/cuda'
        cuda_include_path = cuda_path + '/include'
        cuda_lib_path = cuda_path + '/lib64'
    include_dirs.append(cuda_include_path)
    extra_link_args.append('-L' + cuda_lib_path)
    extra_link_args.append('-Wl,-rpath,' + cuda_lib_path)
    extra_compile_args += ['-DWITH_CUDA']
    main_libraries += ['THC']
    main_sources += [
        "torch/csrc/cuda/Module.cpp",
        "torch/csrc/cuda/Storage.cpp",
        "torch/csrc/cuda/Tensor.cpp",
        "torch/csrc/cuda/utils.cpp",
        "torch/csrc/cuda/serialization.cpp",
    ]

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
packages = find_packages(exclude=('tools.*', 'torch.cuda', 'torch.legacy.cunn'))

C = Extension("torch._C",
    libraries=main_libraries,
    sources=main_sources,
    language='c++',
    extra_compile_args=extra_compile_args,
    include_dirs=include_dirs,
    extra_link_args=extra_link_args + [make_relative_rpath('lib')]
)
extensions.append(C)

THNN = Extension("torch._thnn._THNN",
    libraries=['TH', 'THNN'],
    sources=['torch/csrc/nn/THNN.cpp'],
    language='c++',
    extra_compile_args=extra_compile_args,
    include_dirs=include_dirs,
    extra_link_args=extra_link_args + [make_relative_rpath('../lib')]
)
extensions.append(THNN)

if WITH_CUDA:
    THCUNN = Extension("torch._thnn._THCUNN",
        libraries=['TH', 'THC', 'THCUNN'],
        sources=['torch/csrc/nn/THCUNN.cpp'],
        language='c++',
        extra_compile_args=extra_compile_args,
        include_dirs=include_dirs,
        extra_link_args=extra_link_args + [make_relative_rpath('../lib')]
    )
    extensions.append(THCUNN)
    packages += ['torch.cuda', 'torch.legacy.cunn']

setup(name="torch", version="0.1",
    ext_modules=extensions,
    cmdclass = {
        'build': build,
        'build_ext': build_ext,
        'build_deps': build_deps,
        'build_module': build_module,
        'install': install,
        'clean': clean,
    },
    packages=packages,
    package_data={'torch': ['lib/*.so*', 'lib/*.dylib*', 'lib/*.h', 'lib/torch_shm_manager']},
    install_requires=['pyyaml'],
)
