from setuptools import setup, Extension, distutils
from tools.nnwrap import generate_wrappers as generate_nn_wrappers
from tools.cwrap import cwrap
from tools.cwrap.plugins.THPPlugin import THPPlugin
from tools.cwrap.plugins.THPLongArgsPlugin import THPLongArgsPlugin
from tools.cwrap.plugins.ArgcountSortPlugin import ArgcountSortPlugin
from tools.cwrap.plugins.AutoGPU import AutoGPU
import platform
import subprocess
import sys
import os

# TODO: detect CUDA
WITH_CUDA = True
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
# Build libraries
################################################################################

if subprocess.call(['bash', 'torch/lib/build_all.sh'] + (['--with-cuda'] if WITH_CUDA else [])) != 0:
   sys.exit(1)

################################################################################
# Generate cpp code
################################################################################

cwrap('torch/csrc/generic/TensorMethods.cwrap', plugins=[THPLongArgsPlugin(), THPPlugin(), ArgcountSortPlugin(), AutoGPU()])
generate_nn_wrappers()

################################################################################
# Configure compile flags
################################################################################

include_dirs = []
extra_link_args = []
extra_compile_args = ['-std=c++11']

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

main_libraries = ['TH']
main_sources = [
    "torch/csrc/Module.cpp",
    "torch/csrc/Generator.cpp",
    "torch/csrc/Tensor.cpp",
    "torch/csrc/Storage.cpp",
    "torch/csrc/utils.cpp",
]

try:
    import numpy as np
    include_dirs += [np.get_include()]
    main_sources += ["torch/csrc/numpy.cpp"]
    extra_compile_args += ['-DWITH_NUMPY']
except ImportError:
    pass

if WITH_CUDA:
    if platform.system() == 'Darwin':
        include_dirs += ['/Developer/NVIDIA/CUDA-7.5/include']
    else:
        include_dirs += ['/usr/local/cuda/include']
    extra_compile_args += ['-DWITH_CUDA']
    main_libraries += ['THC']
    main_sources += [
        "torch/csrc/cuda/Module.cpp",
        "torch/csrc/cuda/Storage.cpp",
        "torch/csrc/cuda/Tensor.cpp",
        "torch/csrc/cuda/utils.cpp",
    ]

if DEBUG:
    extra_compile_args += ['-O0', '-g']
    extra_link_args += ['-O0', '-g']

################################################################################
# Declare extensions and the package
################################################################################

extensions = []

C = Extension("torch._C",
    libraries=main_libraries,
    sources=main_sources,
    language='c++',
    extra_compile_args=extra_compile_args,
    include_dirs=include_dirs,
    extra_link_args=extra_link_args + ['-Wl,-rpath,$ORIGIN/lib'],
)
extensions.append(C)

THNN = Extension("torch._thnn._THNN",
    libraries=['TH', 'THNN'],
    sources=['torch/csrc/nn/THNN.cpp'],
    language='c++',
    extra_compile_args=extra_compile_args,
    include_dirs=include_dirs,
    extra_link_args=extra_link_args + ['-Wl,-rpath,$ORIGIN/../lib'],
)
extensions.append(THNN)

if WITH_CUDA:
    THCUNN = Extension("torch._thnn._THCUNN",
        libraries=['TH', 'THC', 'THCUNN'],
        sources=['torch/csrc/nn/THCUNN.cpp'],
        language='c++',
        extra_compile_args=extra_compile_args,
        include_dirs=include_dirs,
        extra_link_args=extra_link_args + ['-Wl,-rpath,$ORIGIN/../lib'],
    )
    extensions.append(THCUNN)

setup(name="torch", version="0.1",
    ext_modules=extensions,
    packages=['torch', 'torch._thnn', 'torch.legacy', 'torch.legacy.nn', 'torch.legacy.optim'] + (['torch.cuda', 'torch.legacy.cunn'] if WITH_CUDA else []),
    package_data={'torch': ['lib/*.so*', 'lib/*.h']},
    install_requires=['pyyaml'],
)
