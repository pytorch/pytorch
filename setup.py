from setuptools import setup, Extension, distutils
from os.path import expanduser
from tools.cwrap import cwrap
import platform

# TODO: detect CUDA
WITH_CUDA = False
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
# Generate Tensor methods
################################################################################

cwrap_src = ['torch/csrc/generic/TensorMethods.cwrap.cpp']
for src in cwrap_src:
    print("Generating code for " + src)
    cwrap(src)

################################################################################
# Declare the package
################################################################################

extra_link_args = []

# TODO: remove and properly submodule TH in the repo itself
th_path = expanduser("~/torch/install/")
torch_headers = th_path + "include"
th_header_path = th_path + "include/TH"
th_lib_path = th_path + "lib"
extra_link_args.append('-L' + th_lib_path)
extra_link_args.append('-Wl,-rpath,' + th_lib_path)

libraries = ['TH']
extra_compile_args = ['-std=c++11']
sources = [
    "torch/csrc/Module.cpp",
    "torch/csrc/Generator.cpp",
    "torch/csrc/Tensor.cpp",
    "torch/csrc/Storage.cpp",
    "torch/csrc/utils.cpp",
]

if WITH_CUDA:
    libraries += ['THC']
    extra_compile_args += ['-DWITH_CUDA']
    sources += [
        "torch/csrc/cuda/Module.cpp",
        "torch/csrc/cuda/Storage.cpp",
        "torch/csrc/cuda/Tensor.cpp",
        "torch/csrc/cuda/utils.cpp",
    ]

C = Extension("torch._C",
    libraries=libraries,
    sources=sources,
    language='c++',
    extra_compile_args=extra_compile_args + (['-O0', '-g'] if DEBUG else []),
    include_dirs=([".", "torch/csrc", "cutorch/csrc", torch_headers, th_header_path, "/Developer/NVIDIA/CUDA-7.5/include", "/usr/local/cuda/include"]),
    extra_link_args = extra_link_args + (['-O0', '-g'] if DEBUG else []),
)

setup(name="torch", version="0.1",
    ext_modules=[C],
    packages=['torch', 'torch.legacy', 'torch.legacy.nn', 'torch.legacy.optim'] + (['torch.cuda', 'torch.legacy.cunn'] if WITH_CUDA else []),
)
