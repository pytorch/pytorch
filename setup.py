from setuptools import setup, Extension, distutils
from os.path import expanduser
from tools.cwrap import cwrap
import platform

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
th_header_path = th_path + "include"
th_lib_path = th_path + "lib"
if platform.system() == 'Darwin':
    extra_link_args.append('-L' + th_lib_path)
    extra_link_args.append('-Wl,-rpath,' + th_lib_path)

sources = [
    "torch/csrc/Module.cpp",
    "torch/csrc/Generator.cpp",
    "torch/csrc/Tensor.cpp",
    "torch/csrc/Storage.cpp",
    "torch/csrc/utils.cpp",
]
C = Extension("torch._C",
              libraries=['TH'],
              sources=sources,
              language='c++',
              extra_compile_args=['-std=c++11'],
              include_dirs=(["torch/csrc", th_header_path]),
              extra_link_args = extra_link_args,
)



setup(name="torch", version="0.1",
      ext_modules=[C],
      packages=['torch'],
)
