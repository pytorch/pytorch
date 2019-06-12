import sys
import torch.cuda
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, CUDAExtension
from torch.utils.cpp_extension import CUDA_HOME

CXX_FLAGS = [] if sys.platform == 'win32' else ['-g', '-Werror']

ext_modules = [
    CppExtension(
        'torch_test_cpp_extension.cpp', ['extension.cpp'],
        extra_compile_args=CXX_FLAGS),
]

if torch.cuda.is_available() and CUDA_HOME is not None:
    extension = CUDAExtension(
        'torch_test_cpp_extension.cuda', [
            'cuda_extension.cpp',
            'cuda_extension_kernel.cu',
            'cuda_extension_kernel2.cu',
        ],
        extra_compile_args={'cxx': CXX_FLAGS,
                            'nvcc': ['-O2']})
    ext_modules.append(extension)

setup(
    name='torch_test_cpp_extension',
    packages=['torch_test_cpp_extension'],
    ext_modules=ext_modules,
    cmdclass={'build_ext': torch.utils.cpp_extension.BuildExtension})
