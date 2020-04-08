import sys
import torch.cuda
import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
from torch.utils.cpp_extension import CUDA_HOME, ROCM_HOME

if sys.platform == 'win32':
    vc_version = os.getenv('VCToolsVersion', '')
    if vc_version.startswith('14.16.'):
        CXX_FLAGS = ['/sdl']
    else:
        CXX_FLAGS = ['/sdl', '/permissive-']
else:
    CXX_FLAGS = ['-g']

USE_NINJA = os.getenv('USE_NINJA') == '1'

ext_modules = [
    CppExtension(
        'torch_test_cpp_extension.cpp', ['extension.cpp'],
        extra_compile_args=CXX_FLAGS),
    CppExtension(
        'torch_test_cpp_extension.msnpu', ['msnpu_extension.cpp'],
        extra_compile_args=CXX_FLAGS),
    CppExtension(
        'torch_test_cpp_extension.rng', ['rng_extension.cpp'],
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
elif torch.cuda.is_available() and ROCM_HOME is not None:
    from torch.utils.hipify import hipify_python
    this_dir = os.path.dirname(os.path.abspath(__file__))
    hipify_python.hipify(
        project_directory=this_dir,
        output_directory=this_dir,
        includes="./*",
        show_detailed=True,
        is_pytorch_extension=True,)
    extension = CUDAExtension(
        'torch_test_cpp_extension.cuda', [
            'cuda_extension.cpp',
            'hip/hip_extension_kernel.hip',
            'hip/hip_extension_kernel2.hip',
        ])
    ext_modules.append(extension)

setup(
    name='torch_test_cpp_extension',
    packages=['torch_test_cpp_extension'],
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension.with_options(use_ninja=USE_NINJA)})
