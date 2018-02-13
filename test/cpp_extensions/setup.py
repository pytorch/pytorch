import torch.cuda
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, CUDAExtension

ext_modules = [
    CppExtension(
        'torch_test_cpp_extensions', ['extension.cpp'],
        extra_compile_args=['-g']),
]

if torch.cuda.is_available():
    extension = CUDAExtension(
        'torch_test_cuda_extension',
        ['cuda_extension.cpp', 'cuda_extension_kernel.cu'],
        extra_compile_args={'cxx': ['-g'],
                            'nvcc': ['-O2']})
    ext_modules.append(extension)

setup(
    name='torch_test_cpp_extensions',
    ext_modules=ext_modules,
    cmdclass={'build_ext': torch.utils.cpp_extension.BuildExtension})
