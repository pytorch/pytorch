from setuptools import setup, Extension
import torch.utils.cpp_extension

ext_modules = [
    Extension(
        'torch_test_cpp_extensions', ['extension.cpp'],
        include_dirs=torch.utils.cpp_extension.include_paths(),
        language='c++'),
]

setup(
    name='torch_test_cpp_extensions',
    ext_modules=ext_modules,
    cmdclass={'build_ext': torch.utils.cpp_extension.BuildExtension})
