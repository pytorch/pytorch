# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='inductor_module',
    ext_modules=[
        CppExtension('inductor_module', ['inductor_module.cpp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })