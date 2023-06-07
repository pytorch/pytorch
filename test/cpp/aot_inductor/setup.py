from setuptools import setup, Extension
import os
import torch

library_path = os.path.dirname(os.path.realpath(__file__))


inductor_module = Extension(
    'inductor_module', 
    sources = ['inductor_module.cpp'],
    library_dirs=[library_path], 
    libraries=['aot_inductor_output'],
    extra_compile_args=['-std=c++14']
)

setup(
    name='inductor_module',
    version='0.0.1',
    ext_modules=[inductor_module]
)
