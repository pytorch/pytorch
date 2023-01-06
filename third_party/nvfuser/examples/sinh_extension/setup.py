from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

import os

nvfuser_csrc_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "csrc")

setup(
    name='nvfuser_extension',
    ext_modules=[
        CUDAExtension(
            name='nvfuser_extension',
            pkg='nvfuser_extension',
            include_dirs=[nvfuser_csrc_dir],
            libraries=['nvfuser_codegen'],
            sources=['main.cpp'])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
