from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='nvfuser_extension',
    ext_modules=[
        CUDAExtension(
            name='nvfuser_extension',
            pkg='nvfuser_extension',
            sources=['main.cpp'])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
