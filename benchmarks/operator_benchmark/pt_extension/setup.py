from setuptools import setup

from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(name='cpp_extension',
      ext_modules=[CppExtension('cpp_extension', ['extension.cpp'])],
      cmdclass={'build_ext': BuildExtension})
