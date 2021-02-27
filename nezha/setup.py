from setuptools import setup
import os
import glob
from torch.utils.cpp_extension import BuildExtension, CppExtension

#Header file directory
include_dirs=os.path.dirname(os.path.abspath (__file__))

#Source code directory
source_cpu=glob.glob(os.path.join(include_dirs, "core_func.cpp"))
setup(
  name="nezha_helper", #module name,Need to be called in python
  version="0.1", ext_modules=[
    CppExtension("nezha_helper", sources=source_cpu, include_dirs=[include_dirs]), ], cmdclass={
    "build_ext":BuildExtension
  }
)