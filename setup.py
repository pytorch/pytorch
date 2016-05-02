from setuptools import setup, Extension

C = Extension("torch.C",
              libraries=['TH'],
              sources=["torch/csrc/Module.c", "torch/csrc/Tensor.c", "torch/csrc/Storage.c"],
              include_dirs=["torch/csrc"])

setup(name="torch", version="0.1",
      ext_modules=[C],
      packages=['torch'])
