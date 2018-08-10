import torch
from setuptools import setup
from torch.utils.cpp_extension import CppExtension


setup(
    name="custom_op",
    ext_modules=[CppExtension("custom_op", ["op.cpp"], extra_compile_args=["-DWITH_PYTHON_OP_BINDINGS"])],
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)
