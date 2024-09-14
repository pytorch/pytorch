from setuptools import setup

from torch.utils.cpp_extension import BuildExtension, CppExtension


setup(
    name="benchmark_cpp_extension",
    ext_modules=[CppExtension("benchmark_cpp_extension", ["extension.cpp"])],
    cmdclass={"build_ext": BuildExtension},
)
