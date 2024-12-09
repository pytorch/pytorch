from setuptools import setup

from torch.utils.cpp_extension import BuildExtension, CppExtension


setup(
    name="no_python_abi_suffix_test",
    ext_modules=[
        CppExtension("no_python_abi_suffix_test", ["no_python_abi_suffix_test.cpp"])
    ],
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
)
