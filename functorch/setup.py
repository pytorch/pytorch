import distutils
import shutil
import glob
import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import (
    CppExtension,
    BuildExtension,
)


# class clean(distutils.command.clean.clean):
#     def run(self):
#         with open(".gitignore", "r") as f:
#             ignores = f.read()
#             for wildcard in filter(None, ignores.split("\n")):
#                 for filename in glob.glob(wildcard):
#                     try:
#                         os.remove(filename)
#                     except OSError:
#                         shutil.rmtree(filename, ignore_errors=True)
# 
#         # It's an old-style class in Python 2.7...
#         distutils.command.clean.clean.run(self)


def get_extensions():
    extension = CppExtension

    define_macros = []

    extra_link_args = []
    extra_compile_args = {"cxx": ["-O3", "-g", "-std=c++14"]}
    if int(os.environ.get("DEBUG", 0)):
    # if True:
        extra_compile_args = {
            "cxx": ["-O0", "-fno-inline", "-g", "-std=c++14"]}
        extra_link_args = ["-O0", "-g"]

    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "functorch", "csrc")

    extension_sources = set(
        os.path.join(extensions_dir, p)
        for p in glob.glob(os.path.join(extensions_dir, "*.cpp"))
    )
    sources = list(extension_sources)
    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            "functorch._C",
            sources,
            include_dirs=[this_dir],
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ]

    return ext_modules


setup(
    name='functorch',
    url="https://github.com/zou3519/functorch",
    packages=find_packages(),
    ext_modules=get_extensions(),
    cmdclass={
        # "clean": clean,
        "build_ext": BuildExtension
    })
