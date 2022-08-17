# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import distutils.command.clean
import shutil
import glob
import os
import subprocess
from setuptools import setup, find_packages
from torch.utils.cpp_extension import (
    CppExtension,
    BuildExtension,
)

cwd = os.path.dirname(os.path.abspath(__file__))
version_txt = os.path.join(cwd, 'version.txt')
with open(version_txt, 'r') as f:
    version = f.readline().strip()

try:
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=cwd).decode('ascii').strip()
except Exception:
    sha = 'Unknown'
package_name = 'functorch'

if os.getenv('BUILD_VERSION'):
    version = os.getenv('BUILD_VERSION')
elif sha != 'Unknown':
    version += '+' + sha[:7]


def write_version_file():
    version_path = os.path.join(cwd, 'functorch', 'version.py')
    with open(version_path, 'w') as f:
        f.write("__version__ = '{}'\n".format(version))
        f.write("git_version = {}\n".format(repr(sha)))


# pytorch_dep = 'torch'
# if os.getenv('PYTORCH_VERSION'):
#     pytorch_dep += "==" + os.getenv('PYTORCH_VERSION')
requirements = [
    # This represents a nightly version of PyTorch.
    # It can be installed as a binary or from source.
    "torch>=1.13.0.dev",
]

extras = {}
extras["aot"] = ["networkx", ]


class clean(distutils.command.clean.clean):
    def run(self):

        with open(".gitignore", "r") as f:
            ignores = f.read()
            for wildcard in filter(None, ignores.split("\n")):
                for filename in glob.glob(wildcard):
                    try:
                        os.remove(filename)
                    except OSError:
                        shutil.rmtree(filename, ignore_errors=True)

        # It's an old-style class in Python 2.7...
        distutils.command.clean.clean.run(self)


def get_extensions():
    extension = CppExtension

    # See functorch/csrc/Macros.h
    define_macros = [('FUNCTORCH_BUILD_MAIN_LIB', None)]

    extra_link_args = []
    extra_compile_args = {"cxx": [
        "-O3",
        "-std=c++14",
        "-fdiagnostics-color=always",
    ]}
    debug_mode = os.getenv('DEBUG', '0') == '1'
    if debug_mode:
        print("Compiling in debug mode")
        extra_compile_args = {
            "cxx": [
                "-O0",
                "-fno-inline",
                "-g",
                "-std=c++14",
                "-fdiagnostics-color=always",
            ]}
        extra_link_args = ["-O0", "-g"]

    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "functorch", "csrc")

    extension_sources = set(
        os.path.join(extensions_dir, p)
        for p in glob.glob(os.path.join(extensions_dir, "*.cpp"))
    )
    sources = list(extension_sources)
    sources.append(os.path.join(extensions_dir, "dim", "dim.cpp"))

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


class BuildExtension_(BuildExtension):
    def build_extensions(self, *args, **kwargs):
        # It turns out for windows this isn't populated?
        if hasattr(self.compiler, 'compiler_so'):
            if '-Wstrict-prototypes' in self.compiler.compiler_so:
                self.compiler.compiler_so.remove('-Wstrict-prototypes')
        super().build_extensions(*args, **kwargs)


if __name__ == '__main__':
    print("Building wheel {}-{}".format(package_name, version))
    write_version_file()
    setup(
        # Metadata
        name=package_name,
        version=version,
        author='PyTorch Core Team',
        url="https://github.com/pytorch/functorch",
        description='JAX-like composable function transforms for PyTorch',
        license='BSD',

        # Package info
        packages=find_packages(),
        install_requires=requirements,
        extras_require=extras,
        ext_modules=get_extensions(),
        cmdclass={
            "build_ext": BuildExtension_.with_options(no_python_abi_suffix=True),
            'clean': clean,
        })
