# Copyright (c) Meta Platforms, Inc. and affiliates

import argparse
import distutils.command.clean
import glob
import os
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

import setuptools


ROOT_DIR = Path(__file__).parent.resolve()


def _get_version(nightly=False, release=False):
    version = "0.12.0"
    sha = "Unknown"
    try:
        sha = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT_DIR))
            .decode("ascii")
            .strip()
        )
    except Exception:
        pass

    if nightly:
        today = date.today()
        version = version[:-2] + ".dev" + f"{today.year}{today.month}{today.day}"
    elif release:
        version = version[:-2]
    else:
        os_build_version = os.getenv("BUILD_VERSION")
        if os_build_version:
            version = os_build_version
        elif sha != "Unknown":
            version += "+" + sha[:7]

    return version, sha


def _export_version(version, sha):
    version_path = ROOT_DIR / "maskedtensor" / "version.py"
    with open(version_path, "w") as f:
        f.write(f"__version__ = '{version}'\n")
        f.write(f"git_version = {repr(sha)}\n")


def get_parser():
    parser = argparse.ArgumentParser(description="MaskedTensor setup")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--nightly",
        action="store_true",
        help="Nightly Release",
    )
    group.add_argument(
        "--release",
        action="store_true",
        help="Official/RC Release",
    )
    return parser


class clean(distutils.command.clean.clean):
    def run(self):
        with open(".gitignore") as f:
            ignores = f.read()
            for wildcard in filter(None, ignores.split("\n")):
                for filename in glob.glob(wildcard):
                    try:
                        os.remove(filename)
                    except OSError:
                        shutil.rmtree(filename, ignore_errors=True)

        # It's an old-style class in Python 2.7...
        distutils.command.clean.clean.run(self)


requirements = [
    "torch>1.12.0",
]
readme = open("README.md").read()


if __name__ == "__main__":
    args, unknown = get_parser().parse_known_args()

    VERSION, SHA = _get_version(args.nightly, args.release)
    _export_version(VERSION, SHA)

    print("-- Building version " + VERSION)

    sys.argv = [sys.argv[0]] + unknown
    # Commented out sections we may need later on to enable C++ extension
    setuptools.setup(
        # Metadata
        name="maskedtensor_nightly" if args.nightly else "maskedtensor",
        version=VERSION,
        url="https://github.com/pytorch/maskedtensor",
        description="MaskedTensors for PyTorch",
        long_description=readme,
        long_description_content_type="text/markdown",
        author="PyTorch Team",
        author_email="packages@pytorch.org",
        install_requires=requirements,
        python_requires=">=3.7",
        classifiers=[
            "Programming Language :: Python :: 3",
            "Operating System :: OS Independent",
        ],
        # cmdclass={
        #     "clean": clean,
        #     # "build_ext": BuildExtension.with_options(no_python_abi_suffix=True,),
        # },
        # ext_modules=get_extensions(),
        # Package Info
        packages=setuptools.find_packages(exclude=["test*", "examples*"]),
        zip_safe=False,
    )
