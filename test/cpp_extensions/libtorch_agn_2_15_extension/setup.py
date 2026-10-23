import distutils.command.clean
import shutil
from pathlib import Path

from setuptools import find_packages, setup

from torch.utils.cpp_extension import BuildExtension, CppExtension, IS_WINDOWS


ROOT_DIR = Path(__file__).parent
CSRC_DIR = ROOT_DIR / "csrc"


class clean(distutils.command.clean.clean):
    def run(self):
        # Run default behavior first
        distutils.command.clean.clean.run(self)

        # Remove extension
        for path in (ROOT_DIR / "libtorch_agn_2_15").glob("**/*.so"):
            path.unlink()
        # Remove build and dist and egg-info directories
        dirs = [
            ROOT_DIR / "build",
            ROOT_DIR / "dist",
            ROOT_DIR / "libtorch_agn_2_15.egg-info",
        ]
        for path in dirs:
            if path.exists():
                shutil.rmtree(str(path), ignore_errors=True)


def get_extension():
    common_cxx = ["-DTORCH_TARGET_VERSION=0x020f000000000000"]
    if not IS_WINDOWS:
        common_cxx.append("-fdiagnostics-color=always")

    # No STABLE_TORCH_LIBRARY ops yet, only the interop module: an importable
    # abi3 module, imported rather than loaded via torch.ops.load_library. An op
    # extension (_C) inheriting the 2.9-2.14 csrc can be added when the first
    # 2.15 op lands, mirroring libtorch_agn_2_14_extension.
    return [
        CppExtension(
            "libtorch_agn_2_15._interop",
            sources=[str(CSRC_DIR / "pyobject_interop_module.cpp")],
            py_limited_api=True,
            extra_compile_args={
                "cxx": common_cxx + ["-DSTABLE_LIB_NAME=libtorch_agn_2_15_interop"]
            },
        ),
    ]


setup(
    name="libtorch_agn_2_15",
    version="0.0",
    author="PyTorch Core Team",
    description="Example of libtorch agnostic extension for PyTorch 2.15+",
    packages=find_packages(exclude=("test",)),
    package_data={"libtorch_agn_2_15": ["*.dll", "*.dylib", "*.so"]},
    install_requires=[
        "torch",
    ],
    ext_modules=get_extension(),
    cmdclass={
        "build_ext": BuildExtension.with_options(no_python_abi_suffix=True),
        "clean": clean,
    },
    options={"bdist_wheel": {"py_limited_api": "cp310"}},
)
