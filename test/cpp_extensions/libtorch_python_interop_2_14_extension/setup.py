import distutils.command.clean
import shutil
from pathlib import Path

from setuptools import find_packages, setup

from torch.utils.cpp_extension import BuildExtension, CppExtension, IS_WINDOWS


ROOT_DIR = Path(__file__).parent
CSRC_DIR = ROOT_DIR / "csrc"
PACKAGE = "libtorch_python_interop_2_14"


class clean(distutils.command.clean.clean):
    def run(self):
        distutils.command.clean.clean.run(self)
        for path in (ROOT_DIR / PACKAGE).glob("**/*.so"):
            path.unlink()
        for path in (
            ROOT_DIR / "build",
            ROOT_DIR / "dist",
            ROOT_DIR / f"{PACKAGE}.egg-info",
        ):
            if path.exists():
                shutil.rmtree(str(path), ignore_errors=True)


def get_extension():
    extra_compile_args = {
        "cxx": [
            "-DTORCH_TARGET_VERSION=0x020e000000000000",
            f"-DSTABLE_LIB_NAME={PACKAGE}",
        ],
    }
    if not IS_WINDOWS:
        extra_compile_args["cxx"].append("-fdiagnostics-color=always")

    sources = sorted(str(s) for s in CSRC_DIR.glob("**/*.cpp"))

    # This extension is py_limited_api (abi3) like the libtorch_agn_* family:
    # the module only uses the stable Python C API and the stable void* shim,
    # so it stays portable across Python versions. It differs in one way: the
    # python-interop shims live in libtorch_python, which py_limited_api does
    # not link automatically, so torch_python is added explicitly.
    return [
        CppExtension(
            f"{PACKAGE}._C",
            sources=sources,
            py_limited_api=True,
            libraries=["torch_python"],
            extra_compile_args=extra_compile_args,
            extra_link_args=[],
        )
    ]


setup(
    name=PACKAGE,
    version="0.0",
    author="PyTorch Core Team",
    description="Test extension for the python-aware (libtorch_python) stable "
    "shims, PyTorch 2.14+",
    packages=find_packages(exclude=("test",)),
    package_data={PACKAGE: ["*.dll", "*.dylib", "*.so"]},
    install_requires=["torch"],
    ext_modules=get_extension(),
    cmdclass={
        "build_ext": BuildExtension.with_options(no_python_abi_suffix=True),
        "clean": clean,
    },
    options={"bdist_wheel": {"py_limited_api": "cp39"}},
)
