import distutils.command.clean
import shutil
from pathlib import Path

from setuptools import find_packages, setup

from torch.utils.cpp_extension import BuildExtension, CppExtension


ROOT_DIR = Path(__file__).parent
CSRC_DIR = ROOT_DIR / "torch_stable_test" / "csrc"


class clean(distutils.command.clean.clean):
    def run(self):
        # Run default behavior first
        distutils.command.clean.clean.run(self)

        # Remove extension
        for path in (ROOT_DIR / "torch_stable_test").glob("**/*.so"):
            path.unlink()
        # Remove build and dist and egg-info directories
        dirs = [
            ROOT_DIR / "build",
            ROOT_DIR / "dist",
            ROOT_DIR / "torch_stable_test.egg-info",
        ]
        for path in dirs:
            if path.exists():
                shutil.rmtree(str(path), ignore_errors=True)


def get_extension():
    extra_compile_args = {
        "cxx": ["-fdiagnostics-color=always", "-DTORCH_STABLE_ONLY"],
    }

    sources = list(CSRC_DIR.glob("**/*.cpp"))

    return [
        CppExtension(
            "torch_stable_test._C",
            sources=sorted(str(s) for s in sources),
            py_limited_api=True,
            extra_compile_args=extra_compile_args,
            extra_link_args=[],
        )
    ]


setup(
    name="torch_stable_test",
    version="0.0",
    author="PyTorch Core Team",
    description="Test extension to verify TORCH_STABLE_ONLY flag",
    packages=find_packages(exclude=("test",)),
    package_data={"torch_stable_test": ["*.dll", "*.dylib", "*.so"]},
    install_requires=[
        "torch",
    ],
    ext_modules=get_extension(),
    cmdclass={
        "build_ext": BuildExtension.with_options(no_python_abi_suffix=True),
        "clean": clean,
    },
    options={"bdist_wheel": {"py_limited_api": "cp39"}},
)
