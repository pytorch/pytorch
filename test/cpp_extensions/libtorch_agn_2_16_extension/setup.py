from setuptools import find_packages, setup

from torch.utils.cpp_extension import BuildExtension, CppExtension


setup(
    name="libtorch_agn_2_16",
    version="0.0",
    packages=find_packages(),
    ext_modules=[
        CppExtension(
            "libtorch_agn_2_16._c10d",
            sources=["csrc/c10d.cpp"],
            py_limited_api=True,
            define_macros=[("TORCH_TARGET_VERSION", "0x0210000000000000")],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    options={"bdist_wheel": {"py_limited_api": "cp310"}},
)
