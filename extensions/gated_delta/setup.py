"""Build the GDN CUDA kernel against an already-installed torch 2.13.

Compile on the GPU host with CUDA_VISIBLE_DEVICES empty so this does not
touch the cards that are running JEPA.
"""
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

arch = os.environ.get("TORCH_CUDA_ARCH_LIST", "12.0")
os.environ["TORCH_CUDA_ARCH_LIST"] = arch

setup(
    name="gated_delta_ext",
    version="0.0.1",
    ext_modules=[
        CUDAExtension(
            name="gated_delta_ext",
            sources=["gated_delta.cpp", "gated_delta_kernel.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
