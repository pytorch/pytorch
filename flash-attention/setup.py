# Adapted from https://github.com/NVIDIA/apex/blob/master/setup.py
import sys
import warnings
import os
from pathlib import Path

from setuptools import setup, find_packages
import subprocess

import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


# ninja build does not work unless include_dirs are abs path
this_dir = os.path.dirname(os.path.abspath(__file__))


def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    release = output[release_idx].split(".")
    bare_metal_major = release[0]
    bare_metal_minor = release[1][0]

    return raw_output, bare_metal_major, bare_metal_minor


def check_cuda_torch_binary_vs_bare_metal(cuda_dir):
    raw_output, bare_metal_major, bare_metal_minor = get_cuda_bare_metal_version(cuda_dir)
    torch_binary_major = torch.version.cuda.split(".")[0]
    torch_binary_minor = torch.version.cuda.split(".")[1]

    print("\nCompiling cuda extensions with")
    print(raw_output + "from " + cuda_dir + "/bin\n")

    if (bare_metal_major != torch_binary_major) or (bare_metal_minor != torch_binary_minor):
        raise RuntimeError(
            "Cuda extensions are being compiled with a version of Cuda that does "
            "not match the version used to compile Pytorch binaries.  "
            "Pytorch binaries were compiled with Cuda {}.\n".format(torch.version.cuda)
            + "In some cases, a minor-version mismatch will not cause later errors:  "
            "https://github.com/NVIDIA/apex/pull/323#discussion_r287021798.  "
            "You can try commenting out this check (at your own risk)."
        )


def raise_if_cuda_home_none(global_option: str) -> None:
    if CUDA_HOME is not None:
        return
    raise RuntimeError(
        f"{global_option} was requested, but nvcc was not found.  Are you sure your environment has nvcc available?  "
        "If you're installing within a container from https://hub.docker.com/r/pytorch/pytorch, "
        "only images whose names contain 'devel' will provide nvcc."
    )


def append_nvcc_threads(nvcc_extra_args):
    _, bare_metal_major, bare_metal_minor = get_cuda_bare_metal_version(CUDA_HOME)
    if int(bare_metal_major) >= 11 and int(bare_metal_minor) >= 2:
        return nvcc_extra_args + ["--threads", "4"]
    return nvcc_extra_args


if not torch.cuda.is_available():
    # https://github.com/NVIDIA/apex/issues/486
    # Extension builds after https://github.com/pytorch/pytorch/pull/23408 attempt to query torch.cuda.get_device_capability(),
    # which will fail if you are compiling in an environment without visible GPUs (e.g. during an nvidia-docker build command).
    print(
        "\nWarning: Torch did not find available GPUs on this system.\n",
        "If your intention is to cross-compile, this is not an error.\n"
        "By default, We cross-compile for Volta (compute capability 7.0), "
        "Turing (compute capability 7.5),\n"
        "and, if the CUDA version is >= 11.0, Ampere (compute capability 8.0).\n"
        "If you wish to cross-compile for a single specific architecture,\n"
        'export TORCH_CUDA_ARCH_LIST="compute capability" before running setup.py.\n',
    )
    if os.environ.get("TORCH_CUDA_ARCH_LIST", None) is None:
        _, bare_metal_major, bare_metal_minor = get_cuda_bare_metal_version(CUDA_HOME)
        if int(bare_metal_major) == 11:
            os.environ["TORCH_CUDA_ARCH_LIST"] = "7.0;7.5;8.0"
            if int(bare_metal_minor) > 0:
                os.environ["TORCH_CUDA_ARCH_LIST"] = "7.0;7.5;8.0;8.6"
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = "7.0;7.5"

print("\n\ntorch.__version__  = {}\n\n".format(torch.__version__))
TORCH_MAJOR = int(torch.__version__.split(".")[0])
TORCH_MINOR = int(torch.__version__.split(".")[1])

cmdclass = {}
ext_modules = []

# Check, if ATen/CUDAGeneratorImpl.h is found, otherwise use ATen/cuda/CUDAGeneratorImpl.h
# See https://github.com/pytorch/pytorch/pull/70650
generator_flag = []
torch_dir = torch.__path__[0]
if os.path.exists(os.path.join(torch_dir, "include", "ATen", "CUDAGeneratorImpl.h")):
    generator_flag = ["-DOLD_GENERATOR_PATH"]

raise_if_cuda_home_none("flash_attn")
# Check, if CUDA11 is installed for compute capability 8.0
cc_flag = []
_, bare_metal_major, _ = get_cuda_bare_metal_version(CUDA_HOME)
if int(bare_metal_major) < 11:
    raise RuntimeError("FlashAttention is only supported on CUDA 11")
cc_flag.append("-gencode")
cc_flag.append("arch=compute_75,code=sm_75")
cc_flag.append("-gencode")
cc_flag.append("arch=compute_80,code=sm_80")

subprocess.run(["git", "submodule", "update", "--init", "csrc/flash_attn/cutlass"])
ext_modules.append(
    CUDAExtension(
        name="flash_attn_cuda",
        # name="flash_attn_cuda_cutlass",
        sources=[
            "csrc/flash_attn/fmha_api.cpp",
            "csrc/flash_attn/src/fmha_fprop_kernel_dispatch.cu",
        ],
        extra_compile_args={
            "cxx": ["-O3"] + generator_flag,
            "nvcc": append_nvcc_threads(
                [
                    "-O3",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "--use_fast_math",
                    "--ptxas-options=-v",
                    "-lineinfo"
                ]
                + generator_flag
                + cc_flag
            ),
        },
        include_dirs=[
            Path(this_dir) / 'csrc' / 'flash_attn',
            Path(this_dir) / 'csrc' / 'flash_attn' / 'src',
            Path(this_dir) / 'csrc' / 'flash_attn' / 'cutlass' / 'include',
        ],
    )
)

setup(
    name="flash_attn",
    version="0.1",
    packages=find_packages(
        exclude=("build", "csrc", "include", "tests", "dist", "docs", "benchmarks", "flash_attn.egg-info",)
    ),
    author="Tri Dao",
    author_email="trid@stanford.edu",
    description="Flash Attention: Fast and Memory-Efficient Exact Attention",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/HazyResearch/flash-attention",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Apache 2.0",
        "Operating System :: Linux",
    ],
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension} if ext_modules else {},
    python_requires=">=3.7",
    install_requires=[
        "torch",
        "einops",
    ],
)
