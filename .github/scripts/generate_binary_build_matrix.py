#!/usr/bin/env python3

"""Generates a matrix to be utilized through github actions

Will output a condensed version of the matrix if on a pull request that only
includes the latest version of python we support built on three different
architectures:
    * CPU
    * Latest CUDA
    * Latest ROCM
"""

from typing import Dict, List


# CUDA_ARCHES = ["10.2", "11.1", "11.3", "11.5"]
CUDA_ARCHES = ["10.2", "11.5"]


# ROCM_ARCHES = ["4.2", "4.3.1"]
ROCM_ARCHES = ["4.3.1"]


def arch_type(arch_version: str) -> str:
    if arch_version in CUDA_ARCHES:
        return "cuda"
    elif arch_version in ROCM_ARCHES:
        return "rocm"
    else:  # arch_version should always be "cpu" in this case
        return "cpu"


WHEEL_CONTAINER_IMAGES = {
    **{
        gpu_arch: f"pytorch/manylinux-builder:cuda{gpu_arch}"
        for gpu_arch in CUDA_ARCHES
    },
    **{
        gpu_arch: f"pytorch/manylinux-builder:rocm{gpu_arch}"
        for gpu_arch in ROCM_ARCHES
    },
    "cpu": "pytorch/manylinux-builder:cpu",
}

CONDA_CONTAINER_IMAGES = {
    **{gpu_arch: f"pytorch/conda-builder:cuda{gpu_arch}" for gpu_arch in CUDA_ARCHES},
    "cpu": "pytorch/conda-builder:cpu",
}

LIBTORCH_CONTAINER_IMAGES = {
    **{
        (gpu_arch, "pre-cxx11"): f"pytorch/manylinux-builder:cuda{gpu_arch}"
        for gpu_arch in CUDA_ARCHES
    },
    **{
        (gpu_arch, "cxx11-abi"): f"pytorch/libtorch-cxx11-builder:cuda{gpu_arch}"
        for gpu_arch in CUDA_ARCHES
    },
    ("cpu", "pre-cxx11"): "pytorch/manylinux-builder:cpu",
    ("cpu", "cxx11-abi"): "pytorch/libtorch-cxx11-builder:cpu",
}

FULL_PYTHON_VERSIONS = [
    "3.7",
    # "3.8",
    # "3.9",
]


def translate_desired_cuda(gpu_arch_type: str, gpu_arch_version: str) -> str:
    return {
        "cpu": "cpu",
        "cuda": f"cu{gpu_arch_version.replace('.', '')}",
        "rocm": f"rocm{gpu_arch_version}"
    }.get(gpu_arch_type, gpu_arch_version)


def generate_conda_matrix() -> List[Dict[str, str]]:
    ret: List[Dict[str, str]] = []
    for python_version in FULL_PYTHON_VERSIONS:
        # We don't currently build conda packages for rocm
        for arch_version in ["cpu"] + CUDA_ARCHES:
            gpu_arch_type = arch_type(arch_version)
            gpu_arch_version = "" if arch_version == "cpu" else arch_version
            ret.append(
                {
                    "python_version": python_version,
                    "gpu_arch_type": gpu_arch_type,
                    "gpu_arch_version": gpu_arch_version,
                    "desired_cuda": translate_desired_cuda(
                        gpu_arch_type, gpu_arch_version
                    ),
                    "container_image": CONDA_CONTAINER_IMAGES[arch_version],
                    "package_type": "conda",
                    "build_name": f"conda-py{python_version}-{gpu_arch_type}{gpu_arch_version}".replace(
                        ".", "_"
                    ),
                }
            )
    return ret


def generate_libtorch_matrix() -> List[Dict[str, str]]:
    libtorch_variants = [
        "shared-with-deps",
        # "shared-without-deps",
        "static-with-deps",
        # "static-without-deps",
    ]
    ret: List[Dict[str, str]] = []
    for arch_version in ["cpu"] + CUDA_ARCHES:
        for libtorch_variant in libtorch_variants:
            for abi_version in ["cxx11-abi", "pre-cxx11"]:
                # We don't currently build libtorch for rocm
                # one of the values in the following list must be exactly
                # "cxx11-abi", but the precise value of the other one doesn't
                # matter
                gpu_arch_type = arch_type(arch_version)
                gpu_arch_version = "" if arch_version == "cpu" else arch_version
                ret.append(
                    {
                        "gpu_arch_type": gpu_arch_type,
                        "gpu_arch_version": gpu_arch_version,
                        "desired_cuda": translate_desired_cuda(
                            gpu_arch_type, gpu_arch_version
                        ),
                        "libtorch_variant": libtorch_variant,
                        "devtoolset": abi_version,
                        "container_image": LIBTORCH_CONTAINER_IMAGES[
                            (arch_version, abi_version)
                        ],
                        "package_type": "libtorch",
                        "build_name": f"libtorch-{gpu_arch_type}{gpu_arch_version}-{libtorch_variant}-{abi_version}".replace(
                            ".", "_"
                        ),
                    }
                )
    return ret


def generate_wheels_matrix() -> List[Dict[str, str]]:
    arches = ["cpu"] + CUDA_ARCHES + ROCM_ARCHES
    ret: List[Dict[str, str]] = []
    for python_version in FULL_PYTHON_VERSIONS:
        for arch_version in arches:
            gpu_arch_type = arch_type(arch_version)
            gpu_arch_version = "" if arch_version == "cpu" else arch_version
            ret.append(
                {
                    "python_version": python_version,
                    "gpu_arch_type": gpu_arch_type,
                    "gpu_arch_version": gpu_arch_version,
                    "desired_cuda": translate_desired_cuda(
                        gpu_arch_type, gpu_arch_version
                    ),
                    "container_image": WHEEL_CONTAINER_IMAGES[arch_version],
                    "package_type": "manywheel",
                    "build_name": f"manywheel-py{python_version}-{gpu_arch_type}{gpu_arch_version}".replace(
                        ".", "_"
                    ),
                }
            )
    return ret


def generate_binary_build_matrix(os: str) -> List[Dict[str, str]]:
    return [
        *generate_conda_matrix(),
        *generate_libtorch_matrix(),
        *generate_wheels_matrix(),
    ]
