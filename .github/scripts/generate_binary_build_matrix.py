#!/usr/bin/env python3

"""Generates a matrix to be utilized through github actions

Will output a condensed version of the matrix if on a pull request that only
includes the latest version of python we support built on three different
architectures:
    * CPU
    * Latest CUDA
    * Latest ROCM
"""

import argparse
import os
import json
from typing import Dict, List


CUDA_ARCHES = ["10.2", "11.1", "11.3", "11.5"]


ROCM_ARCHES = ["4.2", "4.3.1"]


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
    "3.8",
    "3.9",
]


def is_pull_request() -> bool:
    return os.environ.get("GITHUB_HEAD_REF", None) != ""


def snip_if(is_pr: bool, versions: List[str]) -> List[str]:
    """
    Return the full list of versions, or just the latest if on a PR.
    """
    return [versions[-1]] if is_pr else versions


def generate_conda_matrix(is_pr: bool) -> List[Dict[str, str]]:
    return [
        {
            "python_version": python_version,
            "gpu_arch_type": arch_type(arch_version),
            "gpu_arch_version": "" if arch_version == "cpu" else arch_version,
            "container_image": CONDA_CONTAINER_IMAGES[arch_version],
        }
        for python_version in FULL_PYTHON_VERSIONS
        # We don't currently build conda packages for rocm
        for arch_version in ["cpu"] + CUDA_ARCHES
    ]


def generate_libtorch_matrix(is_pr: bool) -> List[Dict[str, str]]:
    libtorch_variants = [
        "shared-with-deps",
        "shared-without-deps",
        "static-with-deps",
        "static-without-deps",
    ]
    return [
        {
            "gpu_arch_type": arch_type(arch_version),
            "gpu_arch_version": "" if arch_version == "cpu" else arch_version,
            "libtorch_variant": libtorch_variant,
            "devtoolset": abi_version,
            "container_image": LIBTORCH_CONTAINER_IMAGES[(arch_version, abi_version)],
        }
        # We don't currently build libtorch for rocm
        for arch_version in ["cpu"] + CUDA_ARCHES
        for libtorch_variant in libtorch_variants
        # one of the values in the following list must be exactly
        # "cxx11-abi", but the precise value of the other one doesn't
        # matter
        for abi_version in ["cxx11-abi", "pre-cxx11"]
    ]


def generate_wheels_matrix(is_pr: bool) -> List[Dict[str, str]]:
    arches = ["cpu"] + CUDA_ARCHES + ROCM_ARCHES
    return [
        {
            "python_version": python_version,
            "gpu_arch_type": arch_type(arch_version),
            "gpu_arch_version": "" if arch_version == "cpu" else arch_version,
            "container_image": WHEEL_CONTAINER_IMAGES[arch_version],
        }
        for python_version in FULL_PYTHON_VERSIONS
        for arch_version in arches
    ]


def from_includes(includes: List[Dict[str, str]]) -> str:
    return json.dumps({"include": includes})


def main() -> None:
    modes = {
        "conda": generate_conda_matrix,
        "libtorch": generate_libtorch_matrix,
        "manywheel": generate_wheels_matrix,
    }
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=modes.keys())
    args = parser.parse_args()

    is_pr = is_pull_request()
    print(from_includes(modes[args.mode](is_pr)))


if __name__ == "__main__":
    main()
