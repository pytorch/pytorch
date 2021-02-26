#!/usr/bin/env python3

"""Generates a matrix to be utilized through github actions

Will output a condensed version of the matrix if on a pull request that only
includes the latest version of python we support built on three different
architectures:
    * CPU
    * Latest CUDA
    * Latest ROCM
"""

import json
import os
import itertools

CUDA_ARCHES = [
    "10.1",
    "10.2",
    "11.1"
]

ROCM_ARCHES = [
    "3.10",
    "4.0"
]

WHEEL_CONTAINER_IMAGES = {
    **{
        # TODO: Re-do manylinux CUDA image tagging scheme to be similar to
        #       ROCM so we don't have to do this replacement
        gpu_arch: f"pytorch/manylinux-cuda{gpu_arch.replace('.', '')}"
        for gpu_arch in CUDA_ARCHES
    },
    **{
        gpu_arch: f"pytorch/manylinux-rocm:{gpu_arch}"
        for gpu_arch in ROCM_ARCHES
    },
    "cpu": "pytorch/manylinux-cpu"
}

CONDA_CONTAINER_IMAGES = {
    **{
        # TODO: Re-do manylinux CUDA image tagging scheme to be similar to
        #       ROCM so we don't have to do this replacement
        gpu_arch: f"pytorch/conda-builder:cuda{gpu_arch}"
        for gpu_arch in CUDA_ARCHES
    },
    "cpu": "pytorch/conda-builder:cpu"
}

FULL_PYTHON_VERSIONS = [
    "3.6",
    "3.7",
    "3.8",
    "3.9",
]


def is_pull_request():
    return False
    # return os.environ.get("GITHUB_HEAD_REF")

def generate_matrix(include_cuda=True, include_rocm=False):
    python_versions = FULL_PYTHON_VERSIONS
    cuda_versions = CUDA_ARCHES
    rocm_versions = ROCM_ARCHES
    arches = ["cpu"]
    if is_pull_request():
        python_versions = [python_versions[-1]]
        cuda_versions = [cuda_versions[-1]]
        rocm_versions = [rocm_versions[-1]]
    if include_cuda:
        arches += cuda_versions
    if include_rocm:
        arches += rocm_versions
    matrix = []
    for item in itertools.product(python_versions, arches):
        python_version, arch_version = item
        # Not my favorite code here
        gpu_arch_type = "cuda"
        if "rocm" in WHEEL_CONTAINER_IMAGES[arch_version]:
            gpu_arch_type = "rocm"
        elif "cpu" in WHEEL_CONTAINER_IMAGES[arch_version]:
            gpu_arch_type = "cpu"
        matrix.append({
            "python_version": python_version,
            "gpu_arch_type": gpu_arch_type,
            "gpu_arch_version": arch_version,
            "wheel_container_image": WHEEL_CONTAINER_IMAGES.get(
                arch_version, ""
            ),
            "conda_container_image": CONDA_CONTAINER_IMAGES.get(
                arch_version, ""
            )
        })
    return json.dumps({"include": matrix})

def main():
    include_cuda = True
    include_rocm = True
    if os.environ.get("NO_ROCM", False):
        include_rocm = False
    if os.environ.get("NO_CUDA", False):
        include_cuda = False
    print(
        generate_matrix(
            include_cuda=include_cuda,
            include_rocm=include_rocm
        )
    )

if __name__ == "__main__":
    main()
