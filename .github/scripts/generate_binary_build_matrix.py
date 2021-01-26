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
    "11.0"
]

ROCM_ARCHES = [
    "3.10",
    "4.0"
]

FULL_ARCHES = [
    "cpu",
    *CUDA_ARCHES,
    *ROCM_ARCHES
]

CONTAINER_IMAGES = {
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

FULL_PYTHON_VERSIONS = [
    "3.6",
    "3.7",
    "3.8",
    "3.9",
]


def is_pull_request():
    return os.environ.get("GITHUB_HEAD_REF")

def generate_matrix():
    python_versions = FULL_PYTHON_VERSIONS
    arches = FULL_ARCHES
    if is_pull_request():
        python_versions = [python_versions[-1]]
        arches = ["cpu", CUDA_ARCHES[-1], ROCM_ARCHES[-1]]
    matrix = []
    for item in itertools.product(python_versions, arches):
        python_version, arch_version = item
        # Not my favorite code here
        gpu_arch_type = "cuda"
        if "rocm" in CONTAINER_IMAGES[arch_version]:
            gpu_arch_type = "rocm"
        elif "cpu" in CONTAINER_IMAGES[arch_version]:
            gpu_arch_type = "cpu"
        matrix.append({
            "python_version": python_version,
            "gpu_arch_type": gpu_arch_type,
            "gpu_arch_version": arch_version,
            "container_image": CONTAINER_IMAGES[arch_version]
        })
    return json.dumps({"include": matrix})

def main():
    print(generate_matrix())

if __name__ == "__main__":
    main()
