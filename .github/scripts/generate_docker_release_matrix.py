#!/usr/bin/env python3

"""Generates a matrix for docker releases through github actions

Will output a condensed version of the matrix. Will include fllowing:
    * CUDA version short
    * CUDA full version
    * CUDNN version short
    * Image type either runtime or devel
    * Platform linux/arm64,linux/amd64

"""

import json
from typing import Dict, List

import generate_binary_build_matrix

DOCKER_IMAGE_TYPES = ["runtime", "devel"]


def generate_docker_matrix() -> Dict[str, List[Dict[str, str]]]:
    ret: List[Dict[str, str]] = []
    # CUDA amd64 Docker images are available as both runtime and devel while
    # CPU arm64 image is only available as runtime.
    for cuda, version in generate_binary_build_matrix.CUDA_ARCHES_FULL_VERSION.items():
        for image in DOCKER_IMAGE_TYPES:
            ret.append(
                {
                    "cuda": cuda,
                    "cuda_full_version": version,
                    "cudnn_version": generate_binary_build_matrix.CUDA_ARCHES_CUDNN_VERSION[
                        cuda
                    ],
                    "image_type": image,
                    "platform": "linux/amd64",
                }
            )
    ret.append(
        {
            "cuda": "cpu",
            "cuda_full_version": "",
            "cudnn_version": "",
            "image_type": "runtime",
            "platform": "linux/arm64",
        }
    )

    return {"include": ret}


if __name__ == "__main__":
    build_matrix = generate_docker_matrix()
    print(json.dumps(build_matrix))
