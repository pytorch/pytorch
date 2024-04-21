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
                    "platform": "linux/arm64,linux/amd64",
                }
            )
    return {"include": ret}


if __name__ == "__main__":
    build_matrix = generate_docker_matrix()
    print(json.dumps(build_matrix))
