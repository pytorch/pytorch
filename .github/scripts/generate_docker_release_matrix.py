
import json
import generate_binary_build_matrix
from typing import Dict, List

DOCKER_IMAGE_TYPES = ["runtime", "devel"]

def generate_docker_matrix() -> List[Dict[str, str]]:
    ret: List[Dict[str, str]] = []
    for cuda, version in generate_binary_build_matrix.CUDA_ARCHES_FULL_VERSION:
        for image in DOCKER_IMAGE_TYPES:
            ret.append(
                {
                    "cuda": cuda,
                    "cuda_full_version": version,
                    "image_type": image,
                    "platform": "linux/arm64,linux/amd64",
                }
            )
    return {"include": ret}


if __name__ == "__main__":
    build_matrix = generate_docker_matrix()
    print(json.dumps(build_matrix))
