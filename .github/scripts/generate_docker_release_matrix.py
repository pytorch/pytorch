
import json
import generate_binary_build_matrix
from typing import Dict, List

DOCKER_IMAGE_TYPES = ["runtime", "devel"]

def generate_docker_matrix() -> List[Dict[str, str]]:
    ret: List[Dict[str, str]] = []
    for cuda in generate_binary_build_matrix.CUDA_ARCHES:
        for image in DOCKER_IMAGE_TYPES:
            platform = "linux/arm64,linux/amd64" if image == "runtime" else "linux/amd6"
            ret.append(
                {
                    "cuda": cuda,
                    "image_type": image,
                    "platform": platform,
                }
            )
    return ret


if __name__ == "__main__":
    build_matrix = generate_docker_matrix()
    print(json.dumps(build_matrix))
