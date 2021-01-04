AWS_DOCKER_HOST = "308535385114.dkr.ecr.us-east-1.amazonaws.com"

def gen_docker_image(container_type):
    return (
        "/".join([AWS_DOCKER_HOST, "pytorch", container_type]),
        f"docker-{container_type}",
    )

def gen_docker_image_requires(image_name):
    return [f"docker-{image_name}"]


DOCKER_IMAGE_BASIC, DOCKER_REQUIREMENT_BASE = gen_docker_image(
    "pytorch-linux-xenial-py3.6-gcc5.4"
)

DOCKER_IMAGE_CUDA_10_2, DOCKER_REQUIREMENT_CUDA_10_2 = gen_docker_image(
    "pytorch-linux-xenial-cuda10.2-cudnn7-py3-gcc7"
)

DOCKER_IMAGE_GCC7, DOCKER_REQUIREMENT_GCC7 = gen_docker_image(
    "pytorch-linux-xenial-py3.6-gcc7"
)


def gen_mobile_docker(specifier):
    container_type = "pytorch-linux-xenial-py3-clang5-" + specifier
    return gen_docker_image(container_type)


DOCKER_IMAGE_ASAN, DOCKER_REQUIREMENT_ASAN = gen_mobile_docker("asan")

DOCKER_IMAGE_NDK, DOCKER_REQUIREMENT_NDK = gen_mobile_docker("android-ndk-r19c")
