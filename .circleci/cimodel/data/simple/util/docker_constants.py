AWS_DOCKER_HOST = "308535385114.dkr.ecr.us-east-1.amazonaws.com"

DOCKER_IMAGE_TAG = ":9a3986fa-7ce7-4a36-a001-3c9bef9892e2"
DOCKER_IMAGE_BASIC = "/".join([
    AWS_DOCKER_HOST,
    "pytorch",
    "pytorch-linux-xenial-py3.6-gcc5.4" + DOCKER_IMAGE_TAG,
])

DOCKER_IMAGE_CUDA_10_2 = "/".join([
    AWS_DOCKER_HOST,
    "pytorch",
    "pytorch-linux-xenial-cuda10.2-cudnn7-py3-gcc7" + DOCKER_IMAGE_TAG,
])

DOCKER_IMAGE_GCC7 = "/".join([
    AWS_DOCKER_HOST,
    "pytorch",
    "pytorch-linux-xenial-py3.6-gcc7" + DOCKER_IMAGE_TAG,
])


def gen_mobile_docker_name(specifier):

    final_path_part = "".join([
        "pytorch-linux-xenial-py3-clang5-",
        specifier,
        DOCKER_IMAGE_TAG,
    ])

    parts = [
        AWS_DOCKER_HOST,
        "pytorch",
        final_path_part,
    ]

    return "/".join(parts)


DOCKER_IMAGE_ASAN = gen_mobile_docker_name("asan")

DOCKER_IMAGE_NDK = gen_mobile_docker_name("android-ndk-r19c")
