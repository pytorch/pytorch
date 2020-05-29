AWS_DOCKER_HOST = "308535385114.dkr.ecr.us-east-1.amazonaws.com"

# ARE YOU EDITING THIS NUMBER?  MAKE SURE YOU READ THE GUIDANCE AT THE
# TOP OF .circleci/config.yml
DOCKER_IMAGE_TAG = "209062ef-ab58-422a-b295-36c4eed6e906"


def gen_docker_image_path(container_type):
    return "/".join([
        AWS_DOCKER_HOST,
        "pytorch",
        container_type + ":" + DOCKER_IMAGE_TAG,
    ])


DOCKER_IMAGE_BASIC = gen_docker_image_path("pytorch-linux-xenial-py3.6-gcc5.4")

DOCKER_IMAGE_CUDA_10_2 = gen_docker_image_path("pytorch-linux-xenial-cuda10.2-cudnn7-py3-gcc7")

DOCKER_IMAGE_GCC7 = gen_docker_image_path("pytorch-linux-xenial-py3.6-gcc7")


def gen_mobile_docker_name(specifier):
    container_type = "pytorch-linux-xenial-py3-clang5-" + specifier
    return gen_docker_image_path(container_type)


DOCKER_IMAGE_ASAN = gen_mobile_docker_name("asan")

DOCKER_IMAGE_NDK = gen_mobile_docker_name("android-ndk-r19c")
