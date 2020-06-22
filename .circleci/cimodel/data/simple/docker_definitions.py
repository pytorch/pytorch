from collections import OrderedDict

from cimodel.lib.miniutils import quote


# TODO: make this generated from a matrix rather than just a static list
IMAGE_NAMES = [
    "pytorch-linux-bionic-cuda10.2-cudnn7-py3.8-gcc9",
    "pytorch-linux-bionic-py3.6-clang9",
    "pytorch-linux-bionic-cuda10.2-cudnn7-py3.6-clang9",
    "pytorch-linux-bionic-py3.8-gcc9",
    "pytorch-linux-xenial-cuda10-cudnn7-py3-gcc7",
    "pytorch-linux-xenial-cuda10.1-cudnn7-py3-gcc7",
    "pytorch-linux-xenial-cuda10.2-cudnn7-py3-gcc7",
    "pytorch-linux-xenial-cuda9.2-cudnn7-py3-gcc5.4",
    "pytorch-linux-xenial-cuda9.2-cudnn7-py3-gcc7",
    "pytorch-linux-xenial-py3-clang5-android-ndk-r19c",
    "pytorch-linux-xenial-py3-clang5-asan",
    "pytorch-linux-xenial-py3.8",
    "pytorch-linux-xenial-py3.6-clang7",
    "pytorch-linux-xenial-py3.6-gcc4.8",
    "pytorch-linux-xenial-py3.6-gcc5.4",
    "pytorch-linux-xenial-py3.6-gcc7.2",
    "pytorch-linux-xenial-py3.6-gcc7",
    "pytorch-linux-xenial-pynightly",
    "pytorch-linux-xenial-rocm3.3-py3.6",
]


def get_workflow_jobs():
    """Generates a list of docker image build definitions"""
    return [
        OrderedDict(
            {
                "docker_build_job": OrderedDict(
                    {"name": quote(image_name), "image_name": quote(image_name)}
                )
            }
        )
        for image_name in IMAGE_NAMES
    ]
