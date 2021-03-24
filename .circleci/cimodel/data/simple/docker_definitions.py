from collections import OrderedDict

from cimodel.lib.miniutils import quote
from cimodel.data.simple.util.branch_filters import gen_filter_dict, RC_PATTERN


# TODO: make this generated from a matrix rather than just a static list
IMAGE_NAMES = [
    "pytorch-linux-bionic-cuda10.2-cudnn7-py3.8-gcc9",
    "pytorch-linux-bionic-py3.6-clang9",
    "pytorch-linux-bionic-cuda10.2-cudnn7-py3.6-clang9",
    "pytorch-linux-bionic-py3.8-gcc9",
    "pytorch-linux-xenial-cuda10-cudnn7-py3-gcc7",
    "pytorch-linux-xenial-cuda10.1-cudnn7-py3-gcc7",
    "pytorch-linux-xenial-cuda10.2-cudnn7-py3-gcc7",
    "pytorch-linux-xenial-cuda11.1-cudnn8-py3-gcc7",
    "pytorch-linux-xenial-cuda11.2-cudnn8-py3-gcc7",
    "pytorch-linux-xenial-py3-clang5-android-ndk-r19c",
    "pytorch-linux-xenial-py3-clang5-asan",
    "pytorch-linux-xenial-py3-clang7-onnx",
    "pytorch-linux-xenial-py3.8",
    "pytorch-linux-xenial-py3.6-clang7",
    "pytorch-linux-xenial-py3.6-gcc5.4",  # this one is used in doc builds
    "pytorch-linux-xenial-py3.6-gcc7.2",
    "pytorch-linux-xenial-py3.6-gcc7",
    "pytorch-linux-bionic-rocm3.9-py3.6",
    "pytorch-linux-bionic-rocm3.10-py3.6",
    "pytorch-linux-bionic-rocm4.0.1-py3.6",
    "pytorch-linux-bionic-rocm4.1-py3.6",
]


def get_workflow_jobs():
    """Generates a list of docker image build definitions"""
    ret = []
    for image_name in IMAGE_NAMES:
        parameters = OrderedDict({
            "name": quote(f"docker-{image_name}"),
            "image_name": quote(image_name),
        })
        if image_name == "pytorch-linux-xenial-py3.6-gcc5.4":
            # pushing documentation on tags requires CircleCI to also
            # build all the dependencies on tags, including this docker image
            parameters['filters'] = gen_filter_dict(branches_list=r"/.*/",
                                                    tags_list=RC_PATTERN)
        ret.append(OrderedDict(
            {
                "docker_build_job": parameters
            }
        ))
    return ret
