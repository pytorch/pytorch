from collections import OrderedDict

from cimodel.lib.miniutils import quote
from cimodel.data.simple.util.branch_filters import gen_filter_dict, RC_PATTERN


# TODO: make this generated from a matrix rather than just a static list
IMAGE_NAMES = [
    "pytorch-linux-bionic-rocm4.1-py3.6",
    "pytorch-linux-bionic-rocm4.2-py3.6",
    "pytorch-linux-bionic-rocm4.3.1-py3.6",
]

# This entry should be an element from the list above
# This should contain the image matching the "slow_gradcheck" entry in
# pytorch_build_data.py
SLOW_GRADCHECK_IMAGE_NAME = "pytorch-linux-xenial-cuda10.2-cudnn7-py3-gcc7"

def get_workflow_jobs(images=IMAGE_NAMES, only_slow_gradcheck=False):
    """Generates a list of docker image build definitions"""
    ret = []
    for image_name in images:
        if only_slow_gradcheck and image_name is not SLOW_GRADCHECK_IMAGE_NAME:
            continue

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
