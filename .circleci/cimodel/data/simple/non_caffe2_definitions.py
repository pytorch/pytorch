"""
PyTorch builds without CAFFE2 options
"""

import cimodel.lib.miniutils as miniutils
import cimodel.data.simple.util.branch_filters
from cimodel.data.simple.util.docker_constants import (
    DOCKER_IMAGE_BASIC,
    DOCKER_REQUIREMENT_BASE,
)


class NonCaffe2Job:
    def __init__(
            self,
            docker_image,
            docker_requires,
            variant_parts,
            is_master_only=False):
        self.docker_image = docker_image
        self.docker_requires = docker_requires
        self.variant_parts = variant_parts
        self.is_master_only = is_master_only

    def gen_tree(self):
        non_phase_parts = [
            "pytorch",
            "linux",
            "xenial",
            "py3",
            "clang5",
            "non-caffe2",
        ] + self.variant_parts

        full_job_name = "_".join(non_phase_parts)
        build_env_name = "-".join(non_phase_parts)

        props_dict = {
            "build_environment": build_env_name,
            "build_only": miniutils.quote(str(int(True))),
            "docker_image": self.docker_image,
            "requires": self.docker_requires,
            "name": full_job_name,
        }

        if self.is_master_only:
            props_dict["filters"] = cimodel.data.simple.util.branch_filters.gen_filter_dict()

        return [{"pytorch_linux_build": props_dict}]


WORKFLOW_DATA = [
    NonCaffe2Job(
        DOCKER_IMAGE_BASIC,
        [DOCKER_REQUIREMENT_BASE],
        ["build"]
    ),
]


def get_workflow_jobs():
    return [item.gen_tree() for item in WORKFLOW_DATA]
