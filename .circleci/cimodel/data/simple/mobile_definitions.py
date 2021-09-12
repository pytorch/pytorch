"""
PyTorch Mobile PR builds (use linux host toolchain + mobile build options)
"""

import cimodel.lib.miniutils as miniutils
import cimodel.data.simple.util.branch_filters
from cimodel.data.simple.util.docker_constants import (
    DOCKER_IMAGE_ASAN,
    DOCKER_REQUIREMENT_ASAN,
    DOCKER_IMAGE_NDK,
    DOCKER_REQUIREMENT_NDK
)


class MobileJob:
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
            "mobile",
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
    MobileJob(
        DOCKER_IMAGE_ASAN,
        [DOCKER_REQUIREMENT_ASAN],
        ["build"]
    ),

    # Use LLVM-DEV toolchain in android-ndk-r19c docker image
    MobileJob(
        DOCKER_IMAGE_NDK,
        [DOCKER_REQUIREMENT_NDK],
        ["custom", "build", "dynamic"]
    ),

    MobileJob(
        DOCKER_IMAGE_NDK,
        [DOCKER_REQUIREMENT_NDK],
        ["custom", "build", "static"]
    ),

    # Use LLVM-DEV toolchain in android-ndk-r19c docker image
    # Most of this CI is already covered by "mobile-custom-build-dynamic" job
    MobileJob(
        DOCKER_IMAGE_NDK,
        [DOCKER_REQUIREMENT_NDK],
        ["code", "analysis"],
        True
    ),
]


def get_workflow_jobs():
    return [item.gen_tree() for item in WORKFLOW_DATA]
