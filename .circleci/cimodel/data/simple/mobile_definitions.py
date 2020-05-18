"""
PyTorch Mobile PR builds (use linux host toolchain + mobile build options)
"""

import cimodel.lib.miniutils as miniutils
import cimodel.data.simple.branch_filters


def gen_docker_name(specifier):
    parts = [
        "308535385114.dkr.ecr.us-east-1.amazonaws.com/pytorch/pytorch-linux-xenial-py3-clang5-",
        specifier,
        ":9a3986fa-7ce7-4a36-a001-3c9bef9892e2",
    ]
    return "".join(parts)


DOCKER_IMAGE_ASAN = gen_docker_name("asan")
DOCKER_IMAGE_NDK = gen_docker_name("android-ndk-r19c")


class MobileJob:
    def __init__(self, docker_image, variant_parts, is_master_only=False):
        self.docker_image = docker_image
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
            "name": full_job_name,
            "requires": ["setup"],
        }

        if self.is_master_only:
            props_dict["filters"] = cimodel.data.simple.branch_filters.gen_branch_filter_dict()

        return [{"pytorch_linux_build": props_dict}]


WORKFLOW_DATA = [
    MobileJob(DOCKER_IMAGE_ASAN, ["build"]),
    MobileJob(DOCKER_IMAGE_ASAN, ["custom", "build", "static"]),

    # Use LLVM-DEV toolchain in android-ndk-r19c docker image
    MobileJob(DOCKER_IMAGE_NDK, ["custom", "build", "dynamic"]),

    # Use LLVM-DEV toolchain in android-ndk-r19c docker image
    # Most of this CI is already covered by "mobile-custom-build-dynamic" job
    MobileJob(DOCKER_IMAGE_NDK, ["code", "analysis"], True),
]


def get_workflow_jobs():
    return [item.gen_tree() for item in WORKFLOW_DATA]
