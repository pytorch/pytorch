import cimodel.data.simple.util.branch_filters as branch_filters
from cimodel.data.simple.util.docker_constants import (
    DOCKER_IMAGE_NDK, DOCKER_REQUIREMENT_NDK
)
import cimodel.lib.miniutils as miniutils


class AndroidJob:
    def __init__(self,
                 variant,
                 template_name,
                 is_master_only=True):

        self.variant = variant
        self.template_name = template_name
        self.is_master_only = is_master_only

    def gen_tree(self):

        base_name_parts = [
            "pytorch",
            "linux",
            "xenial",
            "py3",
            "clang5",
            "android",
            "ndk",
            "r19c",
        ] + self.variant + [
            "build",
        ]

        full_job_name = "_".join(base_name_parts)
        build_env_name = "-".join(base_name_parts)

        props_dict = {
            "name": full_job_name,
            "build_environment": "\"{}\"".format(build_env_name),
            "docker_image": "\"{}\"".format(DOCKER_IMAGE_NDK),
            "requires": [DOCKER_REQUIREMENT_NDK]
        }

        if self.is_master_only:
            props_dict["filters"] = branch_filters.gen_filter_dict(branch_filters.NON_PR_BRANCH_LIST)

        return [{self.template_name: props_dict}]


class AndroidGradleJob:
    def __init__(self,
                 job_name,
                 template_name,
                 dependencies,
                 is_master_only=True,
                 is_pr_only=False,
                 extra_props=tuple()):

        self.job_name = job_name
        self.template_name = template_name
        self.dependencies = dependencies
        self.is_master_only = is_master_only
        self.is_pr_only = is_pr_only
        self.extra_props = dict(extra_props)

    def gen_tree(self):

        props_dict = {
            "name": self.job_name,
            "requires": self.dependencies,
        }

        if self.is_master_only:
            props_dict["filters"] = branch_filters.gen_filter_dict(branch_filters.NON_PR_BRANCH_LIST)
        elif self.is_pr_only:
            props_dict["filters"] = branch_filters.gen_filter_dict(branch_filters.PR_BRANCH_LIST)
        if self.extra_props:
            props_dict.update(self.extra_props)

        return [{self.template_name: props_dict}]


WORKFLOW_DATA = [
    AndroidJob(["x86_32"], "pytorch_linux_build", is_master_only=False),
    AndroidJob(["x86_64"], "pytorch_linux_build"),
    AndroidJob(["arm", "v7a"], "pytorch_linux_build"),
    AndroidJob(["arm", "v8a"], "pytorch_linux_build"),
    AndroidGradleJob(
        "pytorch-linux-xenial-py3-clang5-android-ndk-r19c-gradle-build-x86_32",
        "pytorch_android_gradle_build-x86_32",
        ["pytorch_linux_xenial_py3_clang5_android_ndk_r19c_x86_32_build"],
        is_master_only=False,
        is_pr_only=True),
    AndroidGradleJob(
        "pytorch-linux-xenial-py3-clang5-android-ndk-r19c-gradle-custom-build-single",
        "pytorch_android_gradle_custom_build_single",
        [DOCKER_REQUIREMENT_NDK],
        is_master_only=False,
        is_pr_only=True),
    AndroidGradleJob(
        "pytorch-linux-xenial-py3-clang5-android-ndk-r19c-gradle-custom-build-single-full-jit",
        "pytorch_android_gradle_custom_build_single",
        [DOCKER_REQUIREMENT_NDK],
        is_master_only=False,
        is_pr_only=True,
        extra_props=tuple({
            "lite_interpreter": miniutils.quote(str(int(False)))
        }.items())),
    AndroidGradleJob(
        "pytorch-linux-xenial-py3-clang5-android-ndk-r19c-gradle-build",
        "pytorch_android_gradle_build",
        ["pytorch_linux_xenial_py3_clang5_android_ndk_r19c_x86_32_build",
         "pytorch_linux_xenial_py3_clang5_android_ndk_r19c_x86_64_build",
         "pytorch_linux_xenial_py3_clang5_android_ndk_r19c_arm_v7a_build",
         "pytorch_linux_xenial_py3_clang5_android_ndk_r19c_arm_v8a_build"]),
]


def get_workflow_jobs():
    return [item.gen_tree() for item in WORKFLOW_DATA]
