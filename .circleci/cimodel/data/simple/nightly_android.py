from cimodel.data.simple.util.docker_constants import (
    DOCKER_IMAGE_NDK,
    DOCKER_REQUIREMENT_NDK
)


class AndroidNightlyJob:
    def __init__(self,
                 variant,
                 template_name,
                 extra_props=None,
                 with_docker=True,
                 requires=None,
                 no_build_suffix=False):

        self.variant = variant
        self.template_name = template_name
        self.extra_props = extra_props or {}
        self.with_docker = with_docker
        self.requires = requires
        self.no_build_suffix = no_build_suffix

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
        ] + self.variant

        build_suffix = [] if self.no_build_suffix else ["build"]
        full_job_name = "_".join(["nightly"] + base_name_parts + build_suffix)
        build_env_name = "-".join(base_name_parts)

        props_dict = {
            "name": full_job_name,
            "requires": self.requires,
            "filters": {"branches": {"only": "nightly"}},
        }

        props_dict.update(self.extra_props)

        if self.with_docker:
            props_dict["docker_image"] = DOCKER_IMAGE_NDK
            props_dict["build_environment"] = build_env_name

        return [{self.template_name: props_dict}]

BASE_REQUIRES = [DOCKER_REQUIREMENT_NDK]

WORKFLOW_DATA = [
    AndroidNightlyJob(["x86_32"], "pytorch_linux_build", requires=BASE_REQUIRES),
    AndroidNightlyJob(["x86_64"], "pytorch_linux_build", requires=BASE_REQUIRES),
    AndroidNightlyJob(["arm", "v7a"], "pytorch_linux_build", requires=BASE_REQUIRES),
    AndroidNightlyJob(["arm", "v8a"], "pytorch_linux_build", requires=BASE_REQUIRES),
    AndroidNightlyJob(["android_gradle"], "pytorch_android_gradle_build",
                      with_docker=False,
                      requires=[
                          "nightly_pytorch_linux_xenial_py3_clang5_android_ndk_r19c_x86_32_build",
                          "nightly_pytorch_linux_xenial_py3_clang5_android_ndk_r19c_x86_64_build",
                          "nightly_pytorch_linux_xenial_py3_clang5_android_ndk_r19c_arm_v7a_build",
                          "nightly_pytorch_linux_xenial_py3_clang5_android_ndk_r19c_arm_v8a_build"]),
    AndroidNightlyJob(["x86_32_android_publish_snapshot"], "pytorch_android_publish_snapshot",
                      extra_props={"context": "org-member"},
                      with_docker=False,
                      requires=["nightly_pytorch_linux_xenial_py3_clang5_android_ndk_r19c_android_gradle_build"],
                      no_build_suffix=True),
]


def get_workflow_jobs():
    return [item.gen_tree() for item in WORKFLOW_DATA]
