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
]


def get_workflow_jobs():
    return [item.gen_tree() for item in WORKFLOW_DATA]
