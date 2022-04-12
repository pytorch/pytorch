import cimodel.data.simple.util.branch_filters as branch_filters
from cimodel.data.simple.util.docker_constants import (
    DOCKER_IMAGE_NDK, DOCKER_REQUIREMENT_NDK
)


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
                 is_pr_only=False):

        self.job_name = job_name
        self.template_name = template_name
        self.dependencies = dependencies
        self.is_master_only = is_master_only
        self.is_pr_only = is_pr_only

    def gen_tree(self):

        props_dict = {
            "name": self.job_name,
            "requires": self.dependencies,
        }

        if self.is_master_only:
            props_dict["filters"] = branch_filters.gen_filter_dict(branch_filters.NON_PR_BRANCH_LIST)
        elif self.is_pr_only:
            props_dict["filters"] = branch_filters.gen_filter_dict(branch_filters.PR_BRANCH_LIST)

        return [{self.template_name: props_dict}]


WORKFLOW_DATA = [
]


def get_workflow_jobs():
    return [item.gen_tree() for item in WORKFLOW_DATA]
