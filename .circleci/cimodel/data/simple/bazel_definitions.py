from cimodel.data.simple.util.docker_constants import (
    DOCKER_IMAGE_GCC7,
    DOCKER_REQUIREMENT_GCC7
)


def gen_job_name(phase):
    job_name_parts = [
        "pytorch",
        "bazel",
        phase,
    ]

    return "_".join(job_name_parts)


class BazelJob:
    def __init__(self, phase, extra_props=None):
        self.phase = phase
        self.extra_props = extra_props or {}

    def gen_tree(self):

        template_parts = [
            "pytorch",
            "linux",
            "bazel",
            self.phase,
        ]

        build_env_parts = [
            "pytorch",
            "linux",
            "xenial",
            "py3.6",
            "gcc7",
            "bazel",
            self.phase,
        ]

        full_job_name = gen_job_name(self.phase)
        build_env_name = "-".join(build_env_parts)

        extra_requires = (
            [gen_job_name("build")] if self.phase == "test" else
            [DOCKER_REQUIREMENT_GCC7]
        )

        props_dict = {
            "build_environment": build_env_name,
            "docker_image": DOCKER_IMAGE_GCC7,
            "name": full_job_name,
            "requires": extra_requires,
        }

        props_dict.update(self.extra_props)

        template_name = "_".join(template_parts)
        return [{template_name: props_dict}]


WORKFLOW_DATA = [
    BazelJob("build", {"resource_class": "large"}),
    BazelJob("test"),
]


def get_workflow_jobs():
    return [item.gen_tree() for item in WORKFLOW_DATA]
