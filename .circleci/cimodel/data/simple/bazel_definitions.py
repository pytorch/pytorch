from cimodel.data.simple.util import docker_constants


def gen_job_name(phase):
    job_name_parts = [
        "pytorch",
        "bazel",
        phase,
    ]

    return "_".join(job_name_parts)


class BazelJob:
    DOCKER_IMAGE = docker_constants.DOCKER_IMAGE_GCC7

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
        extra_requires = []
        if self.phase == "test":
            extra_requires.append(gen_job_name("build"))
        else:
            # Append our docker image dependency
            extra_requires.append(
                docker_constants.gen_docker_image_dependency(
                    self.DOCKER_IMAGE
                )
            )

        props_dict = {
            "build_environment": build_env_name,
            "docker_image": self.DOCKER_IMAGE,
            "name": full_job_name,
            "requires": ["setup"] + extra_requires,
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
