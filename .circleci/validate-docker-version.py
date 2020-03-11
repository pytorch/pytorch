#!/usr/bin/env python3
import cimodel.data.caffe2_build_definitions as caffe2_build_definitions
import cimodel.data.pytorch_build_definitions as pytorch_build_definitions
from yaml import load

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


def load_config(filename=".circleci/config.yml"):
    with open(filename, "r") as fh:
        return load("".join(fh.readlines()), Loader)


def load_tags_for_projects(workflow_config):
    return {
        v["ecr_gc_job"]["project"]: v["ecr_gc_job"]["tags_to_keep"]
        for v in workflow_config["workflows"]["ecr_gc"]["jobs"]
        if isinstance(v, dict) and "ecr_gc_job" in v
    }


def check_version(job, tags, expected_version):
    valid_versions = tags[job].split(",")
    if expected_version not in valid_versions:
        raise RuntimeError(
            "We configured {} to use Docker version {}; but this "
            "version is not configured in job ecr_gc_job_for_{}.  Non-deployed versions will be "
            "garbage collected two weeks after they are created.  DO NOT LAND "
            "THIS TO MASTER without also updating ossci-job-dsl with this version."
            "\n\nDeployed versions: {}".format(job, expected_version, job, tags[job])
        )


def validate_docker_version():
    tags = load_tags_for_projects(load_config())
    check_version("pytorch", tags, pytorch_build_definitions.DOCKER_IMAGE_VERSION)
    check_version("caffe2", tags, caffe2_build_definitions.DOCKER_IMAGE_VERSION)


if __name__ == "__main__":
    validate_docker_version()
