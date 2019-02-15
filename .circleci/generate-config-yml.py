#!/usr/bin/env python3

"""
This script is the source of truth for config.yml.
Please make changes here only, then re-run this
script and commit the result.
"""

import os
import sys
from collections import OrderedDict


DOCKER_IMAGE_PATH_BASE = "308535385114.dkr.ecr.us-east-1.amazonaws.com/pytorch/"


class DockerHide:
    """Hides element for construction of docker path"""
    def __init__(self, val):
        self.val = val

    def __str__(self):
        return self.val


class Conf:
    def __init__(self,
                 distro,
                 parms,
                 pyver=None,
                 use_cuda=False,
                 is_xla=False,
                 restrict_phases=None,
                 cuda_docker_phases=None,
                 gpu_resource=None,
                 docker_version_override=None):

        self.distro = distro
        self.pyver = pyver
        self.parms = parms

        self.use_cuda = use_cuda
        self.is_xla = is_xla
        self.restrict_phases = restrict_phases

        # FIXME does the build phase ever need CUDA runtime?
        self.cuda_docker_phases = cuda_docker_phases or []

        self.gpu_resource = gpu_resource

        # FIXME is this different docker version intentional?
        self.docker_version_override = docker_version_override

    def getParms(self):
        leading = ["pytorch"]
        if self.is_xla:
            leading.append(DockerHide("xla"))
        return leading + ["linux", self.distro] + self.parms

    # TODO: Eliminate this special casing in docker paths
    def genDockerImagePath(self, build_or_test):

        build_env_pieces = self.getParms()
        build_env_pieces = list(filter(lambda x: type(x) is not DockerHide, build_env_pieces))

        build_job_name_pieces = build_env_pieces + [build_or_test]

        base_build_env_name = "-".join(build_env_pieces)

        docker_version = 282
        if self.docker_version_override is not None:
            docker_version = self.docker_version_override

        return DOCKER_IMAGE_PATH_BASE + base_build_env_name + ":" + str(docker_version)


BUILD_ENV_LIST = [
    Conf("trusty", ["py2.7.9"]),
    Conf("trusty", ["py2.7"]),
    Conf("trusty", ["py3.5"]),
    Conf("trusty", ["py3.5"]),
    Conf("trusty", ["py3.6", "gcc4.8"]),
    Conf("trusty", ["py3.6", "gcc5.4"]),
    Conf("trusty", ["py3.6", "gcc5.4"], is_xla=True, docker_version_override=278),
    Conf("trusty", ["py3.6", "gcc7"]),
    Conf("trusty", ["pynightly"]),
    Conf("xenial", ["py3", "clang5", "asan"], pyver="3.6"),
    Conf("xenial", ["cuda8", "cudnn7", "py3"], pyver="3.6", use_cuda=True, gpu_resource="medium",
         cuda_docker_phases=["test"]),
    Conf("xenial", ["cuda8", "cudnn7", "py3", DockerHide("multigpu")], pyver="3.6", use_cuda=True,
         restrict_phases=["test"], cuda_docker_phases=["build", "test"], gpu_resource="large"),
    Conf("xenial", ["cuda8", "cudnn7", "py3", DockerHide("NO_AVX2")], pyver="3.6", use_cuda=True,
         restrict_phases=["test"], cuda_docker_phases=["build", "test"], gpu_resource="medium"),
    Conf("xenial", ["cuda8", "cudnn7", "py3", DockerHide("NO_AVX"), DockerHide("NO_AVX2")], pyver="3.6", use_cuda=True,
         restrict_phases=["test"], cuda_docker_phases=["build", "test"], gpu_resource="medium"),
    Conf("xenial", ["cuda9", "cudnn7", "py2"], pyver="2.7", use_cuda=True, cuda_docker_phases=["test"],
         gpu_resource="medium"),
    Conf("xenial", ["cuda9", "cudnn7", "py3"], pyver="3.6", use_cuda=True, gpu_resource="medium",
         cuda_docker_phases=["test"]),
    Conf("xenial", ["cuda9.2", "cudnn7", "py3", "gcc7"], pyver="3.6", use_cuda=True, gpu_resource="medium",
         cuda_docker_phases=["test"]),
    Conf("xenial", ["cuda10", "cudnn7", "py3", "gcc7"], pyver="3.6", use_cuda=True, restrict_phases=["build"]),
]


def is_dict_like(data):
    return type(data) is dict or type(data) is OrderedDict


FORCED_QUOTED_VALUE_KEYS = set(
    "DOCKER_IMAGE",
    "PYTHON_VERSION",
    "USE_CUDA_DOCKER_RUNTIME",
    "MULTI_GPU",
)


def render_yaml(key, data, fh, depth=0):
    """
    PyYaml does not allow precise control over the quoting
    behavior, especially for merge references.
    Therefore, we use this custom YAML renderer.
    """

    indentation = "  " * depth

    if is_dict_like(data):

        tuples = list(data.items())
        if type(data) is not OrderedDict:
            tuples.sort(key=lambda x: (x[0] == "<<", x[0]))

        for k, v in tuples:
            fh.write(indentation + k + ":")
            whitespace = "\n" if is_dict_like(v) else " "
            fh.write(whitespace)

            render_yaml(k, v, fh, depth + 1)

        if depth == 2:
            fh.write("\n")

    else:
        if type(data) is str:
            maybe_quoted = data
            if key in FORCED_QUOTED_VALUE_KEYS:
                maybe_quoted = '"' + data + '"'
            fh.write(maybe_quoted)

        else:
            fh.write(str(data))

        fh.write("\n")


def generate_config_dict():

    jobs_dict = OrderedDict()
    for conf_options in BUILD_ENV_LIST:

        build_env_pieces = conf_options.getParms()

        def append_environment_dict(build_or_test):

            build_job_name_pieces = build_env_pieces + [build_or_test]

            base_build_env_name = "-".join(map(str, build_env_pieces))
            build_env_name = "-".join(map(str, build_job_name_pieces))

            env_dict = {
                "BUILD_ENVIRONMENT": build_env_name,
                "DOCKER_IMAGE": conf_options.genDockerImagePath(build_or_test),
            }

            if conf_options.pyver:
                env_dict["PYTHON_VERSION"] = conf_options.pyver

            if build_or_test in conf_options.cuda_docker_phases:
                env_dict["USE_CUDA_DOCKER_RUNTIME"] = "1"

            d = {
                "environment": env_dict,
                "<<": "*" + "_".join(["pytorch", "linux", build_or_test, "defaults"]),
            }

            if build_or_test == "test":
                resource_class = "large"
                if conf_options.gpu_resource:
                    resource_class = "gpu." + conf_options.gpu_resource

                    if conf_options.gpu_resource == "large":
                        env_dict["MULTI_GPU"] = "1"

                d["resource_class"] = resource_class

            job_name = ("_".join(map(str, build_job_name_pieces))).replace(".", "_")
            jobs_dict[job_name] = d

        phases = ["build", "test"]
        if conf_options.restrict_phases:
            phases = conf_options.restrict_phases

        for phase in phases:
            append_environment_dict(phase)

    data = OrderedDict([
        ("version", 2),
        ("jobs", jobs_dict),
    ])

    return data


VERBATIM_SOURCE_FILES = [
    "header-section.yml",
    "linux-build-defaults.yml",
    "macos-build-defaults.yml",
    "nightly-binary-build-defaults.yml",
    "linux-binary-build-defaults.yml",
    "macos-binary-build-defaults.yml",
    "nightly-build-smoke-tests-defaults.yml",
]


YAML_GENERATOR_FUNCTIONS = [
    generate_config_dict,
]


def comment_divider(output_filehandle):
    for _i in range(2):
        output_filehandle.write("#" * 78)
        output_filehandle.write("\n")


def stitch_sources(output_filehandle):

    for f in VERBATIM_SOURCE_FILES:
        with open(os.path.join("verbatim-sources", f)) as fh:
            output_filehandle.write(fh.read())

    comment_divider(output_filehandle)
    output_filehandle.write("# Job specifications job specs\n")
    comment_divider(output_filehandle)

    for f in YAML_GENERATOR_FUNCTIONS:
        render_yaml(None, f(), output_filehandle)

    with open("verbatim-sources/remaining-sections.yml") as fh:
        output_filehandle.write(fh.read())


if __name__ == "__main__":

    stitch_sources(sys.stdout)
