from collections import OrderedDict

import miniutils


DOCKER_IMAGE_PATH_BASE = "308535385114.dkr.ecr.us-east-1.amazonaws.com/pytorch/"

DEFAULT_DOCKER_VERSION = 282


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
                 cuda_version=None,
                 is_xla=False,
                 restrict_phases=None,
                 cuda_docker_phases=None,
                 gpu_resource=None):

        self.distro = distro
        self.pyver = pyver
        self.parms = parms

        self.cuda_version = cuda_version
        self.is_xla = is_xla
        self.restrict_phases = restrict_phases

        # FIXME does the build phase ever need CUDA runtime?
        self.cuda_docker_phases = cuda_docker_phases or []

        self.gpu_resource = gpu_resource

    def getParms(self):
        leading = ["pytorch"]
        if self.is_xla:
            leading.append(DockerHide("xla"))

        cuda_parms = []
        if self.cuda_version:
            cuda_parms.extend(["cuda" + self.cuda_version, "cudnn7"])
        return leading + ["linux", self.distro] + cuda_parms + self.parms

    # TODO: Eliminate this special casing in docker paths
    def genDockerImagePath(self, build_or_test):

        build_env_pieces = self.getParms()
        build_env_pieces = list(filter(lambda x: type(x) is not DockerHide, build_env_pieces))

        build_job_name_pieces = build_env_pieces + [build_or_test]

        base_build_env_name = "-".join(build_env_pieces)

        docker_version = DEFAULT_DOCKER_VERSION

        return miniutils.quote(DOCKER_IMAGE_PATH_BASE + base_build_env_name + ":" + str(docker_version))

    def getBuildJobNamePieces(self, build_or_test):
        return self.getParms() + [build_or_test]

    def genBuildName(self, build_or_test):
        return ("_".join(map(str, self.getBuildJobNamePieces(build_or_test)))).replace(".", "_")

    def genYamlTree(self, build_or_test):

        build_job_name_pieces = self.getBuildJobNamePieces(build_or_test)

        base_build_env_name = "-".join(map(str, self.getParms()))
        build_env_name = "-".join(map(str, build_job_name_pieces))

        env_dict = {
            "BUILD_ENVIRONMENT": build_env_name,
            "DOCKER_IMAGE": self.genDockerImagePath(build_or_test),
        }

        if self.pyver:
            env_dict["PYTHON_VERSION"] = miniutils.quote(self.pyver)

        if build_or_test in self.cuda_docker_phases:
            env_dict["USE_CUDA_DOCKER_RUNTIME"] = miniutils.quote("1")

        d = {
            "environment": env_dict,
            "<<": "*" + "_".join(["pytorch", "linux", build_or_test, "defaults"]),
        }

        if build_or_test == "test":
            resource_class = "large"
            if self.gpu_resource:
                resource_class = "gpu." + self.gpu_resource

                if self.gpu_resource == "large":
                    env_dict["MULTI_GPU"] = miniutils.quote("1")

            d["resource_class"] = resource_class

        return d


BUILD_ENV_LIST = [
    Conf("trusty", ["py2.7.9"]),
    Conf("trusty", ["py2.7"]),
    Conf("trusty", ["py3.5"]),
    Conf("trusty", ["py3.5"]),
    Conf("trusty", ["py3.6", "gcc4.8"]),
    Conf("trusty", ["py3.6", "gcc5.4"]),
    Conf("trusty", ["py3.6", "gcc5.4"], is_xla=True),
    Conf("trusty", ["py3.6", "gcc7"]),
    Conf("trusty", ["pynightly"]),
    Conf("xenial", ["py3", "clang5", "asan"], pyver="3.6"),
    Conf("xenial",
         ["py3"],
         pyver="3.6",
         cuda_version="8",
         gpu_resource="medium",
         cuda_docker_phases=["test"]),
    Conf("xenial",
         ["py3", DockerHide("multigpu")],
         pyver="3.6",
         cuda_version="8",
         restrict_phases=["test"],
         cuda_docker_phases=["build", "test"],
         gpu_resource="large"),
    Conf("xenial",
         ["py3", DockerHide("NO_AVX2")],
         pyver="3.6",
         cuda_version="8",
         restrict_phases=["test"],
         cuda_docker_phases=["build", "test"],
         gpu_resource="medium"),
    Conf("xenial",
         ["py3", DockerHide("NO_AVX"), DockerHide("NO_AVX2")],
         pyver="3.6",
         cuda_version="8",
         restrict_phases=["test"],
         cuda_docker_phases=["build", "test"],
         gpu_resource="medium"),
    Conf("xenial",
         ["py2"],
         pyver="2.7",
         cuda_version="9",
         cuda_docker_phases=["test"],
         gpu_resource="medium"),
    Conf("xenial",
         ["py3"],
         pyver="3.6",
         cuda_version="9",
         gpu_resource="medium",
         cuda_docker_phases=["test"]),
    Conf("xenial",
         ["py3", "gcc7"],
         pyver="3.6",
         cuda_version="9.2",
         gpu_resource="medium",
         cuda_docker_phases=["test"]),
    Conf("xenial",
         ["py3", "gcc7"],
         pyver="3.6",
         cuda_version="10",
         restrict_phases=["build"]),
]


def add_build_env_defs(jobs_dict):

    mydict = OrderedDict()
    for conf_options in BUILD_ENV_LIST:

        def append_environment_dict(build_or_test):
            d = conf_options.genYamlTree(build_or_test)
            mydict[conf_options.genBuildName(build_or_test)] = d

        phases = ["build", "test"]
        if conf_options.restrict_phases:
            phases = conf_options.restrict_phases

        for phase in phases:
            append_environment_dict(phase)

    jobs_dict["version"] = 2
    jobs_dict["jobs"] = mydict
