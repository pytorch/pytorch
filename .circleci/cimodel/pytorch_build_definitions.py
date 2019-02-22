#!/usr/bin/env python3

from collections import OrderedDict

import cimodel.miniutils as miniutils
import cimodel.dimensions as dimensions
import cimodel.conf_tree as conf_tree
from cimodel.conf_tree import ConfigNode
import cimodel.visualization as visualization


DOCKER_IMAGE_PATH_BASE = "308535385114.dkr.ecr.us-east-1.amazonaws.com/pytorch/"

DOCKER_IMAGE_VERSION = 282


class DockerHide(object):
    """
    Used for hiding name elements for construction of the Docker image path.
    Name elements that are wrapped in this object may be part of the build configuration name, but
    shall be excluded from the Docker path.
    """
    def __init__(self, val):
        self.val = val

    def __str__(self):
        return self.val


class Conf(object):
    def __init__(self,
                 distro,
                 parms,
                 pyver=None,
                 cuda_version=None,
                 is_xla=False,
                 restrict_phases=None,
                 gpu_resource=None,
                 dependent_tests=None,
                 parent_build=None):

        self.distro = distro
        self.pyver = pyver
        self.parms = parms
        self.cuda_version = cuda_version
        self.is_xla = is_xla
        self.restrict_phases = restrict_phases
        self.gpu_resource = gpu_resource
        self.dependent_tests = dependent_tests or []
        self.parent_build = parent_build

    def get_parms(self):
        leading = ["pytorch"]
        if self.is_xla:
            leading.append(DockerHide("xla"))

        cuda_parms = []
        if self.cuda_version:
            cuda_parms.extend(["cuda" + self.cuda_version, "cudnn7"])
        return leading + ["linux", self.distro] + cuda_parms + self.parms

    # TODO: Eliminate this special casing in docker paths
    def gen_docker_image_path(self):

        build_env_pieces = list(map(str, filter(lambda x: type(x) is not DockerHide, self.get_parms())))
        base_build_env_name = "-".join(build_env_pieces)

        return miniutils.quote(DOCKER_IMAGE_PATH_BASE + base_build_env_name + ":" + str(DOCKER_IMAGE_VERSION))

    def get_build_job_name_pieces(self, build_or_test):
        return self.get_parms() + [build_or_test]

    def gen_build_name(self, build_or_test):
        return ("_".join(map(str, self.get_build_job_name_pieces(build_or_test)))).replace(".", "_")

    def get_dependents(self):
        return self.dependent_tests

    def gen_yaml_tree(self, build_or_test):

        build_job_name_pieces = self.get_build_job_name_pieces(build_or_test)

        build_env_name = "-".join(map(str, build_job_name_pieces))

        env_dict = OrderedDict([
            ("BUILD_ENVIRONMENT", build_env_name),
            ("DOCKER_IMAGE", self.gen_docker_image_path()),
        ])

        if self.pyver:
            env_dict["PYTHON_VERSION"] = miniutils.quote(self.pyver)

        if build_or_test == "test" and self.cuda_version:
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

    def gen_workflow_yaml_item(self, phase):

        if self.is_xla or phase == "test":
            val = OrderedDict()
            if self.is_xla:
                val["filters"] = {"branches": {"only": ["master"]}}

            if phase == "test":
                dependency_build = self.parent_build or self
                val["requires"] = [dependency_build.gen_build_name("build")]

            return {self.gen_build_name(phase): val}
        else:
            return self.gen_build_name(phase)


# TODO This is a hack to special case some configs just for the workflow list
class HiddenConf(object):
    def __init__(self, name, parent_build=None):
        self.name = name
        self.parent_build = parent_build

    def gen_workflow_yaml_item(self, phase):

        val = OrderedDict()
        dependency_build = self.parent_build
        val["requires"] = [dependency_build.gen_build_name("build")]

        return {self.gen_build_name(phase): val}

    def gen_build_name(self, _):
        return self.name


xenial_parent_config = Conf(
    "xenial",
    ["py3"],
    pyver="3.6",
    cuda_version="8",
    gpu_resource="medium",
)


# TODO This is a short-term hack until it is converted to recursive tree traversal
xenial_dependent_configs = [
    Conf("xenial",
         ["py3", DockerHide("multigpu")],
         pyver="3.6",
         cuda_version="8",
         restrict_phases=["test"],
         gpu_resource="large",
         parent_build=xenial_parent_config,
         ),
    Conf("xenial",
         ["py3", DockerHide("NO_AVX2")],
         pyver="3.6",
         cuda_version="8",
         restrict_phases=["test"],
         gpu_resource="medium",
         parent_build=xenial_parent_config,
         ),
    Conf("xenial",
         ["py3", DockerHide("NO_AVX"), DockerHide("NO_AVX2")],
         pyver="3.6",
         cuda_version="8",
         restrict_phases=["test"],
         gpu_resource="medium",
         parent_build=xenial_parent_config,
         ),

    HiddenConf("pytorch_short_perf_test_gpu", parent_build=xenial_parent_config),
    HiddenConf("pytorch_doc_push", parent_build=xenial_parent_config),
]


xenial_parent_config.dependent_tests = xenial_dependent_configs


# TODO this hierarchy is a work in progress
CONFIG_TREE_DATA = [
    ("trusty", [
        ("py2.7.9", []),
        ("py2.7", []),
        ("py3.5", []),
        ("py3.6", [
            ("gcc4.8", []),
            ("gcc5.4", [False, True]),
            ("gcc7", []),
        ]),
        ("pynightly", []),
    ]),
    ("xenial", [
        ("clang", [
            ("X", [("py3", [])]),
        ]),
        ("cuda", [
            ("8", [("3.6", [])]),
            ("9", [
                ("2.7", []),
                ("3.6", []),
            ]),
            ("9.2", [("3.6", [])]),
            ("10", [("3.6", [])]),
        ]),
    ]),
]


def get_root():
    return TopLevelNode("Pytorch Builds", CONFIG_TREE_DATA)


def gen_tree():
    root = get_root()
    configs_list = conf_tree.dfs(root)
    return configs_list


class TopLevelNode(ConfigNode):
    def __init__(self, node_name, config_tree_data):
        super(TopLevelNode, self).__init__(None, node_name)

        self.config_tree_data = config_tree_data

    def get_children(self):
        return [DistroConfigNode(self, d, p) for (d, p) in self.config_tree_data]


class DistroConfigNode(ConfigNode):
    def __init__(self, parent, distro_name, py_tree):
        super(DistroConfigNode, self).__init__(parent, distro_name)

        self.py_tree = py_tree
        self.props["distro_name"] = distro_name

    def get_children(self):
        return [PyVerConfigNode(self, k, v) for k, v in self.py_tree]


class PyVerConfigNode(ConfigNode):
    def __init__(self, parent, pyver, compiler_tree):
        super(PyVerConfigNode, self).__init__(parent, pyver)

        self.compiler_tree = compiler_tree
        self.props["pyver"] = pyver

    def get_children(self):

        if self.find_prop("distro_name") == "trusty":
            return [CompilerConfigNode(self, v, xla_options) for (v, xla_options) in self.compiler_tree]
        else:
            return []


class CompilerConfigNode(ConfigNode):
    def __init__(self, parent, compiler_name, xla_options):
        super(CompilerConfigNode, self).__init__(parent, compiler_name)

        self.xla_options = xla_options

    def get_children(self):
        return [XlaConfigNode(self, v) for v in self.xla_options]


class XlaConfigNode(ConfigNode):
    def __init__(self, parent, xla_enabled):
        super(XlaConfigNode, self).__init__(parent, "XLA=" + str(xla_enabled))

        self.xla_enabled = xla_enabled

    def get_children(self):
        return []


BUILD_ENV_LIST = [
    Conf("trusty", ["py2.7.9"]),
    Conf("trusty", ["py2.7"]),
    Conf("trusty", ["py3.5"]),
    Conf("trusty", ["py3.6", "gcc4.8"]),
    Conf("trusty", ["py3.6", "gcc5.4"]),
    Conf("trusty", ["py3.6", "gcc5.4"], is_xla=True),
    Conf("trusty", ["py3.6", "gcc7"]),
    Conf("trusty", ["pynightly"]),
    Conf("xenial", ["py3", "clang5", "asan"], pyver="3.6"),
    xenial_parent_config,
    Conf("xenial",
         ["py2"],
         pyver="2.7",
         cuda_version="9",
         gpu_resource="medium"),
    Conf("xenial",
         ["py3"],
         pyver="3.6",
         cuda_version="9",
         gpu_resource="medium"),
    Conf("xenial",
         ["py3", "gcc7"],
         pyver="3.6",
         cuda_version="9.2",
         gpu_resource="medium"),
    Conf("xenial",
         ["py3", "gcc7"],
         pyver="3.6",
         cuda_version="10",
         restrict_phases=["build"]),  # TODO why does this not have a test?
]


def add_build_env_defs(jobs_dict):

    mydict = OrderedDict()

    def append_steps(build_list):
        for conf_options in filter(lambda x: type(x) is not HiddenConf, build_list):

            def append_environment_dict(build_or_test):
                d = conf_options.gen_yaml_tree(build_or_test)
                mydict[conf_options.gen_build_name(build_or_test)] = d

            phases = dimensions.PHASES
            if conf_options.restrict_phases:
                phases = conf_options.restrict_phases

            for phase in phases:
                append_environment_dict(phase)

            # Recurse
            dependents = conf_options.get_dependents()
            if dependents:
                append_steps(dependents)

    append_steps(BUILD_ENV_LIST)

    jobs_dict["version"] = 2
    jobs_dict["jobs"] = mydict

    graph = visualization.generate_graph(get_root())
    graph.draw("aaa-config-dimensions.png", prog="twopi")


def get_workflow_list():

    x = []
    for conf_options in BUILD_ENV_LIST:

        phases = dimensions.PHASES
        if conf_options.restrict_phases:
            phases = conf_options.restrict_phases

        for phase in phases:
            x.append(conf_options.gen_workflow_yaml_item(phase))

        # TODO convert to recursion
        for conf in conf_options.dependent_tests:
            x.append(conf.gen_workflow_yaml_item("test"))

    return x
