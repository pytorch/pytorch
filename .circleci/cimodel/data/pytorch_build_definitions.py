#!/usr/bin/env python3

from collections import OrderedDict

from cimodel.data.pytorch_build_data import TopLevelNode, CONFIG_TREE_DATA
import cimodel.data.dimensions as dimensions
import cimodel.lib.conf_tree as conf_tree
import cimodel.lib.miniutils as miniutils
import cimodel.lib.visualization as visualization

from dataclasses import dataclass, field
from typing import List, Optional


DOCKER_IMAGE_PATH_BASE = "308535385114.dkr.ecr.us-east-1.amazonaws.com/pytorch/"

DOCKER_IMAGE_VERSION = 300


@dataclass
class Conf:
    distro: str
    parms: List[str]
    pyver: Optional[str] = None
    cuda_version: Optional[str] = None
    # TODO expand this to cover all the USE_* that we want to test for
    #  tesnrorrt, leveldb, lmdb, redis, opencv, mkldnn, ideep, etc.
    # (from https://github.com/pytorch/pytorch/pull/17323#discussion_r259453608)
    is_xla: bool = False
    restrict_phases: Optional[List[str]] = None
    gpu_resource: Optional[str] = None
    dependent_tests: List = field(default_factory=list)
    parent_build: Optional['Conf'] = None
    is_namedtensor: bool = False
    is_important: bool = False

    # TODO: Eliminate the special casing for docker paths
    # In the short term, we *will* need to support special casing as docker images are merged for caffe2 and pytorch
    def get_parms(self, for_docker):
        leading = []
        # We just don't run non-important jobs on pull requests;
        # previously we also named them in a way to make it obvious
        # if self.is_important and not for_docker:
        #    leading.append("AAA")
        leading.append("pytorch")
        if self.is_xla and not for_docker:
            leading.append("xla")
        if self.is_namedtensor and not for_docker:
            leading.append("namedtensor")

        cuda_parms = []
        if self.cuda_version:
            cuda_parms.extend(["cuda" + self.cuda_version, "cudnn7"])
        return leading + ["linux", self.distro] + cuda_parms + self.parms

    def gen_docker_image_path(self):

        parms_source = self.parent_build or self
        base_build_env_name = "-".join(parms_source.get_parms(True))

        return miniutils.quote(DOCKER_IMAGE_PATH_BASE + base_build_env_name + ":" + str(DOCKER_IMAGE_VERSION))

    def get_build_job_name_pieces(self, build_or_test):
        return self.get_parms(False) + [build_or_test]

    def gen_build_name(self, build_or_test):
        return ("_".join(map(str, self.get_build_job_name_pieces(build_or_test)))).replace(".", "_").replace("-", "_")

    def get_dependents(self):
        return self.dependent_tests or []

    def gen_yaml_tree(self, build_or_test):

        build_job_name_pieces = self.get_build_job_name_pieces(build_or_test)

        build_env_name = "-".join(map(str, build_job_name_pieces))

        env_dict = OrderedDict([
            ("BUILD_ENVIRONMENT", build_env_name),
            ("DOCKER_IMAGE", self.gen_docker_image_path()),
        ])

        if self.pyver:
            env_dict["PYTHON_VERSION"] = miniutils.quote(self.pyver)

        if build_or_test == "test" and self.gpu_resource:
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

        # All jobs require the setup job
        parameters = OrderedDict({"requires": ["setup"]})

        if phase == "test":

            # TODO When merging the caffe2 and pytorch jobs, it might be convenient for a while to make a
            #  caffe2 test job dependent on a pytorch build job. This way we could quickly dedup the repeated
            #  build of pytorch in the caffe2 build job, and just run the caffe2 tests off of a completed
            #  pytorch build job (from https://github.com/pytorch/pytorch/pull/17323#discussion_r259452641)

            dependency_build = self.parent_build or self
            parameters["requires"].append(dependency_build.gen_build_name("build"))

        if not self.is_important:
            # If you update this, update
            # caffe2_build_definitions.py too
            parameters["filters"] = {"branches": {"only": ["master", r"/ci-all\/.*/"]}}

        return {self.gen_build_name(phase): parameters}


# TODO This is a hack to special case some configs just for the workflow list
class HiddenConf(object):
    def __init__(self, name, parent_build=None):
        self.name = name
        self.parent_build = parent_build

    def gen_workflow_yaml_item(self, phase):

        return {self.gen_build_name(phase): {"requires": [self.parent_build.gen_build_name("build")]}}

    def gen_build_name(self, _):
        return self.name


# TODO Convert these to graph nodes
def gen_dependent_configs(xenial_parent_config):

    extra_parms = [
        (["multigpu"], "large"),
        (["NO_AVX2"], "medium"),
        (["NO_AVX", "NO_AVX2"], "medium"),
        (["slow"], "medium"),
        (["nogpu"], None),
    ]

    configs = []
    for parms, gpu in extra_parms:

        c = Conf(
            xenial_parent_config.distro,
            ["py3"] + parms,
            pyver="3.6",
            cuda_version=xenial_parent_config.cuda_version,
            restrict_phases=["test"],
            gpu_resource=gpu,
            parent_build=xenial_parent_config,
            is_important=xenial_parent_config.is_important,
        )

        configs.append(c)

    for x in ["pytorch_short_perf_test_gpu", "pytorch_doc_push"]:
        configs.append(HiddenConf(x, parent_build=xenial_parent_config))

    return configs


def get_root():
    return TopLevelNode("PyTorch Builds", CONFIG_TREE_DATA)


def gen_tree():
    root = get_root()
    configs_list = conf_tree.dfs(root)
    return configs_list


def instantiate_configs():

    config_list = []

    root = get_root()
    found_configs = conf_tree.dfs(root)
    restrict_phases = None
    for fc in found_configs:

        distro_name = fc.find_prop("distro_name")

        python_version = None
        if distro_name == "xenial":
            python_version = fc.find_prop("pyver")
            parms_list = [fc.find_prop("abbreviated_pyver")]
        else:
            parms_list = ["py" + fc.find_prop("pyver")]

        compiler_name = fc.find_prop("compiler_name")

        cuda_version = None
        if compiler_name == "cuda":
            cuda_version = fc.find_prop("compiler_version")

        elif compiler_name == "android":
            android_ndk_version = fc.find_prop("compiler_version")
            # TODO: do we need clang to compile host binaries like protoc?
            parms_list.append("clang5")
            parms_list.append("android-ndk-" + android_ndk_version)
            restrict_phases = ["build"]

        elif compiler_name:
            gcc_version = compiler_name + (fc.find_prop("compiler_version") or "")
            parms_list.append(gcc_version)

            # TODO: This is a nasty special case
            if compiler_name == "clang":
                parms_list.append("asan")

        if cuda_version in ["9.2", "10"]:
            # TODO The gcc version is orthogonal to CUDA version?
            parms_list.append("gcc7")

        is_xla = fc.find_prop("is_xla") or False
        is_namedtensor = fc.find_prop("is_namedtensor") or False
        is_important = fc.find_prop("is_important") or False

        gpu_resource = None
        if cuda_version and cuda_version != "10":
            gpu_resource = "medium"

        c = Conf(
            distro_name,
            parms_list,
            python_version,
            cuda_version,
            is_xla,
            restrict_phases,
            gpu_resource,
            is_namedtensor=is_namedtensor,
            is_important=is_important,
        )

        if cuda_version == "9" and python_version == "3.6":
            c.dependent_tests = gen_dependent_configs(c)

        config_list.append(c)

    return config_list


def add_build_env_defs(jobs_dict):

    mydict = OrderedDict()

    config_list = instantiate_configs()
    for c in config_list:

        phases = c.restrict_phases or dimensions.PHASES

        for phase in phases:

            # TODO why does this not have a test?
            if phase == "test" and c.cuda_version == "10":
                continue

            d = c.gen_yaml_tree(phase)
            mydict[c.gen_build_name(phase)] = d

            if phase == "test":
                for x in filter(lambda x: type(x) is not HiddenConf, c.get_dependents()):

                    d = x.gen_yaml_tree(phase)
                    mydict[x.gen_build_name(phase)] = d

    # this is the circleci api version and probably never changes
    jobs_dict["version"] = 2
    jobs_dict["jobs"] = mydict

    graph = visualization.generate_graph(get_root())
    graph.draw("pytorch-config-dimensions.png", prog="twopi")


def get_workflow_list():

    config_list = instantiate_configs()

    x = ["setup"]
    for conf_options in config_list:

        phases = conf_options.restrict_phases or dimensions.PHASES

        for phase in phases:

            # TODO why does this not have a test?
            if phase == "test" and conf_options.cuda_version == "10":
                continue

            x.append(conf_options.gen_workflow_yaml_item(phase))

        # TODO convert to recursion
        for conf in conf_options.get_dependents():
            x.append(conf.gen_workflow_yaml_item("test"))

    return x
