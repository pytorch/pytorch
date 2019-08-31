#!/usr/bin/env python3

from collections import OrderedDict

from cimodel.data.pytorch_build_data import TopLevelNode, CONFIG_TREE_DATA
import cimodel.data.dimensions as dimensions
import cimodel.lib.conf_tree as conf_tree
import cimodel.lib.miniutils as miniutils

from dataclasses import dataclass, field, asdict
from typing import List, Optional


DOCKER_IMAGE_PATH_BASE = "308535385114.dkr.ecr.us-east-1.amazonaws.com/pytorch/"

DOCKER_IMAGE_VERSION = 339


@dataclass
class Conf:
    distro: str
    parms: List[str]
    parms_list_ignored_for_docker_image: Optional[List[str]] = None
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
        result = leading + ["linux", self.distro] + cuda_parms + self.parms
        if (not for_docker and self.parms_list_ignored_for_docker_image is not None):
            result = result + self.parms_list_ignored_for_docker_image
        return result

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

    def gen_workflow_params(self, phase):
        parameters = OrderedDict()
        build_job_name_pieces = self.get_build_job_name_pieces(phase)

        build_env_name = "-".join(map(str, build_job_name_pieces))
        parameters["build_environment"] = miniutils.quote(build_env_name)
        parameters["docker_image"] = self.gen_docker_image_path()
        if phase == "test" and self.gpu_resource:
            parameters["use_cuda_docker_runtime"] = miniutils.quote("1")
        if phase == "test":
            resource_class = "large"
            if self.gpu_resource:
                resource_class = "gpu." + self.gpu_resource
            parameters["resource_class"] = resource_class
        return parameters

    def gen_workflow_job(self, phase):
        # All jobs require the setup job
        job_def = OrderedDict()
        job_def["name"] = self.gen_build_name(phase)
        job_def["requires"] = ["setup"]

        if phase == "test":

            # TODO When merging the caffe2 and pytorch jobs, it might be convenient for a while to make a
            #  caffe2 test job dependent on a pytorch build job. This way we could quickly dedup the repeated
            #  build of pytorch in the caffe2 build job, and just run the caffe2 tests off of a completed
            #  pytorch build job (from https://github.com/pytorch/pytorch/pull/17323#discussion_r259452641)

            dependency_build = self.parent_build or self
            job_def["requires"].append(dependency_build.gen_build_name("build"))
            job_name = "pytorch_linux_test"
        else:
            job_name = "pytorch_linux_build"


        if not self.is_important:
            # If you update this, update
            # caffe2_build_definitions.py too
            job_def["filters"] = {"branches": {"only": ["master", r"/ci-all\/.*/"]}}
        job_def.update(self.gen_workflow_params(phase))

        return {job_name : job_def}


# TODO This is a hack to special case some configs just for the workflow list
class HiddenConf(object):
    def __init__(self, name, parent_build=None):
        self.name = name
        self.parent_build = parent_build

    def gen_workflow_job(self, phase):
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

    for x in ["pytorch_short_perf_test_gpu", "pytorch_python_doc_push", "pytorch_cpp_doc_push"]:
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
        compiler_name = fc.find_prop("compiler_name")
        is_xla = fc.find_prop("is_xla") or False
        parms_list_ignored_for_docker_image = []

        python_version = None
        if compiler_name == "cuda" or compiler_name == "android":
            python_version = fc.find_prop("pyver")
            parms_list = [fc.find_prop("abbreviated_pyver")]
        else:
            parms_list = ["py" + fc.find_prop("pyver")]

        cuda_version = None
        if compiler_name == "cuda":
            cuda_version = fc.find_prop("compiler_version")

        elif compiler_name == "android":
            android_ndk_version = fc.find_prop("compiler_version")
            # TODO: do we need clang to compile host binaries like protoc?
            parms_list.append("clang5")
            parms_list.append("android-ndk-" + android_ndk_version)
            android_abi = fc.find_prop("android_abi")
            parms_list_ignored_for_docker_image.append(android_abi)
            restrict_phases = ["build"]

        elif compiler_name:
            gcc_version = compiler_name + (fc.find_prop("compiler_version") or "")
            parms_list.append(gcc_version)

            # TODO: This is a nasty special case
            if compiler_name == "clang" and not is_xla:
                parms_list.append("asan")
                python_version = fc.find_prop("pyver")
                parms_list[0] = fc.find_prop("abbreviated_pyver")

        if cuda_version in ["9.2", "10", "10.1"]:
            # TODO The gcc version is orthogonal to CUDA version?
            parms_list.append("gcc7")

        is_namedtensor = fc.find_prop("is_namedtensor") or False
        is_important = fc.find_prop("is_important") or False

        gpu_resource = None
        if cuda_version and cuda_version != "10":
            gpu_resource = "medium"

        c = Conf(
            distro_name,
            parms_list,
            parms_list_ignored_for_docker_image,
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


@dataclass
class BuildParameters:
    """
    Corresponds to parameters specified in pytorch-build-params.yml
    """
    name: str
    build_environment: str
    docker_image: str
    resource_class: str
    use_cuda_docker_runtime: str
    requires: List[str]


@dataclass
class Spec:
    base_name: str
    phase: str
    resource_class: str = "large"
    extra_params: Optional[str] = None
    dependency: "Optional[Spec]" = None

    @property
    def job_name(self):
        full_name = self.base_name
        if self.extra_params:
            full_name += "_" + self.extra_params

        return full_name.replace(".", "_").replace("-", "_") + "_" + self.phase

    @property
    def docker_image(self):
        return DOCKER_IMAGE_PATH_BASE + self.base_name + ":" + str(DOCKER_IMAGE_VERSION)

    @property
    def build_environment(self):
        if self.extra_params is not None:
            extra_params = "-" + self.extra_params
        else:
            extra_params = ""

        return self.base_name + extra_params + "-" + self.phase


@dataclass
class NamedTensorSpec(Spec):
    @property
    def docker_image(self):
        docker_image = super().docker_image
        return docker_image.replace("namedtensor-", "")

@dataclass
class XLASpec(Spec):
    @property
    def docker_image(self):
        docker_image = super().docker_image
        return docker_image.replace("xla-", "")

specs = []

specs += Spec(base_name="pytorch-linux-xenial-py2.7.9", phase="build"),
specs += Spec(base_name="pytorch-linux-xenial-py2.7.9", phase="test", dependency=specs[-1]),
specs += Spec(base_name="pytorch-linux-xenial-py2.7",  phase="build"),
specs += Spec(base_name="pytorch-linux-xenial-py2.7", phase="test", dependency=specs[-1]),
specs += Spec(base_name="pytorch-linux-xenial-py3.5", phase="build"),
specs += Spec(base_name="pytorch-linux-xenial-py3.5", phase="test", dependency=specs[-1]),
specs += Spec(base_name="pytorch-linux-xenial-pynightly", phase="build"),
specs += Spec(base_name="pytorch-linux-xenial-pynightly", phase="test", dependency=specs[-1]),
specs += Spec(base_name="pytorch-linux-xenial-py3.6-gcc4.8", phase="build"),
specs += Spec(base_name="pytorch-linux-xenial-py3.6-gcc4.8", phase="test", dependency=specs[-1]),
specs += Spec(base_name="pytorch-linux-xenial-py3.6-gcc5.4", phase="build"),
specs += Spec(base_name="pytorch-linux-xenial-py3.6-gcc5.4", phase="test", dependency=specs[-1]),
specs += NamedTensorSpec(base_name="pytorch-namedtensor-linux-xenial-py3.6-gcc5.4", phase="build"),
specs += NamedTensorSpec(base_name="pytorch-namedtensor-linux-xenial-py3.6-gcc5.4", phase="test", dependency=specs[-1]),
specs += Spec(base_name="pytorch-linux-xenial-py3.6-gcc7", phase="build"),
specs += Spec(base_name="pytorch-linux-xenial-py3.6-gcc7", phase="test", dependency=specs[-1]),
specs += Spec(base_name="pytorch-linux-xenial-py3-clang5-asan", phase="build"),
specs += Spec(base_name="pytorch-linux-xenial-py3-clang5-asan", phase="test", dependency=specs[-1]),
specs += NamedTensorSpec(base_name="pytorch-namedtensor-linux-xenial-py3-clang5-asan", phase="build"),
specs += NamedTensorSpec(base_name="pytorch-namedtensor-linux-xenial-py3-clang5-asan", phase="test", dependency=specs[-1]),
specs += XLASpec(base_name="pytorch-xla-linux-xenial-py3.6-clang7", phase="build"),
specs += XLASpec(base_name="pytorch-xla-linux-xenial-py3.6-clang7", phase="test", dependency=specs[-1]),
specs += Spec(base_name="pytorch-linux-xenial-cuda9-cudnn7-py2", phase="build"),
specs += Spec(base_name="pytorch-linux-xenial-cuda9-cudnn7-py2", phase="test", resource_class="gpu.medium", dependency=specs[-1]),

gpu_py3_spec = Spec(base_name="pytorch-linux-xenial-cuda9-cudnn7-py3", phase="build")
specs += gpu_py3_spec,
specs += Spec(base_name="pytorch-linux-xenial-cuda9-cudnn7-py3", phase="test", resource_class="gpu.medium", dependency=gpu_py3_spec),
specs += Spec(base_name="pytorch-linux-xenial-cuda9-cudnn7-py3", extra_params="multigpu", resource_class="gpu.large", phase="test", dependency=gpu_py3_spec),
specs += Spec(base_name="pytorch-linux-xenial-cuda9-cudnn7-py3", extra_params="NO_AVX2", phase="test", resource_class="gpu.medium", dependency=gpu_py3_spec),
specs += Spec(base_name="pytorch-linux-xenial-cuda9-cudnn7-py3", extra_params="NO_AVX-NO_AVX2", phase="test", resource_class="gpu.medium", dependency=gpu_py3_spec),
specs += Spec(base_name="pytorch-linux-xenial-cuda9-cudnn7-py3", extra_params="slow", phase="test", resource_class="gpu.medium", dependency=gpu_py3_spec),
specs += Spec(base_name="pytorch-linux-xenial-cuda9-cudnn7-py3", extra_params="nogpu", phase="test", dependency=gpu_py3_spec),

specs += NamedTensorSpec(base_name="pytorch-namedtensor-linux-xenial-cuda9-cudnn7-py2", phase="build"),
specs += NamedTensorSpec(base_name="pytorch-namedtensor-linux-xenial-cuda9-cudnn7-py2", phase="test", resource_class="gpu.medium", dependency=specs[-1]),
specs += Spec(base_name="pytorch-linux-xenial-cuda9.2-cudnn7-py3-gcc7", phase="build"),
specs += Spec(base_name="pytorch-linux-xenial-cuda9.2-cudnn7-py3-gcc7", phase="test", resource_class="gpu.medium", dependency=specs[-1]),
specs += Spec(base_name="pytorch-linux-xenial-cuda10-cudnn7-py3-gcc7", phase="build"),
specs += Spec(base_name="pytorch-linux-xenial-cuda10.1-cudnn7-py3-gcc7", phase="build"),
specs += Spec(base_name="pytorch-linux-xenial-cuda10.1-cudnn7-py3-gcc7", phase="test", resource_class="gpu.medium", dependency=specs[-1]),
specs += Spec(base_name="pytorch-linux-xenial-py3-clang5-android-ndk-r19c", extra_params="x86_32", phase="build"),
specs += Spec(base_name="pytorch-linux-xenial-py3-clang5-android-ndk-r19c", extra_params="x86_64", phase="build"),
specs += Spec(base_name="pytorch-linux-xenial-py3-clang5-android-ndk-r19c", extra_params="arm-v7a", phase="build"),
specs += Spec(base_name="pytorch-linux-xenial-py3-clang5-android-ndk-r19c", extra_params="arm-v8a", phase="build"),


def get_jobs_from_spec(spec: Spec) -> List[BuildParameters]:
    job_name = spec.job_name
    build_environment = spec.build_environment
    docker_image = spec.docker_image
    resource_class = spec.resource_class

    use_cuda_docker_runtime = "1" if "gpu" in resource_class else ""

    requires = ["setup"]
    if spec.dependency:
        requires.append(spec.dependency.job_name)

    build_params = BuildParameters(
        name=job_name,
        build_environment=miniutils.quote(build_environment),
        docker_image=miniutils.quote(docker_image),
        resource_class=resource_class,
        use_cuda_docker_runtime=miniutils.quote(use_cuda_docker_runtime),
        requires=requires
    )
    base_job_name = "pytorch_linux_" + spec.phase
    return {base_job_name : asdict(build_params)}


def get_workflow_jobs():
    jobs = ["setup"]
    for spec in specs:
        jobs.append(get_jobs_from_spec(spec))

    # special jobs
    return jobs



    # config_list = instantiate_configs()

    # x = ["setup"]
    # for conf_options in config_list:

    #     phases = conf_options.restrict_phases or dimensions.PHASES

    #     for phase in phases:

    #         # TODO why does this not have a test?
    #         if phase == "test" and conf_options.cuda_version == "10":
    #             continue

    #         x.append(conf_options.gen_workflow_job(phase))

    #     # TODO convert to recursion
    #     for conf in conf_options.get_dependents():
    #         x.append(conf.gen_workflow_job("test"))

    # return x
