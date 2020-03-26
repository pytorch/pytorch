from collections import OrderedDict

import cimodel.data.dimensions as dimensions
import cimodel.lib.conf_tree as conf_tree
from cimodel.lib.conf_tree import Ver
import cimodel.lib.miniutils as miniutils
from cimodel.data.caffe2_build_data import CONFIG_TREE_DATA, TopLevelNode


from dataclasses import dataclass


DOCKER_IMAGE_PATH_BASE = "308535385114.dkr.ecr.us-east-1.amazonaws.com/caffe2/"

DOCKER_IMAGE_VERSION = "345"


@dataclass
class Conf:
    language: str
    distro: Ver
    # There could be multiple compiler versions configured (e.g. nvcc
    # for gpu files and host compiler (gcc/clang) for cpu files)
    compilers: [Ver]
    build_only: bool
    test_only: bool
    is_important: bool

    @property
    def compiler_names(self):
        return [c.name for c in self.compilers]

    # TODO: Eventually we can probably just remove the cudnn7 everywhere.
    def get_cudnn_insertion(self):

        omit = self.language == "onnx_py2" \
            or self.language == "onnx_main_py3.6" \
            or self.language == "onnx_ort1_py3.6" \
            or self.language == "onnx_ort2_py3.6" \
            or set(self.compiler_names).intersection({"android", "mkl", "clang"}) \
            or str(self.distro) in ["ubuntu14.04", "macos10.13"]

        return [] if omit else ["cudnn7"]

    def get_build_name_root_parts(self):
        return [
            "caffe2",
            self.language,
        ] + self.get_build_name_middle_parts()

    def get_build_name_middle_parts(self):
        return [str(c) for c in self.compilers] + self.get_cudnn_insertion() + [str(self.distro)]

    def construct_phase_name(self, phase):
        root_parts = self.get_build_name_root_parts()

        build_name_substitutions = {
            "onnx_ort1_py3.6": "onnx_main_py3.6",
            "onnx_ort2_py3.6": "onnx_main_py3.6",
        }
        if phase == "build":
            root_parts = [miniutils.override(r, build_name_substitutions) for r in root_parts]
        return "_".join(root_parts + [phase]).replace(".", "_")

    def get_platform(self):
        platform = self.distro.name
        if self.distro.name != "macos":
            platform = "linux"
        return platform

    def gen_docker_image(self):

        lang_substitutions = {
            "onnx_py2": "py2",
            "onnx_main_py3.6": "py3.6",
            "onnx_ort1_py3.6": "py3.6",
            "onnx_ort2_py3.6": "py3.6",
            "cmake": "py2",
        }

        lang = miniutils.override(self.language, lang_substitutions)
        parts = [lang] + self.get_build_name_middle_parts()
        return miniutils.quote(DOCKER_IMAGE_PATH_BASE + "-".join(parts) + ":" + str(DOCKER_IMAGE_VERSION))

    def gen_workflow_params(self, phase):
        parameters = OrderedDict()
        lang_substitutions = {
            "onnx_py2": "onnx-py2",
            "onnx_main_py3.6": "onnx-main-py3.6",
            "onnx_ort1_py3.6": "onnx-ort1-py3.6",
            "onnx_ort2_py3.6": "onnx-ort2-py3.6",
        }

        lang = miniutils.override(self.language, lang_substitutions)

        parts = [
            "caffe2",
            lang,
        ] + self.get_build_name_middle_parts() + [phase]

        build_env_name = "-".join(parts)
        parameters["build_environment"] = miniutils.quote(build_env_name)
        if "ios" in self.compiler_names:
            parameters["build_ios"] = miniutils.quote("1")
        if phase == "test":
            # TODO cuda should not be considered a compiler
            if "cuda" in self.compiler_names:
                parameters["use_cuda_docker_runtime"] = miniutils.quote("1")

        if self.distro.name != "macos":
            parameters["docker_image"] = self.gen_docker_image()
            if self.build_only:
                parameters["build_only"] = miniutils.quote("1")
        if phase == "test":
            resource_class = "large" if "cuda" not in self.compiler_names else "gpu.medium"
            parameters["resource_class"] = resource_class

        return parameters

    def gen_workflow_job(self, phase):
        job_def = OrderedDict()
        job_def["name"] = self.construct_phase_name(phase)
        job_def["requires"] = ["setup"]

        if phase == "test":
            job_def["requires"].append(self.construct_phase_name("build"))
            job_name = "caffe2_" + self.get_platform() + "_test"
        else:
            job_name = "caffe2_" + self.get_platform() + "_build"

        if not self.is_important:
            job_def["filters"] = {"branches": {"only": ["master", r"/ci-all\/.*/"]}}
        job_def.update(self.gen_workflow_params(phase))
        return {job_name : job_def}


def get_root():
    return TopLevelNode("Caffe2 Builds", CONFIG_TREE_DATA)


def instantiate_configs():

    config_list = []

    root = get_root()
    found_configs = conf_tree.dfs(root)
    for fc in found_configs:
        c = Conf(
            language=fc.find_prop("language_version"),
            distro=fc.find_prop("distro_version"),
            compilers=fc.find_prop("compiler_version"),
            build_only=fc.find_prop("build_only"),
            test_only=fc.find_prop("test_only"),
            is_important=fc.find_prop("important"),
        )

        config_list.append(c)

    return config_list


def get_workflow_jobs():

    configs = instantiate_configs()

    x = []
    for conf_options in configs:
        phases = ["build"]
        if not conf_options.build_only:
            phases = dimensions.PHASES
        if conf_options.test_only:
            phases = ["test"]

        for phase in phases:
            x.append(conf_options.gen_workflow_job(phase))

    return x
