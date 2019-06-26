#!/usr/bin/env python3

from collections import OrderedDict

import cimodel.data.dimensions as dimensions
import cimodel.lib.conf_tree as conf_tree
from cimodel.lib.conf_tree import Ver
import cimodel.lib.miniutils as miniutils
import cimodel.lib.visualization as visualization
from cimodel.data.caffe2_build_data import CONFIG_TREE_DATA, TopLevelNode


from dataclasses import dataclass


DOCKER_IMAGE_PATH_BASE = "308535385114.dkr.ecr.us-east-1.amazonaws.com/caffe2/"

DOCKER_IMAGE_VERSION = 287


@dataclass
class Conf:
    language: str
    distro: Ver
    compiler: Ver
    build_only: bool
    is_important: bool

    # TODO: Eventually we can probably just remove the cudnn7 everywhere.
    def get_cudnn_insertion(self):

        omit = self.language == "onnx_py2" \
            or self.language == "onnx_py3.6" \
            or self.compiler.name in ["android", "mkl", "clang"] \
            or str(self.distro) in ["ubuntu14.04", "macos10.13"]

        return [] if omit else ["cudnn7"]

    def get_build_name_root_parts(self):
        return [
            "caffe2",
            self.language,
        ] + self.get_build_name_middle_parts()

    def get_build_name_middle_parts(self):
        return [str(self.compiler)] + self.get_cudnn_insertion() + [str(self.distro)]

    def construct_phase_name(self, phase):
        root_parts = self.get_build_name_root_parts()
        return "_".join(root_parts + [phase]).replace(".", "_")

    def get_platform(self):
        platform = self.distro.name
        if self.distro.name != "macos":
            platform = "linux"
        return platform

    def gen_docker_image(self):

        lang_substitutions = {
            "onnx_py2": "py2",
            "onnx_py3.6": "py3.6",
            "cmake": "py2",
        }

        lang = miniutils.override(self.language, lang_substitutions)
        parts = [lang] + self.get_build_name_middle_parts()
        return miniutils.quote(DOCKER_IMAGE_PATH_BASE + "-".join(parts) + ":" + str(DOCKER_IMAGE_VERSION))

    def gen_yaml_tree(self, phase):

        tuples = []

        lang_substitutions = {
            "onnx_py2": "onnx-py2",
            "onnx_py3.6": "onnx-py3.6",
        }

        lang = miniutils.override(self.language, lang_substitutions)

        parts = [
            "caffe2",
            lang,
        ] + self.get_build_name_middle_parts() + [phase]

        build_env = "-".join(parts)
        if not self.distro.name == "macos":
            build_env = miniutils.quote(build_env)

        tuples.append(("BUILD_ENVIRONMENT", build_env))

        if self.compiler.name == "ios":
            tuples.append(("BUILD_IOS", miniutils.quote("1")))

        if phase == "test":
            # TODO cuda should not be considered a compiler
            if self.compiler.name == "cuda":
                tuples.append(("USE_CUDA_DOCKER_RUNTIME", miniutils.quote("1")))

        if self.distro.name == "macos":
            tuples.append(("PYTHON_VERSION", miniutils.quote("2")))

        else:
            tuples.append(("DOCKER_IMAGE", self.gen_docker_image()))
            if self.build_only:
                tuples.append(("BUILD_ONLY", miniutils.quote("1")))

        d = OrderedDict({"environment": OrderedDict(tuples)})

        if phase == "test":
            resource_class = "large" if self.compiler.name != "cuda" else "gpu.medium"
            d["resource_class"] = resource_class

        d["<<"] = "*" + "_".join(["caffe2", self.get_platform(), phase, "defaults"])

        return d


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
            compiler=fc.find_prop("compiler_version"),
            build_only=fc.find_prop("build_only"),
            is_important=fc.find_prop("important"),
        )

        config_list.append(c)

    return config_list


def add_caffe2_builds(jobs_dict):
    configs = instantiate_configs()
    for conf_options in configs:
        phases = ["build"]
        if not conf_options.build_only:
            phases = dimensions.PHASES
        for phase in phases:
            jobs_dict[conf_options.construct_phase_name(phase)] = conf_options.gen_yaml_tree(phase)

    graph = visualization.generate_graph(get_root())
    graph.draw("caffe2-config-dimensions.png", prog="twopi")


def get_caffe2_workflows():

    configs = instantiate_configs()

    # TODO Why don't we build this config?
    # See https://github.com/pytorch/pytorch/pull/17323#discussion_r259450540
    filtered_configs = filter(lambda x: not (str(x.distro) == "ubuntu14.04" and str(x.compiler) == "gcc4.9"), configs)

    x = []
    for conf_options in filtered_configs:

        phases = ["build"]
        if not conf_options.build_only:
            phases = dimensions.PHASES

        for phase in phases:

            requires = ["setup"]
            sub_d = {"requires": requires}

            if phase == "test":
                requires.append(conf_options.construct_phase_name("build"))

            if not conf_options.is_important:
                # If you update this, update
                # pytorch_build_definitions.py too
                sub_d["filters"] = {"branches": {"only": ["master", r"/ci-all\/.*/"]}}

            x.append({conf_options.construct_phase_name(phase): sub_d})

    return x
