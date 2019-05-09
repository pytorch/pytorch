#!/usr/bin/env python3

from collections import OrderedDict

import cimodel.lib.conf_tree as conf_tree
import cimodel.lib.miniutils as miniutils
import cimodel.lib.visualization as visualization
from cimodel.data.caffe2_build_data import CONFIG_TREE_DATA, TopLevelNode


DOCKER_IMAGE_PATH_BASE = "308535385114.dkr.ecr.us-east-1.amazonaws.com/caffe2/"

DOCKER_IMAGE_VERSION = 276


class Conf(object):
    def __init__(self, language, distro, compiler, phase, build_only):

        self.language = language
        self.distro = distro
        self.compiler = compiler
        self.phase = phase
        self.build_only = build_only

    # TODO: Eventually we can probably just remove the cudnn7 everywhere.
    def get_cudnn_insertion(self):

        omit = self.language == "onnx_py2" \
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

    def get_name(self):
        return self.construct_phase_name(self.phase)

    def get_platform(self):
        platform = self.distro.name
        if self.distro.name != "macos":
            platform = "linux"
        return platform

    def gen_docker_image(self):

        lang_substitutions = {
            "onnx_py2": "py2",
            "cmake": "py2",
        }

        lang = miniutils.override(self.language, lang_substitutions)
        parts = [lang] + self.get_build_name_middle_parts()
        return miniutils.quote(DOCKER_IMAGE_PATH_BASE + "-".join(parts) + ":" + str(DOCKER_IMAGE_VERSION))

    def gen_yaml_tree(self):

        tuples = []

        lang_substitutions = {
            "onnx_py2": "onnx-py2",
        }

        lang = miniutils.override(self.language, lang_substitutions)

        parts = [
            "caffe2",
            lang,
        ] + self.get_build_name_middle_parts() + [self.phase]

        build_env = "-".join(parts)
        if not self.distro.name == "macos":
            build_env = miniutils.quote(build_env)

        tuples.append(("BUILD_ENVIRONMENT", build_env))

        if self.compiler.name == "ios":
            tuples.append(("BUILD_IOS", miniutils.quote("1")))

        if self.phase == "test":
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

        if self.phase == "test":
            resource_class = "large" if self.compiler.name != "cuda" else "gpu.medium"
            d["resource_class"] = resource_class

        d["<<"] = "*" + "_".join(["caffe2", self.get_platform(), self.phase, "defaults"])

        return d


def get_root():
    return TopLevelNode("Caffe2 Builds", CONFIG_TREE_DATA)


def instantiate_configs():

    config_list = []

    root = get_root()
    found_configs = conf_tree.dfs(root)
    for fc in found_configs:

        c = Conf(
            fc.find_prop("language_version"),
            fc.find_prop("distro_version"),
            fc.find_prop("compiler_version"),
            fc.find_prop("phase_name"),
            fc.find_prop("build_only"),
        )

        config_list.append(c)

    return config_list


def add_caffe2_builds(jobs_dict):

    configs = instantiate_configs()
    for conf_options in configs:
        jobs_dict[conf_options.get_name()] = conf_options.gen_yaml_tree()

    graph = visualization.generate_graph(get_root())
    graph.draw("caffe2-config-dimensions.png", prog="twopi")


def get_caffe2_workflows():

    configs = instantiate_configs()

    # TODO Why don't we build this config?
    # See https://github.com/pytorch/pytorch/pull/17323#discussion_r259450540
    filtered_configs = filter(lambda x: not (str(x.distro) == "ubuntu14.04" and str(x.compiler) == "gcc4.9"), configs)

    x = []
    for conf_options in filtered_configs:

        requires = ["setup"]

        if conf_options.phase == "test":
            requires.append(conf_options.construct_phase_name("build"))

        x.append({conf_options.get_name(): {"requires": requires}})

    return x
