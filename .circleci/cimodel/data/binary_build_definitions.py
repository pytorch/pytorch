#!/usr/bin/env python3

from collections import OrderedDict

import cimodel.data.binary_build_data as binary_build_data
import cimodel.lib.conf_tree as conf_tree
import cimodel.lib.miniutils as miniutils
import cimodel.lib.visualization as visualization


class Conf(object):
    def __init__(self, os, cuda_version, pydistro, parms, smoke, libtorch_variant, devtoolset_version):

        self.os = os
        self.cuda_version = cuda_version
        self.pydistro = pydistro
        self.parms = parms
        self.smoke = smoke
        self.libtorch_variant = libtorch_variant
        self.devtoolset_version = devtoolset_version

    def gen_build_env_parms(self):
        elems = [self.pydistro] + self.parms + [binary_build_data.get_processor_arch_name(self.cuda_version)]
        if self.devtoolset_version is not None:
            elems.append("devtoolset" + str(self.devtoolset_version))
        return elems

    def gen_docker_image(self):

        docker_word_substitution = {
            "manywheel": "manylinux",
            "libtorch": "manylinux",
        }

        docker_distro_prefix = miniutils.override(self.pydistro, docker_word_substitution)

        # The cpu nightlies are built on the soumith/manylinux-cuda80 docker image
        alt_docker_suffix = self.cuda_version or "80"
        docker_distro_suffix = "" if self.pydistro == "conda" else alt_docker_suffix
        return miniutils.quote("soumith/" + docker_distro_prefix + "-cuda" + docker_distro_suffix)

    def get_name_prefix(self):
        return "smoke" if self.smoke else "binary"

    def gen_build_name(self, build_or_test):

        parts = [self.get_name_prefix(), self.os] + self.gen_build_env_parms()

        if self.smoke:
            if self.libtorch_variant:
                parts.append(self.libtorch_variant)
        else:
            parts.append(build_or_test)

        return "_".join(parts)

    def gen_yaml_tree(self, build_or_test):

        env_tuples = [("BUILD_ENVIRONMENT", miniutils.quote(" ".join(self.gen_build_env_parms())))]

        if self.libtorch_variant:
            env_tuples.append(("LIBTORCH_VARIANT", miniutils.quote(self.libtorch_variant)))

        os_name = miniutils.override(self.os, {"macos": "mac"})
        d = {"<<": "*" + "_".join([self.get_name_prefix(), os_name, build_or_test])}

        if build_or_test == "test":

            if not (self.smoke and self.os == "macos"):
                env_tuples.append(("DOCKER_IMAGE", self.gen_docker_image()))

            if self.cuda_version:
                env_tuples.append(("USE_CUDA_DOCKER_RUNTIME", miniutils.quote("1")))

        else:
            if self.os == "linux" and build_or_test != "upload":
                d["docker"] = [{"image": self.gen_docker_image()}]

        d["environment"] = OrderedDict(env_tuples)

        if build_or_test == "test":
            if self.cuda_version:
                d["resource_class"] = "gpu.medium"

        return d


def get_root(smoke, name):

    return binary_build_data.TopLevelNode(
        name,
        binary_build_data.CONFIG_TREE_DATA,
        smoke,
    )


def gen_build_env_list(smoke):

    root = get_root(smoke, "N/A")
    config_list = conf_tree.dfs(root)

    newlist = []
    for c in config_list:
        conf = Conf(
            c.find_prop("os_name"),
            c.find_prop("cu"),
            c.find_prop("package_format"),
            [c.find_prop("pyver")],
            c.find_prop("smoke"),
            c.find_prop("libtorch_variant"),
            c.find_prop("devtoolset_version"),
        )
        newlist.append(conf)

    return newlist


def predicate_exclude_nonlinux_and_libtorch(config):
    return config.os == "linux" and (config.smoke or config.pydistro != "libtorch")


def add_build_entries(jobs_dict, phase, smoke, filter_predicate=lambda x: True):

    configs = gen_build_env_list(smoke)
    for conf_options in filter(filter_predicate, configs):
        jobs_dict[conf_options.gen_build_name(phase)] = conf_options.gen_yaml_tree(phase)


def add_binary_build_specs(jobs_dict):
    add_build_entries(jobs_dict, "build", False)


def add_binary_build_tests(jobs_dict):
    add_build_entries(jobs_dict, "test", False, predicate_exclude_nonlinux_and_libtorch)


def add_binary_build_uploads(jobs_dict):
    add_build_entries(jobs_dict, "upload", False)


def add_smoke_test_specs(jobs_dict):
    add_build_entries(jobs_dict, "test", True)


def get_nightly_tests():

    configs = gen_build_env_list(False)
    filtered_configs = filter(predicate_exclude_nonlinux_and_libtorch, configs)

    tests = []
    for conf_options in filtered_configs:
        params = {"requires": ["setup", conf_options.gen_build_name("build")]}
        tests.append({conf_options.gen_build_name("test"): params})

    return tests

def get_nightly_uploads():

    configs = gen_build_env_list(False)

    def gen_config(conf, phase_dependency):
        return {
            conf.gen_build_name("upload"): OrderedDict([
                ("context", "org-member"),
                ("requires", ["setup", conf.gen_build_name(phase_dependency)]),
            ]),
        }

    mylist = []
    for conf in configs:
        phase_dependency = "test" if predicate_exclude_nonlinux_and_libtorch(conf) else "build"
        mylist.append(gen_config(conf, phase_dependency))

    return mylist


def gen_schedule_tree(cron_timing):
    return [{
        "schedule": {
            "cron": miniutils.quote(cron_timing),
            "filters": {
                "branches": {
                    "only": ["master"],
                },
            },
        },
    }]


def add_jobs_and_render(jobs_dict, toplevel_key, smoke, cron_schedule):

    jobs_list = ["setup"]

    configs = gen_build_env_list(smoke)
    for build_config in configs:
        build_name = build_config.gen_build_name("build")
        jobs_list.append({build_name: {"requires": ["setup"]}})

    jobs_dict[toplevel_key] = OrderedDict(
        triggers=gen_schedule_tree(cron_schedule),
        jobs=jobs_list,
    )

    graph = visualization.generate_graph(get_root(smoke, toplevel_key))
    graph.draw(toplevel_key + "-config-dimensions.png", prog="twopi")


def add_binary_build_jobs(jobs_dict):
    add_jobs_and_render(jobs_dict, "binarybuilds", False, "5 5 * * *")


def add_binary_smoke_test_jobs(jobs_dict):
    add_jobs_and_render(jobs_dict, "binarysmoketests", True, "15 16 * * *")
