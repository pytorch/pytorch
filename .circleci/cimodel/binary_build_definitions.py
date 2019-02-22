#!/usr/bin/env python3

from collections import OrderedDict

import cimodel.conf_tree as conf_tree
import cimodel.miniutils as miniutils
import cimodel.make_build_configs as make_build_configs
import cimodel.visualization as visualization


class Conf(object):
    def __init__(self, os, cuda_version, pydistro, parms, smoke=False, libtorch_variant=None):

        self.os = os
        self.cuda_version = cuda_version
        self.pydistro = pydistro
        self.parms = parms
        self.smoke = smoke
        self.libtorch_variant = libtorch_variant

    def gen_build_env_parms(self):
        return [self.pydistro] + self.parms + [make_build_configs.get_processor_arch_name(self.cuda_version)]

    def gen_docker_image(self):

        docker_word_substitution = {
            "manywheel": "manylinux",
            "libtorch": "manylinux",
        }

        docker_distro_prefix = miniutils.override(self.pydistro, docker_word_substitution)

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

        env_dict = OrderedDict({
            "BUILD_ENVIRONMENT": miniutils.quote(" ".join(self.gen_build_env_parms())),
        })

        if self.libtorch_variant:
            env_dict["LIBTORCH_VARIANT"] = miniutils.quote(self.libtorch_variant)

        os_word_substitution = {
            "macos": "mac",
        }

        os_name = miniutils.override(self.os, os_word_substitution)

        d = {
            "environment": env_dict,
            "<<": "*" + "_".join([self.get_name_prefix(), os_name, build_or_test]),
        }

        if build_or_test == "test":
            tuples = []

            if not (self.smoke and self.os == "macos"):
                tuples.append(("DOCKER_IMAGE", self.gen_docker_image()))

            if self.cuda_version:
                tuples.append(("USE_CUDA_DOCKER_RUNTIME", miniutils.quote("1")))

            for (k, v) in tuples:
                env_dict[k] = v

        else:
            if self.os == "linux" and build_or_test != "upload":
                d["docker"] = [{"image": self.gen_docker_image()}]

        if build_or_test == "test":
            if self.cuda_version:
                d["resource_class"] = "gpu.medium"

        return d


def get_root(smoke):

    return make_build_configs.TopLevelNode(
        "Builds",
        make_build_configs.CONFIG_TREE_DATA,
        smoke,
    )


def gen_build_env_list(smoke):

    root = get_root(smoke)
    config_list = conf_tree.dfs(root)

    newlist = []
    for c in config_list:
        conf = Conf(
            c.find_prop("os_name"),
            c.find_prop("cu"),
            c.find_prop("package_format"),
            [c.find_prop("pyver")],
            c.find_prop("smoke"),
            c.find_prop("libtorch_variant")
        )
        newlist.append(conf)

    return newlist


def add_build_entries(jobs_dict, phase, smoke):

    configs = gen_build_env_list(smoke)
    for conf_options in configs:
        jobs_dict[conf_options.gen_build_name(phase)] = conf_options.gen_yaml_tree(phase)


def add_binary_build_specs(jobs_dict):
    add_build_entries(jobs_dict, "build", False)


def add_binary_build_uploads(jobs_dict):
    add_build_entries(jobs_dict, "upload", False)


def add_smoke_test_specs(jobs_dict):
    add_build_entries(jobs_dict, "test", True)


def get_nightly_tests():

    configs = gen_build_env_list(False)
    filtered_configs = filter(predicate_exclude_nonlinux_and_libtorch, configs)

    mylist = []
    for conf_options in filtered_configs:
        d = {conf_options.gen_build_name("test"): {"requires": [conf_options.gen_build_name("build")]}}
        mylist.append(d)

    return mylist


def get_nightly_uploads():

    configs = gen_build_env_list(False)

    def gen_config(conf, phase_dependency):
        return {
            conf.gen_build_name("upload"): OrderedDict([
                ("context", "org-member"),
                ("requires", [conf.gen_build_name(phase_dependency)]),
            ]),
        }

    mylist = []
    for conf in configs:
        phase_dependency = "test" if predicate_exclude_nonlinux_and_libtorch(conf) else "build"
        mylist.append(gen_config(conf, phase_dependency))

    return mylist


def predicate_exclude_nonlinux_and_libtorch(config):
    return config.os == "linux" and (config.smoke or config.pydistro != "libtorch")


def add_binary_build_tests(jobs_dict):

    configs = gen_build_env_list(False)
    filtered_configs = filter(predicate_exclude_nonlinux_and_libtorch, configs)

    for conf_options in filtered_configs:
        jobs_dict[conf_options.gen_build_name("test")] = conf_options.gen_yaml_tree("test")


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

    jobs_list = []

    configs = gen_build_env_list(smoke)
    for build_config in configs:
        build_name = build_config.gen_build_name("build")
        jobs_list.append(build_name)

    d = OrderedDict(
        triggers=gen_schedule_tree(cron_schedule),
        jobs=jobs_list,
    )

    jobs_dict[toplevel_key] = d

    graph = visualization.generate_graph(get_root(smoke))
    graph.draw(toplevel_key + "-config-dimensions.png", prog="twopi")


def add_binary_build_jobs(jobs_dict):
    add_jobs_and_render(jobs_dict, "binarybuilds", False, "5 5 * * *")


def add_binary_smoke_test_jobs(jobs_dict):
    add_jobs_and_render(jobs_dict, "binarysmoketests", True, "15 16 * * *")
