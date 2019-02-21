from collections import OrderedDict

import conf_tree
import miniutils
import make_build_configs


class Conf:
    def __init__(self, os, cuda_version, pydistro, parms, smoke=False, libtorch_variant=None):

        self.os = os
        self.cuda_version = cuda_version
        self.pydistro = pydistro
        self.parms = parms
        self.smoke = smoke
        self.libtorch_variant = libtorch_variant

    def genBuildEnvParms(self):
        return [self.pydistro] + self.parms + [make_build_configs.get_processor_arch_name(self.cuda_version)]

    def genDockerImage(self):

        docker_word_substitution = {
            "manywheel": "manylinux",
            "libtorch": "manylinux",
        }

        docker_distro_prefix = miniutils.override(self.pydistro, docker_word_substitution)

        alt_docker_suffix = self.cuda_version or "80"
        docker_distro_suffix = "" if self.pydistro == "conda" else alt_docker_suffix
        return miniutils.quote("soumith/" + docker_distro_prefix + "-cuda" + docker_distro_suffix)

    def getNamePrefix(self):
        return "smoke" if self.smoke else "binary"

    def genBuildName(self, build_or_test):
        parts = [self.getNamePrefix(), self.os] + self.genBuildEnvParms()

        if self.smoke:
            if self.libtorch_variant:
                parts.append(self.libtorch_variant)
        else:
            parts.append(build_or_test)

        return "_".join(parts)

    def genYamlTree(self, build_or_test):

        env_dict = OrderedDict({
            "BUILD_ENVIRONMENT": miniutils.quote(" ".join(self.genBuildEnvParms())),
        })

        if self.libtorch_variant:
            env_dict["LIBTORCH_VARIANT"] = miniutils.quote(self.libtorch_variant)

        os_word_substitution = {
            "macos": "mac",
        }

        os_name = miniutils.override(self.os, os_word_substitution)

        d = {
            "environment": env_dict,
            "<<": "*" + "_".join([self.getNamePrefix(), os_name, build_or_test]),
        }

        if build_or_test == "test":
            tuples = []
            if self.cuda_version:
                tuples.append(("USE_CUDA_DOCKER_RUNTIME", miniutils.quote("1")))

            if not (self.smoke and self.os == "macos"):
                tuples.append(("DOCKER_IMAGE", self.genDockerImage()))

            if self.smoke:
                # TODO: Fix this discrepancy upstream
                tuples.reverse()

            for (k, v) in tuples:
                env_dict[k] = v

        else:
            if self.os == "linux" and build_or_test != "upload":
                d["docker"] = [{"image": self.genDockerImage()}]

        if build_or_test == "test":
            if self.cuda_version:
                d["resource_class"] = "gpu.medium"

        return d


def gen_build_env_list(smoke):

    root = make_build_configs.TopLevelNode(
        "Builds",
        make_build_configs.CONFIG_TREE_DATA,
        smoke,
    )

    config_list, dot = conf_tree.dfs(root)

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

    return newlist, dot


def add_build_entries(jobs_dict, phase, smoke):

    configs, _ = gen_build_env_list(smoke)
    for conf_options in configs:
        jobs_dict[conf_options.genBuildName(phase)] = conf_options.genYamlTree(phase)


def add_binary_build_specs(jobs_dict):
    add_build_entries(jobs_dict, "build", False)


def add_binary_build_uploads(jobs_dict):
    add_build_entries(jobs_dict, "upload", False)


def add_smoke_test_specs(jobs_dict):
    add_build_entries(jobs_dict, "test", True)


def add_binary_build_tests(jobs_dict):

    def testable_binary_predicate(x):
        return x.os == "linux" and (x.smoke or x.pydistro != "libtorch")

    configs, _ = gen_build_env_list(False)
    filtered_configs = filter(testable_binary_predicate, configs)

    for conf_options in filtered_configs:
        jobs_dict[conf_options.genBuildName("test")] = conf_options.genYamlTree("test")


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

    configs, graph = gen_build_env_list(smoke)
    for build_config in configs:
        build_name = build_config.genBuildName("build")
        jobs_list.append(build_name)

    d = OrderedDict(
        triggers=gen_schedule_tree(cron_schedule),
        jobs=jobs_list,
    )

    jobs_dict[toplevel_key] = d

    graph.draw(toplevel_key + "-config-dimensions.png", prog="twopi")


def add_binary_build_jobs(jobs_dict):
    add_jobs_and_render(jobs_dict, "binarybuilds", False, "5 5 * * *")


def add_binary_smoke_test_jobs(jobs_dict):
    add_jobs_and_render(jobs_dict, "binarysmoketests", True, "15 16 * * *")
