from collections import OrderedDict

import cimodel.data.binary_build_data as binary_build_data
import cimodel.lib.conf_tree as conf_tree
import cimodel.lib.miniutils as miniutils


class Conf(object):
    def __init__(self, os, cuda_version, pydistro, parms, smoke, libtorch_variant, gcc_config_variant):

        self.os = os
        self.cuda_version = cuda_version
        self.pydistro = pydistro
        self.parms = parms
        self.smoke = smoke
        self.libtorch_variant = libtorch_variant
        self.gcc_config_variant = gcc_config_variant

    def gen_build_env_parms(self):
        elems = [self.pydistro] + self.parms + [binary_build_data.get_processor_arch_name(self.cuda_version)]
        if self.gcc_config_variant is not None:
            elems.append(str(self.gcc_config_variant))
        return elems

    def gen_docker_image(self):
        if self.gcc_config_variant == 'gcc5.4_cxx11-abi':
            return miniutils.quote("pytorch/pytorch-binary-docker-image-ubuntu16.04:latest")

        docker_word_substitution = {
            "manywheel": "manylinux",
            "libtorch": "manylinux",
        }

        docker_distro_prefix = miniutils.override(self.pydistro, docker_word_substitution)

        # The cpu nightlies are built on the pytorch/manylinux-cuda102 docker image
        alt_docker_suffix = self.cuda_version or "102"
        docker_distro_suffix = "" if self.pydistro == "conda" else alt_docker_suffix
        return miniutils.quote("pytorch/" + docker_distro_prefix + "-cuda" + docker_distro_suffix)

    def get_name_prefix(self):
        return "smoke" if self.smoke else "binary"

    def gen_build_name(self, build_or_test, nightly):

        parts = [self.get_name_prefix(), self.os] + self.gen_build_env_parms()

        if nightly:
            parts.append("nightly")

        if self.libtorch_variant:
            parts.append(self.libtorch_variant)

        if not self.smoke:
            parts.append(build_or_test)

        joined = "_".join(parts)
        return joined.replace(".", "_")

    def gen_workflow_job(self, phase, upload_phase_dependency=None, nightly=False):
        job_def = OrderedDict()
        job_def["name"] = self.gen_build_name(phase, nightly)
        job_def["build_environment"] = miniutils.quote(" ".join(self.gen_build_env_parms()))
        job_def["requires"] = ["setup"]
        if self.smoke:
            job_def["requires"].append("update_s3_htmls_for_nightlies")
            job_def["requires"].append("update_s3_htmls_for_nightlies_devtoolset7")
            job_def["filters"] = {"branches": {"only": "postnightly"}}
        else:
            job_def["filters"] = {
                "branches": {
                    "only": "nightly"
                },
                # Will run on tags like v1.5.0-rc1, etc.
                "tags": {
                    # Using a raw string here to avoid having to escape
                    # anything
                    "only": r"/v[0-9]+(\.[0-9]+)*-rc[0-9]+/"
                }
            }
        if self.libtorch_variant:
            job_def["libtorch_variant"] = miniutils.quote(self.libtorch_variant)
        if phase == "test":
            if not self.smoke:
                job_def["requires"].append(self.gen_build_name("build", nightly))
            if not (self.smoke and self.os == "macos"):
                job_def["docker_image"] = self.gen_docker_image()

            if self.cuda_version:
                job_def["use_cuda_docker_runtime"] = miniutils.quote("1")
        else:
            if self.os == "linux" and phase != "upload":
                job_def["docker_image"] = self.gen_docker_image()

        if phase == "test":
            if self.cuda_version:
                job_def["resource_class"] = "gpu.medium"
        if phase == "upload":
            job_def["context"] = "org-member"
            job_def["requires"] = ["setup", self.gen_build_name(upload_phase_dependency, nightly)]

        os_name = miniutils.override(self.os, {"macos": "mac"})
        job_name = "_".join([self.get_name_prefix(), os_name, phase])
        return {job_name : job_def}

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
            c.find_prop("gcc_config_variant"),
        )
        newlist.append(conf)

    return newlist


def predicate_exclude_nonlinux_and_libtorch(config):
    return config.os == "linux"


def get_nightly_uploads():
    configs = gen_build_env_list(False)
    mylist = []
    for conf in configs:
        phase_dependency = "test" if predicate_exclude_nonlinux_and_libtorch(conf) else "build"
        mylist.append(conf.gen_workflow_job("upload", phase_dependency, nightly=True))

    return mylist

def get_nightly_tests():

    configs = gen_build_env_list(False)
    filtered_configs = filter(predicate_exclude_nonlinux_and_libtorch, configs)

    tests = []
    for conf_options in filtered_configs:
        yaml_item = conf_options.gen_workflow_job("test", nightly=True)
        tests.append(yaml_item)

    return tests


def get_jobs(toplevel_key, smoke):
    jobs_list = []
    configs = gen_build_env_list(smoke)
    phase = "build" if toplevel_key == "binarybuilds" else "test"
    for build_config in configs:
        jobs_list.append(build_config.gen_workflow_job(phase, nightly=True))

    return jobs_list


def get_binary_build_jobs():
    return get_jobs("binarybuilds", False)


def get_binary_smoke_test_jobs():
    return get_jobs("binarysmoketests", True)
