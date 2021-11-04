from collections import OrderedDict

import cimodel.data.simple.util.branch_filters as branch_filters
import cimodel.data.binary_build_data as binary_build_data
import cimodel.lib.conf_tree as conf_tree
import cimodel.lib.miniutils as miniutils

class Conf(object):
    def __init__(self, os, gpu_version, pydistro, parms, smoke, libtorch_variant, gcc_config_variant, libtorch_config_variant):

        self.os = os
        self.gpu_version = gpu_version
        self.pydistro = pydistro
        self.parms = parms
        self.smoke = smoke
        self.libtorch_variant = libtorch_variant
        self.gcc_config_variant = gcc_config_variant
        self.libtorch_config_variant = libtorch_config_variant

    def gen_build_env_parms(self):
        elems = [self.pydistro] + self.parms + [binary_build_data.get_processor_arch_name(self.gpu_version)]
        if self.gcc_config_variant is not None:
            elems.append(str(self.gcc_config_variant))
        if self.libtorch_config_variant is not None:
            elems.append(str(self.libtorch_config_variant))
        return elems

    def gen_docker_image(self):
        if self.gcc_config_variant == 'gcc5.4_cxx11-abi':
            if self.gpu_version is None:
                return miniutils.quote("pytorch/libtorch-cxx11-builder:cpu")
            else:
                return miniutils.quote(
                    f"pytorch/libtorch-cxx11-builder:{self.gpu_version}"
                )
        if self.pydistro == "conda":
            if self.gpu_version is None:
                return miniutils.quote("pytorch/conda-builder:cpu")
            else:
                return miniutils.quote(
                    f"pytorch/conda-builder:{self.gpu_version}"
                )

        docker_word_substitution = {
            "manywheel": "manylinux",
            "libtorch": "manylinux",
        }

        docker_distro_prefix = miniutils.override(self.pydistro, docker_word_substitution)

        # The cpu nightlies are built on the pytorch/manylinux-cuda102 docker image
        # TODO cuda images should consolidate into tag-base images similar to rocm
        alt_docker_suffix = "cuda102" if not self.gpu_version else (
            "rocm:" + self.gpu_version.strip("rocm") if self.gpu_version.startswith("rocm") else self.gpu_version)
        docker_distro_suffix = alt_docker_suffix if self.pydistro != "conda" else (
            "cuda" if alt_docker_suffix.startswith("cuda") else "rocm")
        return miniutils.quote("pytorch/" + docker_distro_prefix + "-" + docker_distro_suffix)

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
        if self.smoke:
            job_def["requires"] = [
                "update_s3_htmls",
            ]
            job_def["filters"] = branch_filters.gen_filter_dict(
                branches_list=["postnightly"],
            )
        else:
            filter_branch = r"/.*/"
            job_def["filters"] = branch_filters.gen_filter_dict(
                branches_list=[filter_branch],
                tags_list=[branch_filters.RC_PATTERN],
            )
        if self.libtorch_variant:
            job_def["libtorch_variant"] = miniutils.quote(self.libtorch_variant)
        if phase == "test":
            if not self.smoke:
                job_def["requires"] = [self.gen_build_name("build", nightly)]
            if not (self.smoke and self.os == "macos") and self.os != "windows":
                job_def["docker_image"] = self.gen_docker_image()

            # fix this. only works on cuda not rocm
            if self.os != "windows" and self.gpu_version:
                job_def["use_cuda_docker_runtime"] = miniutils.quote("1")
        else:
            if self.os == "linux" and phase != "upload":
                job_def["docker_image"] = self.gen_docker_image()

        if phase == "test":
            if self.gpu_version:
                if self.os == "windows":
                    job_def["executor"] = "windows-with-nvidia-gpu"
                else:
                    job_def["resource_class"] = "gpu.medium"

        os_name = miniutils.override(self.os, {"macos": "mac"})
        job_name = "_".join([self.get_name_prefix(), os_name, phase])
        return {job_name : job_def}

    def gen_upload_job(self, phase, requires_dependency):
        """Generate binary_upload job for configuration

        Output looks similar to:

      - binary_upload:
          name: binary_linux_manywheel_3_7m_cu113_devtoolset7_nightly_upload
          context: org-member
          requires: binary_linux_manywheel_3_7m_cu113_devtoolset7_nightly_test
          filters:
            branches:
              only:
                - nightly
            tags:
              only: /v[0-9]+(\\.[0-9]+)*-rc[0-9]+/
          package_type: manywheel
          upload_subfolder: cu113
        """
        return {
            "binary_upload": OrderedDict({
                "name": self.gen_build_name(phase, nightly=True),
                "context": "org-member",
                "requires": [self.gen_build_name(
                    requires_dependency,
                    nightly=True
                )],
                "filters": branch_filters.gen_filter_dict(
                    branches_list=["nightly"],
                    tags_list=[branch_filters.RC_PATTERN],
                ),
                "package_type": self.pydistro,
                "upload_subfolder": binary_build_data.get_processor_arch_name(
                    self.gpu_version,
                ),
            })
        }

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
            c.find_prop("gpu"),
            c.find_prop("package_format"),
            [c.find_prop("pyver")],
            c.find_prop("smoke") and not (c.find_prop("os_name") == "macos_arm64"),  # don't test arm64
            c.find_prop("libtorch_variant"),
            c.find_prop("gcc_config_variant"),
            c.find_prop("libtorch_config_variant"),
        )
        newlist.append(conf)

    return newlist

def predicate_exclude_macos(config):
    return config.os == "linux" or config.os == "windows"

def get_nightly_uploads():
    configs = gen_build_env_list(False)
    mylist = []
    for conf in configs:
        phase_dependency = "test" if predicate_exclude_macos(conf) else "build"
        mylist.append(conf.gen_upload_job("upload", phase_dependency))

    return mylist

def get_post_upload_jobs():
    return [
        {
            "update_s3_htmls": {
                "name": "update_s3_htmls",
                "context": "org-member",
                "filters": branch_filters.gen_filter_dict(
                    branches_list=["postnightly"],
                ),
            },
        },
    ]

def get_nightly_tests():

    configs = gen_build_env_list(False)
    filtered_configs = filter(predicate_exclude_macos, configs)

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
        # don't test for macos_arm64 as it's cross compiled
        if phase != "test" or build_config.os != "macos_arm64":
            jobs_list.append(build_config.gen_workflow_job(phase, nightly=True))

    return jobs_list


def get_binary_build_jobs():
    return get_jobs("binarybuilds", False)


def get_binary_smoke_test_jobs():
    return get_jobs("binarysmoketests", True)
