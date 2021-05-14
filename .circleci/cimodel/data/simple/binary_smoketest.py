"""
TODO: Refactor circleci/cimodel/data/binary_build_data.py to generate this file
       instead of doing one offs here
 Binary builds (subset, to smoke test that they'll work)

 NB: If you modify this file, you need to also modify
 the binary_and_smoke_tests_on_pr variable in
 pytorch-ci-hud to adjust the allowed build list
 at https://github.com/ezyang/pytorch-ci-hud/blob/master/src/BuildHistoryDisplay.js

 Note:
 This binary build is currently broken, see https://github_com/pytorch/pytorch/issues/16710
 - binary_linux_conda_3_6_cu90_devtoolset7_build
 - binary_linux_conda_3_6_cu90_devtoolset7_test

 TODO
 we should test a libtorch cuda build, but they take too long
 - binary_linux_libtorch_3_6m_cu90_devtoolset7_static-without-deps_build
"""

import cimodel.lib.miniutils as miniutils
import cimodel.data.simple.util.branch_filters


class SmoketestJob:
    def __init__(self,
                 template_name,
                 build_env_parts,
                 docker_image,
                 job_name,
                 is_master_only=False,
                 requires=None,
                 has_libtorch_variant=False,
                 extra_props=None):

        self.template_name = template_name
        self.build_env_parts = build_env_parts
        self.docker_image = docker_image
        self.job_name = job_name
        self.is_master_only = is_master_only
        self.requires = requires or []
        self.has_libtorch_variant = has_libtorch_variant
        self.extra_props = extra_props or {}

    def gen_tree(self):

        props_dict = {
            "build_environment": " ".join(self.build_env_parts),
            "name": self.job_name,
            "requires": self.requires,
        }

        if self.docker_image:
            props_dict["docker_image"] = self.docker_image

        if self.is_master_only:
            props_dict["filters"] = cimodel.data.simple.util.branch_filters.gen_filter_dict()

        if self.has_libtorch_variant:
            props_dict["libtorch_variant"] = "shared-with-deps"

        props_dict.update(self.extra_props)

        return [{self.template_name: props_dict}]


WORKFLOW_DATA = [
    SmoketestJob(
        "binary_linux_build",
        ["manywheel", "3.7m", "cu102", "devtoolset7"],
        "pytorch/manylinux-cuda102",
        "binary_linux_manywheel_3_7m_cu102_devtoolset7_build",
        is_master_only=True,
    ),
    SmoketestJob(
        "binary_linux_build",
        ["libtorch", "3.7m", "cpu", "devtoolset7"],
        "pytorch/manylinux-cuda102",
        "binary_linux_libtorch_3_7m_cpu_devtoolset7_shared-with-deps_build",
        is_master_only=True,
        has_libtorch_variant=True,
    ),
    SmoketestJob(
        "binary_linux_build",
        ["libtorch", "3.7m", "cpu", "gcc5.4_cxx11-abi"],
        "pytorch/pytorch-binary-docker-image-ubuntu16.04:latest",
        "binary_linux_libtorch_3_7m_cpu_gcc5_4_cxx11-abi_shared-with-deps_build",
        is_master_only=False,
        has_libtorch_variant=True,
    ),
    SmoketestJob(
        "binary_mac_build",
        ["wheel", "3.7", "cpu"],
        None,
        "binary_macos_wheel_3_7_cpu_build",
        is_master_only=True,
    ),
    # This job has an average run time of 3 hours o.O
    # Now only running this on master to reduce overhead
    SmoketestJob(
        "binary_mac_build",
        ["libtorch", "3.7", "cpu"],
        None,
        "binary_macos_libtorch_3_7_cpu_build",
        is_master_only=True,
    ),
    SmoketestJob(
        "binary_windows_build",
        ["libtorch", "3.7", "cpu", "debug"],
        None,
        "binary_windows_libtorch_3_7_cpu_debug_build",
        is_master_only=True,
    ),
    SmoketestJob(
        "binary_windows_build",
        ["libtorch", "3.7", "cpu", "release"],
        None,
        "binary_windows_libtorch_3_7_cpu_release_build",
        is_master_only=True,
    ),
    SmoketestJob(
        "binary_windows_build",
        ["wheel", "3.7", "cu102"],
        None,
        "binary_windows_wheel_3_7_cu102_build",
        is_master_only=True,
    ),

    SmoketestJob(
        "binary_windows_test",
        ["libtorch", "3.7", "cpu", "debug"],
        None,
        "binary_windows_libtorch_3_7_cpu_debug_test",
        is_master_only=True,
        requires=["binary_windows_libtorch_3_7_cpu_debug_build"],
    ),
    SmoketestJob(
        "binary_windows_test",
        ["libtorch", "3.7", "cpu", "release"],
        None,
        "binary_windows_libtorch_3_7_cpu_release_test",
        is_master_only=False,
        requires=["binary_windows_libtorch_3_7_cpu_release_build"],
    ),
    SmoketestJob(
        "binary_windows_test",
        ["wheel", "3.7", "cu102"],
        None,
        "binary_windows_wheel_3_7_cu102_test",
        is_master_only=True,
        requires=["binary_windows_wheel_3_7_cu102_build"],
        extra_props={
            "executor": "windows-with-nvidia-gpu",
        },
    ),



    SmoketestJob(
        "binary_linux_test",
        ["manywheel", "3.7m", "cu102", "devtoolset7"],
        "pytorch/manylinux-cuda102",
        "binary_linux_manywheel_3_7m_cu102_devtoolset7_test",
        is_master_only=True,
        requires=["binary_linux_manywheel_3_7m_cu102_devtoolset7_build"],
        extra_props={
            "resource_class": "gpu.medium",
            "use_cuda_docker_runtime": miniutils.quote((str(1))),
        },
    ),
    SmoketestJob(
        "binary_linux_test",
        ["libtorch", "3.7m", "cpu", "devtoolset7"],
        "pytorch/manylinux-cuda102",
        "binary_linux_libtorch_3_7m_cpu_devtoolset7_shared-with-deps_test",
        is_master_only=True,
        requires=["binary_linux_libtorch_3_7m_cpu_devtoolset7_shared-with-deps_build"],
        has_libtorch_variant=True,
    ),
    SmoketestJob(
        "binary_linux_test",
        ["libtorch", "3.7m", "cpu", "gcc5.4_cxx11-abi"],
        "pytorch/pytorch-binary-docker-image-ubuntu16.04:latest",
        "binary_linux_libtorch_3_7m_cpu_gcc5_4_cxx11-abi_shared-with-deps_test",
        is_master_only=True,
        requires=["binary_linux_libtorch_3_7m_cpu_gcc5_4_cxx11-abi_shared-with-deps_build"],
        has_libtorch_variant=True,
    ),
]


def get_workflow_jobs():
    return [item.gen_tree() for item in WORKFLOW_DATA]
