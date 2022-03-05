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



]


def get_workflow_jobs():
    return [item.gen_tree() for item in WORKFLOW_DATA]
