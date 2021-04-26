#!/usr/bin/env python3

"""
This script is the source of truth for config.yml.
Please see README.md in this directory for details.
"""

import os
import shutil
import sys
from collections import namedtuple

import cimodel.data.binary_build_definitions as binary_build_definitions
import cimodel.data.pytorch_build_definitions as pytorch_build_definitions
import cimodel.data.simple.android_definitions
import cimodel.data.simple.bazel_definitions
import cimodel.data.simple.binary_smoketest
import cimodel.data.simple.docker_definitions
import cimodel.data.simple.ge_config_tests
import cimodel.data.simple.ios_definitions
import cimodel.data.simple.macos_definitions
import cimodel.data.simple.mobile_definitions
import cimodel.data.simple.nightly_android
import cimodel.data.simple.nightly_ios
import cimodel.data.simple.anaconda_prune_defintions
import cimodel.data.windows_build_definitions as windows_build_definitions
import cimodel.lib.miniutils as miniutils
import cimodel.lib.miniyaml as miniyaml


class File(object):
    """
    Verbatim copy the contents of a file into config.yml
    """

    def __init__(self, filename):
        self.filename = filename

    def write(self, output_filehandle):
        with open(os.path.join("verbatim-sources", self.filename)) as fh:
            shutil.copyfileobj(fh, output_filehandle)


class FunctionGen(namedtuple("FunctionGen", "function depth")):
    __slots__ = ()


class Treegen(FunctionGen):
    """
    Insert the content of a YAML tree into config.yml
    """

    def write(self, output_filehandle):
        miniyaml.render(output_filehandle, self.function(), self.depth)


class Listgen(FunctionGen):
    """
    Insert the content of a YAML list into config.yml
    """

    def write(self, output_filehandle):
        miniyaml.render(output_filehandle, self.function(), self.depth)


def horizontal_rule():
    return "".join("#" * 78)


class Header(object):
    def __init__(self, title, summary=None):
        self.title = title
        self.summary_lines = summary or []

    def write(self, output_filehandle):
        text_lines = [self.title] + self.summary_lines
        comment_lines = ["# " + x for x in text_lines]
        lines = miniutils.sandwich([horizontal_rule()], comment_lines)

        for line in filter(None, lines):
            output_filehandle.write(line + "\n")


def gen_build_workflows_tree():
    build_workflows_functions = [
        cimodel.data.simple.docker_definitions.get_workflow_jobs,
        pytorch_build_definitions.get_workflow_jobs,
        cimodel.data.simple.macos_definitions.get_workflow_jobs,
        cimodel.data.simple.android_definitions.get_workflow_jobs,
        cimodel.data.simple.ios_definitions.get_workflow_jobs,
        cimodel.data.simple.mobile_definitions.get_workflow_jobs,
        cimodel.data.simple.ge_config_tests.get_workflow_jobs,
        cimodel.data.simple.bazel_definitions.get_workflow_jobs,
        cimodel.data.simple.binary_smoketest.get_workflow_jobs,
        cimodel.data.simple.nightly_ios.get_workflow_jobs,
        cimodel.data.simple.nightly_android.get_workflow_jobs,
        cimodel.data.simple.anaconda_prune_defintions.get_workflow_jobs,
        windows_build_definitions.get_windows_workflows,
        binary_build_definitions.get_post_upload_jobs,
        binary_build_definitions.get_binary_smoke_test_jobs,
    ]

    binary_build_functions = [
        binary_build_definitions.get_binary_build_jobs,
        binary_build_definitions.get_nightly_tests,
        binary_build_definitions.get_nightly_uploads,
    ]

    return {
        "workflows": {
            "binary_builds": {
                "when": r"<< pipeline.parameters.run_binary_tests >>",
                "jobs": [f() for f in binary_build_functions],
            },
            "build": {
                "when": r"<< pipeline.parameters.run_build >>",
                "jobs": [f() for f in build_workflows_functions]
            },
        }
    }


# Order of this list matters to the generated config.yml.
YAML_SOURCES = [
    File("header-section.yml"),
    File("commands.yml"),
    File("nightly-binary-build-defaults.yml"),
    Header("Build parameters"),
    File("build-parameters/pytorch-build-params.yml"),
    File("build-parameters/binary-build-params.yml"),
    File("build-parameters/promote-build-params.yml"),
    Header("Job specs"),
    File("job-specs/pytorch-job-specs.yml"),
    File("job-specs/binary-job-specs.yml"),
    File("job-specs/job-specs-custom.yml"),
    File("job-specs/job-specs-promote.yml"),
    File("job-specs/binary_update_htmls.yml"),
    File("job-specs/binary-build-tests.yml"),
    File("job-specs/docker_jobs.yml"),
    Header("Workflows"),
    Treegen(gen_build_workflows_tree, 0),
    File("workflows/workflows-scheduled-ci.yml"),
    File("workflows/workflows-ecr-gc.yml"),
    File("workflows/workflows-promote.yml"),
]


def stitch_sources(output_filehandle):
    for f in YAML_SOURCES:
        f.write(output_filehandle)


if __name__ == "__main__":

    stitch_sources(sys.stdout)
