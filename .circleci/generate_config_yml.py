#!/usr/bin/env python3

"""
This script is the source of truth for config.yml.
Please see README.md in this directory for details.

In this module,
"""

import os
import sys
from collections import OrderedDict

import cimodel.pytorch_build_definitions as pytorch_build_definitions
import cimodel.binary_build_definitions as binary_build_definitions
import cimodel.caffe2_build_definitions as caffe2_build_definitions
import cimodel.miniyaml as miniyaml


class File(object):
    """
    Verbatim copy the contents of a file into config.yml
    """
    def __init__(self, filename):
        self.filename = filename

    def write(self, output_filehandle):
        with open(os.path.join("verbatim-sources", self.filename)) as fh:
            output_filehandle.write(fh.read())


class Treegen(object):
    """
    Insert the content of a YAML tree into config.yml
    """
    def __init__(self, function, depth):
        self.function = function
        self.depth = depth

    def write(self, output_filehandle):
        build_dict = OrderedDict()
        self.function(build_dict)
        miniyaml.render(output_filehandle, build_dict, self.depth)


class Listgen(object):
    """
    Insert the content of a YAML list into config.yml
    """
    def __init__(self, function, depth):
        self.function = function
        self.depth = depth

    def write(self, output_filehandle):
        miniyaml.render(output_filehandle, self.function(), self.depth)


# Order of this list matters to the generated config.yml.
YAML_SOURCES = [
    File("header-section.yml"),
    File("linux-build-defaults.yml"),
    File("macos-build-defaults.yml"),
    File("nightly-binary-build-defaults.yml"),
    File("linux-binary-build-defaults.yml"),
    File("macos-binary-build-defaults.yml"),
    File("nightly-build-smoke-tests-defaults.yml"),
    File("job-specs-header.yml"),
    Treegen(pytorch_build_definitions.add_build_env_defs, 0),
    File("job-specs-custom.yml"),
    Treegen(caffe2_build_definitions.add_caffe2_builds, 1),
    File("job-specs-html-update.yml"),
    File("binary-build-specs-header.yml"),
    Treegen(binary_build_definitions.add_binary_build_specs, 1),
    File("binary-build-tests-header.yml"),
    Treegen(binary_build_definitions.add_binary_build_tests, 1),
    File("binary-build-tests.yml"),
    File("binary-build-uploads-header.yml"),
    Treegen(binary_build_definitions.add_binary_build_uploads, 1),
    File("smoke-test-specs-header.yml"),
    Treegen(binary_build_definitions.add_smoke_test_specs, 1),
    File("workflows.yml"),
    Listgen(pytorch_build_definitions.get_workflow_list, 3),
    File("workflows-pytorch-macos-builds.yml"),
    Listgen(caffe2_build_definitions.get_caffe2_workflows, 3),
    File("workflows-binary-builds-smoke-subset.yml"),
    File("workflows-binary-smoke-header.yml"),
    Treegen(binary_build_definitions.add_binary_smoke_test_jobs, 1),
    File("workflows-binary-build-header.yml"),
    Treegen(binary_build_definitions.add_binary_build_jobs, 1),
    File("workflows-nightly-tests-header.yml"),
    Listgen(binary_build_definitions.get_nightly_tests, 3),
    File("workflows-nightly-uploads-header.yml"),
    Listgen(binary_build_definitions.get_nightly_uploads, 3),
    File("workflows-s3-html.yml"),
]


def stitch_sources(output_filehandle):
    for f in YAML_SOURCES:
        f.write(output_filehandle)


if __name__ == "__main__":

    stitch_sources(sys.stdout)
