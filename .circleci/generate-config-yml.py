#!/usr/bin/env python3

"""
This script is the source of truth for config.yml.
Please make changes here only, then re-run this
script and commit the result.
"""

import os
import sys
from collections import OrderedDict

import build_env_definitions
import binary_build_definitions
import miniyaml


class File:
    def __init__(self, filename):
        self.filename = filename

    def write(self, output_filehandle):
        with open(os.path.join("verbatim-sources", self.filename)) as fh:
            output_filehandle.write(fh.read())


class Treegen:
    def __init__(self, function, depth):
        self.function = function
        self.depth = depth

    def write(self, output_filehandle):
        build_dict = OrderedDict()
        self.function(build_dict)
        miniyaml.render(output_filehandle, None, build_dict, self.depth)


YAML_SOURCES = [
    File("header-section.yml"),
    File("linux-build-defaults.yml"),
    File("macos-build-defaults.yml"),
    File("nightly-binary-build-defaults.yml"),
    File("linux-binary-build-defaults.yml"),
    File("macos-binary-build-defaults.yml"),
    File("nightly-build-smoke-tests-defaults.yml"),
    File("job-specs-header.yml"),
    Treegen(build_env_definitions.add_build_env_defs, 0),
    File("job-specs-custom.yml"),
    File("job-specs-caffe2-builds.yml"),
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
    File("workflows-pytorch-linux-builds.yml"),
    File("workflows-pytorch-macos-builds.yml"),
    File("workflows-caffe2-builds.yml"),
    File("workflows-caffe2-macos-builds.yml"),
    File("workflows-binary-builds-smoke-subset.yml"),
    File("workflows-binary-smoke-header.yml"),
    Treegen(binary_build_definitions.add_binary_smoke_test_jobs, 1),
    File("workflows-binary-build-header.yml"),
    Treegen(binary_build_definitions.add_binary_build_jobs, 1),
    File("workflows-nightly-tests-header.yml"),
    File("workflows-nightly-tests.yml"),
    File("workflows-nightly-uploads-header.yml"),
    File("workflows-nightly-uploads.yml"),
    File("workflows-s3-html.yml"),
]


def stitch_sources(output_filehandle):
    for f in YAML_SOURCES:
        f.write(output_filehandle)


if __name__ == "__main__":

    stitch_sources(sys.stdout)
