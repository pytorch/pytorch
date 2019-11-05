#!/usr/bin/env python3

"""
This script is the source of truth for config.yml.
Please see README.md in this directory for details.
"""

import os
import sys
import shutil
from collections import namedtuple, OrderedDict

# import cimodel.data.pytorch_build_definitions as pytorch_build_definitions
# import cimodel.data.binary_build_definitions as binary_build_definitions
# import cimodel.data.caffe2_build_definitions as caffe2_build_definitions
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


class FunctionGen(namedtuple('FunctionGen', 'function depth')):
    __slots__ = ()


class Treegen(FunctionGen):
    """
    Insert the content of a YAML tree into config.yml
    """

    def write(self, output_filehandle):
        build_dict = OrderedDict()
        self.function(build_dict)
        miniyaml.render(output_filehandle, build_dict, self.depth)


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


# Order of this list matters to the generated config.yml.
YAML_SOURCES = [
    File("header-section.yml"),
    File("commands.yml"),
    File("nightly-binary-build-defaults.yml"),
    Header("Build parameters"),
    File("pytorch-build-params.yml"),
    # File("caffe2-build-params.yml"),
    File("binary-build-params.yml"),
    Header("Job specs"),
    File("pytorch-job-specs.yml"),
    # File("caffe2-job-specs.yml"),
    File("binary-job-specs.yml"),
    File("job-specs-setup.yml"),
    # File("job-specs-custom.yml"),
    File("binary_update_htmls.yml"),
    File("binary-build-tests.yml"),
    File("workflows.yml"),
    # Listgen(pytorch_build_definitions.get_workflow_jobs, 3),
    # File("workflows-pytorch-macos-builds.yml"),
    # File("workflows-pytorch-android-gradle-build.yml"),
    # File("workflows-pytorch-ios-builds.yml"),
    # Listgen(caffe2_build_definitions.get_workflow_jobs, 3),
    # File("workflows-binary-builds-smoke-subset.yml"),
    # Header("Daily smoke test trigger"),
    # Treegen(binary_build_definitions.add_binary_smoke_test_jobs, 1),
    # Header("Daily binary build trigger"),
    # Treegen(binary_build_definitions.add_binary_build_jobs, 1),
    File("workflows-nightly-ios-binary-builds.yml"),
    # File("workflows-nightly-android-binary-builds.yml"),
    # Header("Nightly tests"),
    # Listgen(binary_build_definitions.get_nightly_tests, 3),
    # File("workflows-nightly-uploads-header.yml"),
    # Listgen(binary_build_definitions.get_nightly_uploads, 3),
    # File("workflows-s3-html.yml"),
]


def stitch_sources(output_filehandle):
    for f in YAML_SOURCES:
        f.write(output_filehandle)


if __name__ == "__main__":

    stitch_sources(sys.stdout)
