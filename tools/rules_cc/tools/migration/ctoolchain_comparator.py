# Copyright 2018 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""A script that compares 2 CToolchains from proto format.

This script accepts two files in either a CROSSTOOL proto text format or a
CToolchain proto text format. It then locates the CToolchains with the given
toolchain_identifier and checks if the resulting CToolchain objects in Java
are the same.

Example usage:

bazel run \
@rules_cc//tools/migration:ctoolchain_comparator -- \
--before=/path/to/CROSSTOOL1 \
--after=/path/to/CROSSTOOL2 \
--toolchain_identifier=id
"""

import os
from absl import app
from absl import flags
from google.protobuf import text_format
from third_party.com.github.bazelbuild.bazel.src.main.protobuf import crosstool_config_pb2
from tools.migration.ctoolchain_comparator_lib import compare_ctoolchains

flags.DEFINE_string(
    "before", None,
    ("A text proto file containing the relevant CTooclchain before the change, "
     "either a CROSSTOOL file or a single CToolchain proto text"))
flags.DEFINE_string(
    "after", None,
    ("A text proto file containing the relevant CToolchain after the change, "
     "either a CROSSTOOL file or a single CToolchain proto text"))
flags.DEFINE_string("toolchain_identifier", None,
                    "The identifier of the CToolchain that is being compared.")
flags.mark_flag_as_required("before")
flags.mark_flag_as_required("after")


def _to_absolute_path(path):
  path = os.path.expanduser(path)
  if os.path.isabs(path):
    return path
  else:
    if "BUILD_WORKING_DIRECTORY" in os.environ:
      return os.path.join(os.environ["BUILD_WORKING_DIRECTORY"], path)
    else:
      return path


def _find_toolchain(crosstool, toolchain_identifier):
  for toolchain in crosstool.toolchain:
    if toolchain.toolchain_identifier == toolchain_identifier:
      return toolchain
  return None


def _read_crosstool_or_ctoolchain_proto(input_file, toolchain_identifier=None):
  """Reads a proto file and finds the CToolchain with the given identifier."""
  with open(input_file, "r") as f:
    text = f.read()
  crosstool_release = crosstool_config_pb2.CrosstoolRelease()
  c_toolchain = crosstool_config_pb2.CToolchain()
  try:
    text_format.Merge(text, crosstool_release)
    if toolchain_identifier is None:
      print("CROSSTOOL proto needs a 'toolchain_identifier' specified in "
            "order to be able to select the right toolchain for comparison.")
      return None
    toolchain = _find_toolchain(crosstool_release, toolchain_identifier)
    if toolchain is None:
      print(("Cannot find a CToolchain with an identifier '%s' in CROSSTOOL "
             "file") % toolchain_identifier)
      return None
    return toolchain
  except text_format.ParseError as crosstool_error:
    try:
      text_format.Merge(text, c_toolchain)
      if (toolchain_identifier is not None and
          c_toolchain.toolchain_identifier != toolchain_identifier):
        print(("Expected CToolchain with identifier '%s', got CToolchain with "
               "identifier '%s'" % (toolchain_identifier,
                                    c_toolchain.toolchain_identifier)))
        return None
      return c_toolchain
    except text_format.ParseError as toolchain_error:
      print(("Error parsing file '%s':" % input_file))  # pylint: disable=superfluous-parens
      print("Attempt to parse it as a CROSSTOOL proto:")  # pylint: disable=superfluous-parens
      print(crosstool_error)  # pylint: disable=superfluous-parens
      print("Attempt to parse it as a CToolchain proto:")  # pylint: disable=superfluous-parens
      print(toolchain_error)  # pylint: disable=superfluous-parens
      return None


def main(unused_argv):

  before_file = _to_absolute_path(flags.FLAGS.before)
  after_file = _to_absolute_path(flags.FLAGS.after)
  toolchain_identifier = flags.FLAGS.toolchain_identifier

  toolchain_before = _read_crosstool_or_ctoolchain_proto(
      before_file, toolchain_identifier)
  toolchain_after = _read_crosstool_or_ctoolchain_proto(after_file,
                                                        toolchain_identifier)

  if not toolchain_before or not toolchain_after:
    print("There was an error getting the required toolchains.")
    exit(1)

  found_difference = compare_ctoolchains(toolchain_before, toolchain_after)
  if found_difference:
    exit(1)


if __name__ == "__main__":
  app.run(main)
