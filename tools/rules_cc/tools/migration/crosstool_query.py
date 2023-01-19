"""Script to make automated CROSSTOOL refactorings easier.

This script reads the CROSSTOOL file and allows for querying of its fields.
"""

from absl import app
from absl import flags
from google.protobuf import text_format
from third_party.com.github.bazelbuild.bazel.src.main.protobuf import crosstool_config_pb2

flags.DEFINE_string("crosstool", None, "CROSSTOOL file path to be queried")
flags.DEFINE_string("identifier", None,
                    "Toolchain identifier to specify toolchain.")
flags.DEFINE_string("print_field", None, "Field to be printed to stdout.")


def main(unused_argv):
  crosstool = crosstool_config_pb2.CrosstoolRelease()

  crosstool_filename = flags.FLAGS.crosstool
  identifier = flags.FLAGS.identifier
  print_field = flags.FLAGS.print_field

  if not crosstool_filename:
    raise app.UsageError("ERROR crosstool unspecified")
  if not identifier:
    raise app.UsageError("ERROR identifier unspecified")

  if not print_field:
    raise app.UsageError("ERROR print_field unspecified")

  with open(crosstool_filename, "r") as f:
    text = f.read()
    text_format.Merge(text, crosstool)

  toolchain_found = False
  for toolchain in crosstool.toolchain:
    if toolchain.toolchain_identifier == identifier:
      toolchain_found = True
      if not print_field:
        continue
      for field, value in toolchain.ListFields():
        if print_field == field.name:
          print value

  if not toolchain_found:
    print "toolchain_identifier %s not found, valid values are:" % identifier
    for toolchain in crosstool.toolchain:
      print "  " + toolchain.toolchain_identifier


if __name__ == "__main__":
  app.run(main)
