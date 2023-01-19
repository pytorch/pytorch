"""Script migrating legacy CROSSTOOL fields into features.

This script migrates the CROSSTOOL to use only the features to describe C++
command lines. It is intended to be added as a last step of CROSSTOOL generation
pipeline. Since it doesn't retain comments, we assume CROSSTOOL owners will want
to migrate their pipeline manually.
"""

# Tracking issue: https://github.com/bazelbuild/bazel/issues/5187
#
# Since C++ rules team is working on migrating CROSSTOOL from text proto into
# Starlark, we advise CROSSTOOL owners to wait for the CROSSTOOL -> Starlark
# migrator before they invest too much time into fixing their pipeline. Tracking
# issue for the Starlark effort is
# https://github.com/bazelbuild/bazel/issues/5380.

from absl import app
from absl import flags
from google.protobuf import text_format
from third_party.com.github.bazelbuild.bazel.src.main.protobuf import crosstool_config_pb2
from tools.migration.legacy_fields_migration_lib import migrate_legacy_fields
import os

flags.DEFINE_string("input", None, "Input CROSSTOOL file to be migrated")
flags.DEFINE_string("output", None,
                    "Output path where to write migrated CROSSTOOL.")
flags.DEFINE_boolean("inline", None, "Overwrite --input file")


def main(unused_argv):
  crosstool = crosstool_config_pb2.CrosstoolRelease()

  input_filename = flags.FLAGS.input
  output_filename = flags.FLAGS.output
  inline = flags.FLAGS.inline

  if not input_filename:
    raise app.UsageError("ERROR --input unspecified")
  if not output_filename and not inline:
    raise app.UsageError("ERROR --output unspecified and --inline not passed")
  if output_filename and inline:
    raise app.UsageError("ERROR both --output and --inline passed")

  with open(to_absolute_path(input_filename), "r") as f:
    input_text = f.read()

  text_format.Merge(input_text, crosstool)

  migrate_legacy_fields(crosstool)
  output_text = text_format.MessageToString(crosstool)

  resolved_output_filename = to_absolute_path(
      input_filename if inline else output_filename)
  with open(resolved_output_filename, "w") as f:
    f.write(output_text)

def to_absolute_path(path):
  path = os.path.expanduser(path)
  if os.path.isabs(path):
    return path
  else:
    if "BUILD_WORKING_DIRECTORY" in os.environ:
      return os.path.join(os.environ["BUILD_WORKING_DIRECTORY"], path)
    else:
      return path


if __name__ == "__main__":
  app.run(main)
