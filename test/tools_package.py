import pathlib
import sys
import tempfile


# The tools/ dir is at the top-level of the pytorch project. We are
# currently in the test/ subdirectory, so we have to go up two levels
# to find it.
TOOLS_DIR = pathlib.Path(__file__).parent.parent / 'tools/'
assert TOOLS_DIR.is_dir()


# We don't want to add TOOLS_DIR to sys.path, otherwise imports will
# need to look like:
# import codegen
#  instead of
# import tools.codegen
#
# We don't want to add TOOLS_DIR.parent to sys.path, otherwise we have
# given another way to import torch: we would be asking for trouble.
#
# Instead, make a temporary directory to add to sys.path. Inside that
# temporary directory, symlink to tools/.


# This will get deleted when the module is destroyed.
temp_dir = tempfile.TemporaryDirectory()
(pathlib.Path(temp_dir.name) / 'tools/').symlink_to(TOOLS_DIR)


# Note that this is not robust if the module is reloaded. If we want
# to support that, we'll need a handle to an object that can remove
# itself from sys.path upon its destruction.
sys.path.append(temp_dir.name)
