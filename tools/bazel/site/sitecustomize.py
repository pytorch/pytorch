"""Runs at the start of every Python process of a test: site imports sitecustomize.

Puts the torch packages on the path, where pip would have installed them. They
are not in the test program's runfiles (see torch_tree.bzl), but next to them in
Bazel's output tree.

Also keeps the record that lets a test's action tell Bazel which Python files no
process of it opened (see run_test.py).
"""

import os
import sys


# <execution root>/bazel-out/<configuration>/bin/<package>/<name>.runfiles/...
runfiles = os.path.realpath(__file__.partition(".runfiles/")[0] + ".runfiles")
root, _, configuration = runfiles.partition("/bazel-out/")
# The directory of //:torch_tree.
sys.path.append(f"{root}/bazel-out/{configuration.partition('/')[0]}/bin/torch_tree")
# What a test imports comes with bytecode. Python would write the rest next to
# its source: into a sandbox, to be thrown away, or into Bazel's outputs.
sys.dont_write_bytecode = True

# run_test.py creates this file for an action that reports what it did not open.
# The place does not depend on the environment, which a test may not pass on to
# the processes it starts.
RECORD = f"{root}/torch_test.opened"

if os.path.exists(RECORD):
    log = os.open(RECORD, os.O_WRONLY | os.O_APPEND)
    seen: set[str] = set()

    def opened(event: str, args: tuple) -> None:
        if event != "open":
            return
        try:
            path = os.fsdecode(args[0])
        except TypeError:
            # Opened by file descriptor.
            return
        if path.endswith((".py", ".pyc")) and path not in seen:
            seen.add(path)
            os.write(log, f"{os.path.abspath(path)}\n".encode())

    sys.addaudithook(opened)
