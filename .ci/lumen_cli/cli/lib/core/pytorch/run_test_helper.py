"""Thin wrappers around common test invocation patterns."""

from __future__ import annotations

import os
import sys

from cli.lib.common.utils import run_command


# Flags stripped from run_test.py invocations when LUMEN_NO_UPLOAD=1.
_UPLOAD_FLAGS = {"--upload-artifacts-while-running"}


def run_test(*args: str) -> None:
    """Invoke python test/run_test.py with the given arguments."""
    if os.environ.get("LUMEN_NO_UPLOAD"):
        args = tuple(a for a in args if a not in _UPLOAD_FLAGS)
    cmd = f"{sys.executable} test/run_test.py " + " ".join(args)
    run_command(cmd)


def run_command_checked(cmd: str) -> None:
    """Run an arbitrary shell command, raising on failure."""
    run_command(cmd, use_shell=True)
