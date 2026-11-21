"""Shared CLI helpers for the ROCm CI debug-env scripts."""

from __future__ import annotations

import sys


class CliError(Exception):
    """Expected failure with a user-facing message."""


def cli_fail(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 1
