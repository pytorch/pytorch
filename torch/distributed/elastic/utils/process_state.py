#!/usr/bin/env python3
# mypy: allow-untyped-defs

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Helpers for inspecting Linux process state from ``/proc``.

These are intentionally agent-agnostic so they can be reused outside of
``torch.distributed.elastic.agent``.
"""

from __future__ import annotations


__all__ = ["read_proc_state", "is_uninterruptible_state"]


def read_proc_state(pid: int) -> str | None:
    """Read the process state char from ``/proc/<pid>/stat``. Linux-only.

    Returns the single-letter state (e.g. ``'R'``, ``'S'``, ``'D'``,
    ``'Z'``) or ``None`` if the file is unreadable (non-Linux, process
    gone, permission denied, or parse error). The ``comm`` field can
    contain spaces and parentheses, so the parser keys off the last
    ``)`` rather than splitting on whitespace.
    """
    try:
        with open(f"/proc/{pid}/stat") as f:
            data = f.read()
        rparen = data.rfind(")")
        if rparen == -1:
            return None
        # Fields after comm are space-separated; state is field 3 overall
        # i.e. the first token after the closing ')'.
        rest = data[rparen + 1 :].split()
        if not rest:
            return None
        return rest[0]
    except OSError:
        return None


def is_uninterruptible_state(state: str | None) -> bool:
    """Return True if ``state`` is Linux uninterruptible sleep ('D').

    Includes the ``'D+'`` (foreground) variant; the state field is
    a single char in the kernel format we read, but we accept any
    string that starts with ``'D'`` defensively.
    """
    return state is not None and state.startswith("D")
