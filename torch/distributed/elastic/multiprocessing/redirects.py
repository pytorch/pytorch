# !/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Taken and modified from original source:
# https://eli.thegreenplace.net/2015/redirecting-all-kinds-of-stdout-in-python/
import ctypes
import logging
import os
import sys
from contextlib import contextmanager
from functools import partial

IS_WINDOWS = sys.platform == "win32"
IS_MACOS = sys.platform == "darwin"


logger = logging.getLogger(__name__)


def get_libc():
    if IS_WINDOWS or IS_MACOS:
        logger.warning(
            "NOTE: Redirects are currently not supported in Windows or MacOs."
        )
        return None
    else:
        return ctypes.CDLL("libc.so.6")


libc = get_libc()


def _c_std(stream: str):
    return ctypes.c_void_p.in_dll(libc, stream)


def _python_std(stream: str):
    return {"stdout": sys.stdout, "stderr": sys.stderr}[stream]


_VALID_STD = {"stdout", "stderr"}


@contextmanager
def redirect(std: str, to_file: str):
    """
    Redirects ``std`` (one of ``"stdout"`` or ``"stderr"``) to a file
    in the path specified by ``to_file``. This method redirects the
    underlying std file descriptor (not just pyton's ``sys.stdout|stderr``).
    See usage for details.

    Directory of ``dst_filename`` is assumed to exist and the destination file
    is overwritten if it already exists.

    .. note:: Due to buffering cross source writes are not guaranteed to
              appear in wall-clock order. For instance in the example below
              it is possible for the C-outputs to appear before the python
              outputs in the log file.

    Usage:

    ::

     # syntactic-sugar for redirect("stdout", "tmp/stdout.log")
     with redirect_stdout("/tmp/stdout.log"):
        print("python stdouts are redirected")
        libc = ctypes.CDLL("libc.so.6")
        libc.printf(b"c stdouts are also redirected"
        os.system("echo system stdouts are also redirected")

     print("stdout restored")

    """

    if std not in _VALID_STD:
        raise ValueError(
            f"unknown standard stream <{std}>, must be one of {_VALID_STD}"
        )

    c_std = _c_std(std)
    python_std = _python_std(std)
    std_fd = python_std.fileno()

    def _redirect(dst):
        libc.fflush(c_std)
        python_std.flush()
        os.dup2(dst.fileno(), std_fd)

    with os.fdopen(os.dup(std_fd)) as orig_std, open(to_file, mode="w+b") as dst:
        _redirect(dst)
        yield
        _redirect(orig_std)


redirect_stdout = partial(redirect, "stdout")
redirect_stderr = partial(redirect, "stderr")
