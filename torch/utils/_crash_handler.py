import os
import sys
import pathlib

import torch

DEFAULT_MINIDUMP_DIR = "/tmp/pytorch_crashes"

def enable_minidumps(directory=DEFAULT_MINIDUMP_DIR):
    if sys.platform != "linux":
        raise RuntimeError("Minidump collection is currently only implemented for Linux platforms")

    if directory == DEFAULT_MINIDUMP_DIR:
        pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
    elif not os.path.exists(directory):
        raise RuntimeError(f"Directory does not exist: {directory}")

    torch._C._enable_minidumps(directory)


def enable_minidumps_on_exceptions():
    torch._C._enable_minidumps_on_exceptions()


def disable_minidumps():
    torch._C._disable_minidumps()
