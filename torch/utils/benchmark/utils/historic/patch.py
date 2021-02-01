"""Utility to patch benchmark utils into earlier versions of PyTorch.

In order to reduce conflicts files are copied to a mirrored location. e.g.
    `from torch.utils_backport.benchmark import Timer`
rather than:
    `from torch.utils.benchmark import Timer`

Caveats:
    1) C++ only works on PyTorch 1.0 and above. The API changes around the major
       version bump are too significant to easily bridge. Python was tested back
       to PyTorch 0.4.
    2) Imports are rewritten to use the modified path.
"""
import functools
import os
import re
import shutil
from typing import Iterable, Sequence, Tuple


TORCH_ROOT = functools.reduce(
    lambda s, _: os.path.split(s)[0], range(5), os.path.abspath(__file__))

BACKPORT_NAME = "utils_backport"
PATTERNS = (
    # Point imports at the new namespace.
    ("torch.utils", f"torch.{BACKPORT_NAME}"),

    (
        "IS_BACK_TESTING_OVERRIDE: bool = False",
        "IS_BACK_TESTING_OVERRIDE: bool = True"
    ),
)


def clean_backport(destination_install: str) -> None:
    destination_root = os.path.join(destination_install, BACKPORT_NAME)
    if os.path.exists(destination_root):
        shutil.rmtree(destination_root)


def backport(destination_install: str) -> None:
    assert os.path.split(TORCH_ROOT)[1] == "torch"
    if os.path.split(destination_install)[1] != "torch":
        raise ValueError(
            f"`{destination_install}` does not appear to be the root of a "
            "PyTorch installation.")

    destination_root = os.path.join(destination_install, BACKPORT_NAME)
    clean_backport(destination_install)
    os.makedirs(destination_root)

    source_root = os.path.join(TORCH_ROOT, "utils")
    for d, _, files in os.walk(os.path.join(source_root, "benchmark")):
        if os.path.split(d)[1] == "__pycache__":
            continue

        for fname in files:
            src = os.path.join(d, fname)
            with open(src, "rt") as f:
                contents = f.read()

            for old, new in PATTERNS:
                contents = re.sub(old, new, contents)

            dest = re.sub(f"^{source_root}", destination_root, src)
            os.makedirs(os.path.split(dest)[0], exist_ok=True)
            with open(dest, "wt") as f:
                f.write(contents)
