#!/usr/bin/env python3
"""Helper script - Return CI variables such as stable cuda, min python version, etc."""

import argparse
import re
import sys
from typing import Optional


def parse_version(version_string: str) -> Optional[tuple[int, ...]]:
    pattern = r"(\d+)\.(\d+)?"
    match = re.match(pattern, version_string)

    if match:
        return tuple(int(group) for group in match.groups())
    else:
        return None


def main(args: list[str]) -> None:
    import generate_binary_build_matrix

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda-stable-version",
        action="store_true",
        help="get cuda stable version",
    )
    parser.add_argument(
        "--min-python-version",
        action="store_true",
        help="get min supported python version",
    )
    parser.add_argument(
        "--old-python-version",
        action="store_true",
        help="get min supported python version - 0.1",
    )
    options = parser.parse_args(args)
    if options.cuda_stable_version:
        return print(generate_binary_build_matrix.CUDA_STABLE)
    if options.min_python_version:
        return print(generate_binary_build_matrix.FULL_PYTHON_VERSIONS[0])
    if options.old_python_version:
        version = parse_version(generate_binary_build_matrix.FULL_PYTHON_VERSIONS[0])
        if version is not None:
            major, minor = version
            return print(f"{major}.{minor - 1}")


if __name__ == "__main__":
    main(sys.argv[1:])
