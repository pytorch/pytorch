#!/usr/bin/env python3
"""Helper script - Return CI variables such as stable cuda, min python version, etc."""

import argparse
import sys


def main(args: list[str]) -> None:
    import generate_binary_build_matrix

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda-stable-version",
        action="store_true",
        help="get cuda stable version",
    )
    parser.add_argument(
        "--cuda-aarch64-version",
        action="store_true",
        help="get cuda aarch64 version",
    )
    parser.add_argument(
        "--min-python-version",
        action="store_true",
        help="get min supported python version",
    )
    options = parser.parse_args(args)
    if options.cuda_stable_version:
        return print(generate_binary_build_matrix.CUDA_STABLE)
    if options.cuda_aarch64_version:
        return print(generate_binary_build_matrix.CUDA_AARCH64_ARCHES[0].removesuffix("-aarch64"))
    if options.min_python_version:
        return print(generate_binary_build_matrix.FULL_PYTHON_VERSIONS[0])


if __name__ == "__main__":
    main(sys.argv[1:])
