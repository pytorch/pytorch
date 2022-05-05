"""
Initializer script that installs stuff to pip.
"""
import argparse
import logging
import subprocess
import sys
import time

from typing import List


def run_command(args: List[str]) -> "subprocess.CompletedProcess[bytes]":
    logging.debug("$ %s", " ".join(args))
    start_time = time.monotonic()
    try:
        return subprocess.run(args, check=True)
    finally:
        end_time = time.monotonic()
        logging.debug("took %dms", (end_time - start_time) * 1000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="pip initializer")
    parser.add_argument(
        "packages",
        nargs="+",
        help="pip packages to install",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="verbose logging",
    )
    parser.add_argument(
        "--dry-run", help="do not install anything, just print what would be done."
    )

    args = parser.parse_args()

    logging.basicConfig(
        format="<%(threadName)s:%(levelname)s> %(message)s",
        level=logging.NOTSET if args.verbose else logging.DEBUG,
        stream=sys.stderr,
    )

    for package in args.packages:
        package_name, _, version = package.partition("=")
        if version == "":
            raise RuntimeError(
                "Package {package_name} did not have a version specified. "
                "Please specify a version to product a consistent linting experience."
            )
    pip_args = ["pip3", "install"]
    pip_args.extend(args.packages)

    dry_run = args.dry_run == "1"
    if dry_run:
        print(f"Would have run: {pip_args}")
        sys.exit(0)

    run_command(pip_args)
