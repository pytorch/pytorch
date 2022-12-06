"""
Initializer script that installs stuff to pip.
"""
import argparse
import logging
import os
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
    parser.add_argument(
        "--no-black-binary",
        help="do not use pre-compiled binaries from pip for black.",
        action="store_true",
    )

    args = parser.parse_args()

    logging.basicConfig(
        format="<%(threadName)s:%(levelname)s> %(message)s",
        level=logging.NOTSET if args.verbose else logging.DEBUG,
        stream=sys.stderr,
    )

    pip_args = ["pip3", "install"]

    # If we are in a global install, use `--user` to install so that you do not
    # need root access in order to initialize linters.
    #
    # However, `pip install --user` interacts poorly with virtualenvs (see:
    # https://bit.ly/3vD4kvl) and conda (see: https://bit.ly/3KG7ZfU). So in
    # these cases perform a regular installation.
    in_conda = os.environ.get("CONDA_PREFIX") is not None
    in_virtualenv = os.environ.get("VIRTUAL_ENV") is not None
    if not in_conda and not in_virtualenv:
        pip_args.append("--user")

    pip_args.extend(args.packages)

    for package in args.packages:
        package_name, _, version = package.partition("=")
        if version == "":
            raise RuntimeError(
                "Package {package_name} did not have a version specified. "
                "Please specify a version to produce a consistent linting experience."
            )
        if args.no_black_binary and "black" in package_name:
            pip_args.append(f"--no-binary={package_name}")

    dry_run = args.dry_run == "1"
    if dry_run:
        print(f"Would have run: {pip_args}")
        sys.exit(0)

    run_command(pip_args)
