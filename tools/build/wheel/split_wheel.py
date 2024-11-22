import argparse
import os
import sys
import subprocess

import logging
from pathlib import Path
from typing import List, Dict, Optional

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# TODO: Add a function to install dependencies from requirements.txt
# TODO: Add a function to do submodule setup


def get_git_root() -> Path:
    """
    Get the current root directory of a Git repository.
    Returns:
        str: The path to the Git repository root.
    """
    try:
        # Run the 'git rev-parse --show-toplevel' command
        git_root = (
            subprocess.check_output(["git", "rev-parse", "--show-toplevel"])
            .decode("utf-8")
            .strip()
        )
        return Path(git_root)
    except subprocess.CalledProcessError:
        # If the command fails, it means we're not in a Git repository
        logger.warning("No git repository detected, assuming you are at the root")
        return Path(".")


def get_setup_py() -> Path:
    return get_git_root() / Path("setup.py")


def setup_py(cmd_args: List[str], extra_env: Optional[Dict[str, str]] = None):
    if extra_env is None:
        extra_env = {}
    cmd = [sys.executable, str(get_setup_py()), *cmd_args]
    logger.debug("+ %s", " ".join(cmd))
    subprocess.run(
        cmd,
        # Give the parent environment to the subprocess
        env={**os.environ, **extra_env},
        check=True,
    )


def bdist_wheel():
    # Equivalent to running the following:
    #     > BUILD_LIBTORCH_WHL=1 BUILD_PYTHON_ONLY=0 python setup.py bdist_wheel
    logger.info("Building libtorch wheel")
    setup_py(
        ["bdist_wheel"],
        extra_env={"BUILD_LIBTORCH_WHL": "1", "BUILD_PYTHON_ONLY": "0"},
    )
    # Equivalent to running the following:
    #     > BUILD_LIBTORCH_WHL=0 BUILD_PYTHON_ONLY=1 python setup.py bdist_wheel --cmake
    logger.info("Building torch wheel")
    setup_py(
        ["bdist_wheel", "--cmake"],
        extra_env={"BUILD_LIBTORCH_WHL": "0", "BUILD_PYTHON_ONLY": "1"},
    )


def install():
    # Equivalent to running the following:
    #     > BUILD_LIBTORCH_WHL=1 BUILD_PYTHON_ONLY=0 python setup.py install
    logger.info("Building libtorch wheel")
    setup_py(
        ["install"],
        extra_env={"BUILD_LIBTORCH_WHL": "1", "BUILD_PYTHON_ONLY": "0"},
    )
    # Equivalent to running the following:
    #     > BUILD_LIBTORCH_WHL=0 BUILD_PYTHON_ONLY=1 python setup.py install --cmake
    logger.info("Building torch wheel")
    setup_py(
        ["install", "--cmake"],
        extra_env={"BUILD_LIBTORCH_WHL": "0", "BUILD_PYTHON_ONLY": "1"},
    )


def develop():
    # Equivalent to running the following:
    #     > BUILD_LIBTORCH_WHL=1 BUILD_PYTHON_ONLY=0 python setup.py develop
    logger.info("Building libtorch wheel")
    setup_py(
        ["develop"],
        extra_env={"BUILD_LIBTORCH_WHL": "1", "BUILD_PYTHON_ONLY": "0"},
    )
    # Equivalent to running the following:
    #     > BUILD_LIBTORCH_WHL=0 BUILD_PYTHON_ONLY=1 python setup.py develop --cmake
    logger.info("Building torch wheel")
    setup_py(
        ["develop", "--cmake"],
        extra_env={"BUILD_LIBTORCH_WHL": "0", "BUILD_PYTHON_ONLY": "1"},
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    command_subparser = parser.add_subparsers(dest="command")
    command_subparser.add_parser("install")
    command_subparser.add_parser("bdist_wheel")
    command_subparser.add_parser("develop")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "bdist_wheel":
        bdist_wheel()
    elif args.command == "install":
        install()
    elif args.command == "develop":
        develop()


if __name__ == "__main__":
    main()
