"""Script to build split pytorch wheels

What is split build / why is it important?
    > Split build is splitting the PyTorch build into a libtorch &
    > PyTorch python frontend package. This allows us to to publish
    > both as separate packages and opens up our ability to have users
    > install different libtorch backends per their PyTorch frontend
    >
    > Example: opening up the door to things like:
    >     pip install torch[cuda]
    >     pip install torch[rocm]
    >     pip install torch[cpu]
    >     etc.

Why does this exist?
    > Currently our split build requires you to invoke setup.py twice
    > Which ends up complicating the build process and adds some level
    > of complexity to our setup.py / build invocation for split builds.
    > Ideally this script will eventually not be needed but for
    > development purposes we should have an easy way to invoke this script
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# NOTE: This will need to be updated if this script is ever moved
ROOT_PATH = Path(__file__).absolute().parents[2]
SETUP_PY_PATH = ROOT_PATH / "setup.py"


def requirements_installed() -> bool:
    try:
        import setuptools  # type: ignore[import-untyped]  # noqa: F401

        return True
    except ImportError:
        logger.error(
            "Requirements not installed, run the following command to install:"
        )
        logger.error(
            "    > %s -m pip install -r %s/requirements.txt", sys.executable, ROOT_PATH
        )
        return False


def setup_py(cmd_args: list[str], extra_env: Optional[dict[str, str]] = None) -> None:
    if extra_env is None:
        extra_env = {}
    cmd = [sys.executable, str(SETUP_PY_PATH), *cmd_args]
    logger.debug("+ %s", " ".join(cmd))
    subprocess.run(
        cmd,
        # Give the parent environment to the subprocess
        env={**os.environ, **extra_env},
        check=True,
    )


def split_build(cmd: str) -> None:
    logger.info("Running %s for libtorch wheel", cmd)
    setup_py(
        [cmd],
        extra_env={"BUILD_LIBTORCH_WHL": "1", "BUILD_PYTHON_ONLY": "0"},
    )
    logger.info("Running %s for torch wheel", cmd)
    # NOTE: Passing --cmake is necessary here since the torch frontend has it's
    # own cmake files that it needs to generate
    setup_py(
        [cmd, "--cmake"],
        extra_env={"BUILD_LIBTORCH_WHL": "0", "BUILD_PYTHON_ONLY": "1"},
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    command_subparser = parser.add_subparsers(dest="command")
    # Ideally these should mirror setuptools commands if we need support here for that
    command_subparser.add_parser("install")
    command_subparser.add_parser("bdist_wheel")
    command_subparser.add_parser("develop")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not requirements_installed():
        sys.exit(1)
    split_build(args.command)


if __name__ == "__main__":
    main()
