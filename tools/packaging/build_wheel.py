#!/usr/bin/env python3

import argparse
import contextlib
import logging
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Dict, List


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

ROOT_PATH = Path(__file__).absolute().parent.parent.parent
SETUP_PY_PATH = ROOT_PATH / "setup.py"
REQUIREMENTS_PATH = ROOT_PATH / "requirements.txt"


def run_cmd(
    cmd: List[str], capture_output: bool = False
) -> subprocess.CompletedProcess[bytes]:
    logger.debug("Running command: %s", " ".join(cmd))
    return subprocess.run(
        cmd,
        # Give the parent environment to the subprocess
        env={**os.environ},
        capture_output=capture_output,
        check=True,
    )


def interpreter_version(interpreter: str) -> str:
    version_string = (
        run_cmd([interpreter, "--version"], capture_output=True)
        .stdout.decode("utf-8")
        .strip()
    )
    return str(version_string.split(" ")[1])


@contextlib.contextmanager
def venv(interpreter: str) -> Iterator[str]:
    # Should this use EnvBuilder? Probably, maybe a good todo in the future
    python_version = interpreter_version(interpreter)
    with tempfile.TemporaryDirectory(
        suffix=f"_pytorch_builder_{python_version}"
    ) as tmp_dir:
        logger.info(
            "Creating virtual environment (Python %s) at %s",
            python_version,
            tmp_dir,
        )
        run_cmd([interpreter, "-m", "venv", tmp_dir])
        yield str(Path(tmp_dir) / "bin" / "python3")


class Builder:
    # The python interpeter that we should be using
    interpreter: str

    def __init__(self, interpreter: str) -> None:
        self.interpreter = interpreter

    def setup_py(self, cmd_args: List[str]) -> bool:
        return (
            run_cmd([self.interpreter, str(SETUP_PY_PATH), *cmd_args]).returncode == 0
        )

    def bdist_wheel(self, destination: str) -> bool:
        logger.info("Running bdist_wheel -d %s", destination)
        return self.setup_py(["bdist_wheel", "-d", destination])

    def clean(self) -> bool:
        logger.info("Running clean")
        return self.setup_py(["clean"])

    def install_requirements(self) -> None:
        logger.info("Installing requirements")
        run_cmd(
            [self.interpreter, "-m", "pip", "install", "-r", str(REQUIREMENTS_PATH)]
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--python",
        action="append",
        type=str,
        help=(
            "Python interpreters to build packages for, can be set multiple times,"
            " should ideally be full paths, (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "-d",
        "--destination",
        default="dist/",
        type=str,
        help=("Destination to put the compailed binaries" ""),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pythons = args.python or [sys.executable]
    build_times: Dict[str, float] = dict()

    if len(pythons) > 1 and args.destination == "dist/":
        logger.warning(
            "dest is 'dist/' while multiple python versions specified, output will be overwritten"
        )

    for interpreter in pythons:
        with venv(interpreter) as venv_interpreter:
            builder = Builder(venv_interpreter)
            # clean actually requires setuptools so we need to ensure we
            # install requriements before
            builder.install_requirements()
            builder.clean()

            start_time = time.time()

            builder.bdist_wheel(args.destination)

            end_time = time.time()

            build_times[interpreter_version(venv_interpreter)] = end_time - start_time
    for version, build_time in build_times.items():
        logger.info("Build time (%s): %fs", version, build_time)


if __name__ == "__main__":
    main()
