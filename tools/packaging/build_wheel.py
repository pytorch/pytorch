#!/usr/bin/env python3

import argparse
import contextlib
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ROOT_PATH = Path(__file__).absolute().parent.parent.parent
REQUIREMENTS_PATH = ROOT_PATH / "requirements.txt"
PYPROJECT_TOML_PATH = ROOT_PATH / "pyproject.toml"


def run_cmd(
    cmd: list[str], capture_output: bool = False
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


def get_supported_python_versions() -> list[str]:
    """Extract supported Python versions from pyproject.toml classifiers."""
    with open(PYPROJECT_TOML_PATH) as f:
        content = f.read()

    # Find Python version classifiers
    pattern = r'"Programming Language :: Python :: (\d+\.\d+)"'
    matches = re.findall(pattern, content)

    # Sort versions and return them
    return sorted(matches, key=lambda x: tuple(map(int, x.split("."))))


def find_python_interpreters(mode: str) -> list[str]:
    """Find Python interpreters based on the specified mode."""
    if mode == "manylinux":
        return _find_manylinux_interpreters()
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def _find_manylinux_interpreters() -> list[str]:
    """Find Python interpreters in manylinux format (/opt/python/)."""
    supported_versions = get_supported_python_versions()
    interpreters = []

    python_root = Path("/opt/python")
    if not python_root.exists():
        logger.warning("Path /opt/python does not exist, no interpreters found")
        return []

    # Find all python3 binaries in /opt/python/
    python_binaries = list(python_root.glob("*/bin/python3"))

    for python_path in python_binaries:
        try:
            # Check if it's PyPy (skip it)
            version_output = run_cmd(
                [str(python_path), "--version"], capture_output=True
            )
            version_string = version_output.stdout.decode("utf-8").strip()

            if "PyPy" in version_string:
                logger.debug("Skipping PyPy interpreter: %s", python_path)
                continue

            # Extract Python version (e.g., "Python 3.9.1" -> "3.9")
            match = re.search(r"Python (\d+\.\d+)", version_string)
            if not match:
                logger.debug("Could not parse version from: %s", version_string)
                continue

            python_version = match.group(1)

            # Check if this version is supported
            if python_version in supported_versions:
                interpreters.append(str(python_path))
                logger.debug(
                    "Found supported Python %s at %s", python_version, python_path
                )
            else:
                logger.debug(
                    "Python %s not in supported versions: %s",
                    python_version,
                    supported_versions,
                )

        except subprocess.CalledProcessError as e:
            logger.debug("Failed to get version for %s: %s", python_path, e)
            continue
    return interpreters


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
    # The python interpreter that we should be using
    interpreter: str

    def __init__(self, interpreter: str) -> None:
        self.interpreter = interpreter

    def build_wheel(self, destination: str) -> bool:
        logger.info("Running bdist_wheel -d %s", destination)
        return (
            run_cmd(
                [
                    self.interpreter,
                    "-m",
                    "build",
                    "--wheel",
                    "--no-isolation",
                    "--outdir",
                    destination,
                    str(ROOT_PATH),
                ]
            ).returncode
            == 0
        )

    def clean(self) -> bool:
        logger.info("Running clean")
        return run_cmd([self.interpreter, "setup.py", "clean"]).returncode == 0

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
        "--find-python",
        type=str,
        choices=["manylinux"],
        help=(
            "Automatically find Python interpreters based on the specified mode. "
            "Available modes: 'manylinux' (searches /opt/python/ for interpreters "
            "matching supported versions in pyproject.toml)"
        ),
    )
    parser.add_argument(
        "-d",
        "--destination",
        default="dist/",
        type=str,
        help="Destination to put the compiled binaries",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.find_python:
        if args.python:
            logger.warning(
                "Both --python and --find-python specified. Using --find-python and ignoring --python."
            )
        pythons = find_python_interpreters(args.find_python)
        if not pythons:
            logger.error(
                "No Python interpreters found with --find-python %s", args.find_python
            )
            sys.exit(1)
        logger.info(
            "Found %d supported Python interpreters: %s",
            len(pythons),
            ", ".join(pythons),
        )
    else:
        pythons = args.python or [sys.executable]

    build_times: dict[str, float] = dict()

    if len(pythons) > 1 and args.destination == "dist/":
        logger.warning(
            "dest is 'dist/' while multiple python versions specified, output will be overwritten"
        )

    for interpreter in pythons:
        with venv(interpreter) as venv_interpreter:
            builder = Builder(venv_interpreter)
            # clean actually requires setuptools so we need to ensure we
            # install requirements before
            builder.install_requirements()
            builder.clean()

            start_time = time.time()

            builder.build_wheel(args.destination)

            end_time = time.time()

            build_times[interpreter_version(venv_interpreter)] = end_time - start_time
    for version, build_time in build_times.items():
        logger.info("Build time (%s): %fs", version, build_time)


if __name__ == "__main__":
    main()
