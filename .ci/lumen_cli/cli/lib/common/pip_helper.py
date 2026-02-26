import glob
import logging
import shlex
import shutil
import sys
from collections.abc import Iterable
from importlib.metadata import PackageNotFoundError, version  # noqa: UP035
from typing import Optional, Union

from cli.lib.common.utils import run_command


logger = logging.getLogger(__name__)


def pip_install_packages(
    packages: Iterable[str] = (),
    env=None,
    *,
    requirements: Optional[str] = None,
    constraints: Optional[str] = None,
    prefer_uv: bool = False,
) -> None:
    use_uv = prefer_uv and shutil.which("uv") is not None
    base = (
        [sys.executable, "-m", "uv", "pip", "install"]
        if use_uv
        else [sys.executable, "-m", "pip", "install"]
    )
    cmd = base[:]
    if requirements:
        cmd += ["-r", requirements]
    if constraints:
        cmd += ["-c", constraints]
    cmd += list(packages)
    logger.info("pip installing packages: %s", " ".join(map(shlex.quote, cmd)))
    run_command(" ".join(map(shlex.quote, cmd)), env=env)


def pip_install_first_match(pattern: str, extras: Optional[str] = None, pref_uv=False):
    wheel = first_matching_pkg(pattern)
    target = f"{wheel}[{extras}]" if extras else wheel
    logger.info("Installing %s...", target)
    pip_install_packages([target], prefer_uv=pref_uv)


def run_python(args: Union[str, list[str]], env=None):
    """
    Run the python in the current environment.
    """
    if isinstance(args, str):
        args = shlex.split(args)
    cmd = [sys.executable] + args
    run_command(" ".join(map(shlex.quote, cmd)), env=env)


def pkg_exists(name: str) -> bool:
    try:
        pkg_version = version(name)
        logger.info("%s already exist with version: %s", name, pkg_version)
        return True
    except PackageNotFoundError:
        logger.info("%s is not installed", name)
        return False


def first_matching_pkg(pattern: str) -> str:
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No wheel matching: {pattern}")
    return matches[0]
