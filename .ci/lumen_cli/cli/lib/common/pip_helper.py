import glob
import logging
import shlex
import shutil
import sys
from collections.abc import Iterable
from typing import Optional, Union

from cli.lib.common.utils import run_cmd


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

    if use_uv:
        logger.info("Installing packages using uv pip")

    cmd = base[:]
    if requirements:
        cmd += ["-r", requirements]
    if constraints:
        cmd += ["-c", constraints]
    cmd += list(packages)

    logger.info("pip installing packages: %s", " ".join(map(shlex.quote, cmd)))
    run_cmd(" ".join(map(shlex.quote, cmd)), env=env)
    logger.info("Done installing packages")


def pip_install_first_match(pattern: str, extras: Optional[str] = None, pref_uv=False):
    """
    Install the first local whl that matches the given glob pattern.

    Args:
        pattern (str): Glob pattern for the wheel file(s).
        extras (str | None): Optional extras (e.g., "opt_einsum") to install with the wheel.
    """
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files match: {pattern}")
    wheel = matches[0]
    target = f"{wheel}[{extras}]" if extras else wheel
    pip_install_packages([target], prefer_uv=pref_uv)


def run_python(args: Union[str, list[str]], env=None):
    """
    Run the python in the current environment.
    """
    if isinstance(args, str):
        args = shlex.split(args)
    cmd = [sys.executable] + args
    run_cmd(" ".join(map(shlex.quote, cmd)), env=env)
