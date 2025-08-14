import glob
import logging
import shlex
import shutil
import sys
import zipfile
from collections.abc import Iterable
from importlib.metadata import PackageNotFoundError, version
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

    if use_uv:
        logger.info("Installing packages using uv pip")

    cmd = base[:]
    if requirements:
        cmd += ["-r", requirements]
    if constraints:
        cmd += ["-c", constraints]
    cmd += list(packages)

    logger.info("pip installing packages: %s", " ".join(map(shlex.quote, cmd)))
    run_command(" ".join(map(shlex.quote, cmd)), env=env)
    logger.info("Done installing packages")


def pip_install_first_match(pattern: str, extras: Optional[str] = None, pref_uv=False):
    wheel = first_matching_wheel(pattern)
    target = f"{wheel}[{extras}]" if extras else wheel
    logger.info("Installing %s", target)
    pip_install_packages([target], prefer_uv=pref_uv)
    logger.info("Done installing %s", target)


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
        torch_version = version(name)
        logger.info("%s already exist with version: %s", name, torch_version)
        return True
    except PackageNotFoundError:
        logger.info("%s is not installed", name)
        return False


def wheel_version_from_metadata(wheel_path: str) -> Optional[str]:
    try:
        with zipfile.ZipFile(wheel_path) as zf:
            meta_file = next((n for n in zf.namelist() if n.endswith("METADATA")), None)
            if not meta_file:
                return None
            with zf.open(meta_file) as fh:
                return next(
                    (
                        line.decode("utf-8", "replace").split(":", 1)[1].strip()
                        for line in fh
                        if line.startswith(b"Version:")
                    ),
                    None,
                )
    except Exception as e:
        logger.error("Failed to get wheel version from %s: %s", wheel_path, e)
        return None


def first_matching_wheel(pattern: str) -> str:
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No wheel matching: {pattern}")
    return matches[0]
