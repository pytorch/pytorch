"""
General Utility helpers for CLI tasks.
"""

import logging
import os
import shlex
import subprocess
import sys
from contextlib import contextmanager
from typing import Optional


logger = logging.getLogger(__name__)


def run_shell(
    cmd: str,
    log_cmd: bool = True,
    cwd: Optional[str] = None,
    env: Optional[dict] = None,
    check: bool = True,
) -> int:
    """
    Run a shell command via /bin/bash.
    Returns the exit code. If check=True and the exit code is non-zero,
    raises subprocess.CalledProcessError (matching subprocess.run behavior).
    """
    if log_cmd:
        logger.info("[shell] %s", cmd)

    run_env = {**os.environ, **env} if env else None

    proc = subprocess.run(
        cmd,
        shell=True,
        executable="/bin/bash",
        stdout=sys.stdout,
        stderr=sys.stderr,
        cwd=cwd,
        env=run_env,
        check=False,  # handle 'check' manually
    )

    if check and proc.returncode != 0:
        logger.error("[shell] Command failed (exit %s): %s", proc.returncode, cmd)
        raise subprocess.CalledProcessError(proc.returncode, cmd)

    return proc.returncode


def run_cmd(
    cmd: str,
    log_cmd: bool = True,
    cwd: Optional[str] = None,
    env: Optional[dict] = None,
    check: bool = True,
) -> int:
    """
    Run a command using subprocess.run with shell=False.

    Args:
        cmd[str]: Command string (split with shlex).
        log_cmd [bool]: Log command before execution.
        cwd [Optional[str]]: Working directory.
        env [Optional[dict]]: Environment vars to overlay on current env.
        check [bool]: If True, raise on non-zero exit; else return code.

    Returns:
        int: The process's exit code.
    """
    args = shlex.split(cmd)
    if log_cmd:
        logger.info("[cmd] %s", " ".join(args))

    run_env = {**os.environ, **env} if env else None

    proc = subprocess.run(
        args,
        shell=False,
        stdout=sys.stdout,
        stderr=sys.stderr,
        cwd=cwd,
        env=run_env,
        check=False,  # we'll handle check manually
    )

    if check and proc.returncode != 0:
        logger.error("[cmd] Command failed (exit %s): %s", proc.returncode, cmd)
        raise subprocess.CalledProcessError(proc.returncode, args)

    return proc.returncode


def str2bool(value: Optional[str]) -> bool:
    """Convert environment variables to boolean values."""
    if not value:
        return False
    if not isinstance(value, str):
        raise ValueError(
            f"Expected a string value for boolean conversion, got {type(value)}"
        )
    value = value.strip().lower()
    if value in (
        "1",
        "true",
        "t",
        "yes",
        "y",
        "on",
        "enable",
        "enabled",
        "found",
    ):
        return True
    if value in (
        "0",
        "false",
        "f",
        "no",
        "n",
        "off",
        "disable",
        "disabled",
        "notfound",
        "none",
        "null",
        "nil",
        "undefined",
        "n/a",
    ):
        return False
    raise ValueError(f"Invalid string value for boolean conversion: {value}")


@contextmanager
def temp_environ(updates: dict[str, str]):
    """
    Temporarily set environment variables and restore them after the block.
    Args:
        updates: Dict of environment variables to set.
    """
    missing = object()
    old: dict[str, str | object] = {k: os.environ.get(k, missing) for k in updates}
    try:
        os.environ.update(updates)
        yield
    finally:
        for k, v in old.items():
            if v is missing:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v  # type: ignore[arg-type]


@contextmanager
def working_directory(path: str):
    """
    Temporarily change the working directory inside a context.
    """
    if not path:
        # No-op context
        yield
        return
    prev_cwd = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(prev_cwd)
