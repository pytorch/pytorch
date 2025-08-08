import logging
import os
import shlex
import subprocess
import sys
from dataclasses import fields
from textwrap import indent
from typing import Optional


logger = logging.getLogger(__name__)


def generate_dataclass_help(cls) -> str:
    """Auto-generate help text for dataclass default values."""
    lines = []
    for field in fields(cls):
        default = field.default
        if default is not None and default != "":
            lines.append(f"{field.name:<22} = {repr(default)}")
        else:
            lines.append(f"{field.name:<22} = ''")
    return indent("\n".join(lines), "    ")


def run_shell(
    cmd: str,
    log_cmd: bool = True,
    cwd: Optional[str] = None,
    env: Optional[dict] = None,
):
    """
    Run a shell command using /bin/bash.

    Args:
        cmd (str): The command string to execute.
        log_cmd (bool): Whether to log the command before execution.
        cwd (Optional[str]): Working directory to run the command in.
        env (Optional[dict]): Environment variables to set during execution.

    Raises:
        subprocess.CalledProcessError: If the command fails.
    """
    if log_cmd:
        logger.info(f"[shell] {cmd}")
    try:
        subprocess.run(
            cmd,
            shell=True,
            executable="/bin/bash",
            stdout=sys.stdout,
            stderr=sys.stderr,
            check=True,
            env=env,
            cwd=cwd,
        )
    except subprocess.CalledProcessError as e:
        logger.error(
            f"[shell] Command failed.\n"
            f"Command: {cmd}\n"
            f"Exit code: {e.returncode}\n"
            f"STDOUT:\n{getattr(e, 'stdout', '')}\n"
            f"STDERR:\n{getattr(e, 'stderr', '')}"
        )
        raise


def run_cmd(
    cmd: str,
    log_cmd: bool = True,
    cwd: Optional[str] = None,
    env: Optional[dict] = None,
):
    """
    Run a command using subprocess with shell=False (i.e., direct exec).
    This only works for commands that are not shell builtins. It is recommended
    to use this method rather than run_shell().

    Args:
        cmd (str): The command string to execute (will be split using shlex).
        log_cmd (bool): Whether to log the command before execution.
        cwd (Optional[str]): Working directory to run the command in.
        env (Optional[dict]): Environment variables to set during execution.

    Raises:
        subprocess.CalledProcessError: If the command fails.
    """
    args = shlex.split(cmd)

    if log_cmd:
        logger.info(f"[cmd] {' '.join(args)}")
    try:
        subprocess.run(
            args,
            shell=False,
            stdout=sys.stdout,
            stderr=sys.stderr,
            check=True,
            env=env,
            cwd=cwd,
        )
    except subprocess.CalledProcessError as e:
        logger.error(
            f"[cmd] Command failed.\n"
            f"Command: {cmd}\n"
            f"Exit code: {e.returncode}\n"
            f"STDOUT:\n{getattr(e, 'stdout', '')}\n"
            f"STDERR:\n{getattr(e, 'stderr', '')}"
        )
        raise


def get_env(name: str, default: str = "") -> str:
    """
    Get an environment variable with a default fallback.
    """
    return os.environ.get(name, default)
