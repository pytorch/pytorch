import glob
import logging
import os
import shlex
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import fields
from textwrap import indent
from typing import Dict, List, Optional


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


@contextmanager
def temp_environ(updates: Dict[str, str]):
    """
    Temporarily set environment variables and restore them after the block.

    Args:
        updates: Dict of environment variables to set.
    """
    missing = object()
    old: Dict[str, str | object] = {k: os.environ.get(k, missing) for k in updates}
    try:
        os.environ.update(updates)
        yield
    finally:
        for k, v in old.items():
            if v is missing:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v  # type: ignore[arg-type]


def list_to_env_dict(env_list: List[str]) -> Dict[str, str]:
    env_dict: Dict[str, str] = {}
    for item in env_list:
        if "=" not in item:
            raise ValueError(f"Invalid env var format: {item!r}, expected KEY=VALUE")
        key, value = item.split("=", 1)
        env_dict[key.strip()] = value.strip()
    return env_dict


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


def pip_install_first_wheel(pattern: str, extras: str | None = None):
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
    run_cmd(f"{shlex.quote(sys.executable)} -m pip install {shlex.quote(target)}")


def pip_install(args: str | list[str], env=None):
    """
    Install a package using pip for the python in current environment.

    Args:
        args (str | list[str]): Arguments to pass to pip install.
                                If str, it will be split with shlex.split().
    """
    if isinstance(args, str):
        args = shlex.split(args)
    cmd = [sys.executable, "-m", "pip", "install"] + args
    run_cmd(" ".join(map(shlex.quote, cmd)), env=env)


def uv_pip_install(args: str | list[str], env=None):
    """
    Install a package using uv pip for the python in current environment.

    Args:
        args (str | list[str]): Arguments to pass to uv pip install.
                                If str, it will be split with shlex.split().
    """
    if isinstance(args, str):
        args = shlex.split(args)
    cmd = [sys.executable, "-m", "uv", "pip", "install"] + args
    run_cmd(" ".join(map(shlex.quote, cmd)), env=env)
