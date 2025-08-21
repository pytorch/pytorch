"""
General Utility helpers for CLI tasks.
"""

import logging
import os
import shlex
import subprocess
import sys
from typing import Optional


logger = logging.getLogger(__name__)


def run_command(
    cmd: str,
    use_shell: bool = False,
    log_cmd: bool = True,
    cwd: Optional[str] = None,
    env: Optional[dict] = None,
    check: bool = True,
) -> int:
    """Run a command with optional shell execution."""
    if use_shell:
        args = cmd
        log_prefix = "[shell]"
        executable = "/bin/bash"
    else:
        args = shlex.split(cmd)
        log_prefix = "[cmd]"
        executable = None

    if log_cmd:
        display_cmd = cmd if use_shell else " ".join(args)
        logger.info("%s %s", log_prefix, display_cmd)

    run_env = {**os.environ, **(env or {})}

    proc = subprocess.run(
        args,
        shell=use_shell,
        executable=executable,
        stdout=sys.stdout,
        stderr=sys.stderr,
        cwd=cwd,
        env=run_env,
        check=False,
    )

    if check and proc.returncode != 0:
        logger.error(
            "%s Command failed (exit %s): %s", log_prefix, proc.returncode, cmd
        )
        raise subprocess.CalledProcessError(
            proc.returncode, args if not use_shell else cmd
        )

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

    true_value_set = {"1", "true", "t", "yes", "y", "on", "enable", "enabled", "found"}
    false_value_set = {"0", "false", "f", "no", "n", "off", "disable"}

    if value in true_value_set:
        return True
    if value in false_value_set:
        return False
    raise ValueError(f"Invalid string value for boolean conversion: {value}")
