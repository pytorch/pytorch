"""What the rest of ramjet shares: git, the worktree it runs in, and the directory
where it keeps its state."""

import functools
import getpass
import hashlib
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

from tools.ramjet.logger import log


def git(*args: str, env: dict[str, str] | None = None, stdin: str = "") -> str:
    run = {"capture_output": True, "text": True, "env": env, "input": stdin}
    result = subprocess.run(["git", *args], **run)
    if result.returncode != 0:
        log.error("git %s failed: %s", shlex.join(args), result.stderr.strip())
        sys.exit(1)
    return result.stdout.strip()


@functools.cache
def repo_root() -> Path:
    """The top of the worktree you run ramjet in."""
    return Path(git("rev-parse", "--show-toplevel")).resolve()


def ramjet_dir() -> Path:
    """/tmp/ramjet-<user>/<worktree folder>-<hash of its path>, or under $TMPDIR"""
    worktree = repo_root()
    user_dir = Path(tempfile.gettempdir(), f"ramjet-{getpass.getuser()}")
    user_dir.mkdir(mode=0o700, exist_ok=True)
    if user_dir.stat().st_uid != os.getuid():  # /tmp is shared with other users
        log.error("%s belongs to another user", user_dir)
        sys.exit(1)
    digest = hashlib.sha256(bytes(worktree)).hexdigest()[:8]
    path = user_dir / f"{worktree.name}-{digest}"
    path.mkdir(exist_ok=True)
    return path
