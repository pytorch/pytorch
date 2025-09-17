"""
Git Utility helpers for CLI tasks.
"""

import logging
from pathlib import Path

from cli.lib.common.path_helper import remove_dir
from git import GitCommandError, RemoteProgress, Repo


logger = logging.getLogger(__name__)


class PrintProgress(RemoteProgress):
    """Simple progress logger for git operations."""

    def __init__(self, interval: int = 5):
        super().__init__()
        self._last_percent = -1
        self._interval = interval

    def update(self, op_code, cur, max=None, message=""):
        msg = self._cur_line or message
        if max and cur:
            percent = int(cur / max * 100)
            if percent != self._last_percent and percent % self._interval == 0:
                self._last_percent = percent
                logger.info("Progress: %d%% - %s", percent, msg)
        elif msg:
            logger.info(msg)


def clone_external_repo(target: str, repo: str, dst: str = "", update_submodules=False):
    """Clone repository with pinned commit and optional submodules."""
    dst = dst or target

    try:
        logger.info("Cloning %s to %s", target, dst)

        # Clone and fetch
        remove_dir(dst)
        r = Repo.clone_from(repo, dst, progress=PrintProgress())
        r.git.fetch("--all", "--tags")

        # Checkout pinned commit
        commit = get_post_build_pinned_commit(target)
        logger.info("Checking out pinned %s commit %s", target, commit)
        r.git.checkout(commit)

        # Update submodules if requested
        if update_submodules and r.submodules:
            logger.info("Updating %d submodule(s)", len(r.submodules))
            for sm in r.submodules:
                sm.update(init=True, recursive=True, progress=PrintProgress())

        logger.info("Successfully cloned %s", target)
        return r, commit

    except GitCommandError as e:
        logger.error("Git operation failed: %s", e)
        raise


def get_post_build_pinned_commit(name: str, prefix=".github/ci_commit_pins") -> str:
    path = Path(prefix) / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Pin file not found: {path}")
    return path.read_text(encoding="utf-8").strip()
