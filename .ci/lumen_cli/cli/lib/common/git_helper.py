"""
Git Utility helpers for CLI tasks.
"""

import logging
from pathlib import Path

from cli.lib.common.path_helper import remove_dir
from git import GitCommandError, RemoteProgress, Repo


logger = logging.getLogger(__name__)


class PrintProgress(RemoteProgress):
    """
    Class object passes to git package for logging progress
    """

    def __init__(self, every_percent: int = 1):
        super().__init__()
        self._last = -1
        self._every = every_percent

    def update(self, op_code, cur, max=None, message=""):
        line = self._cur_line or message
        if max:
            pct = int(cur / max * 100)
            if pct != self._last and pct % self._every == 0:
                self._last = pct
                logger.info(line)
        elif line:
            logger.info(line)


def clone_external_repo(target: str, repo: str, dst: str = "", update_submodules=False):
    """
    Clone a repo into a given directory.
    Args:
        target: The name of the target to clone, for instance, vllm
        repo: The full URL of the repo to clone.
        dst: The directory to clone the repo into, if not provided, uses target as the directory name
        update_submodules: bool enable/disable updated submodules if any
    Returns:
        The repo object
    """
    try:
        # Clone
        logger.info("cloning %s....", target)

        # if no dst is provided, use the target name
        if not dst:
            dst = target
        remove_dir(dst)
        r = Repo.clone_from(repo, dst, progress=PrintProgress())

        # Fetch all refs/tags to ensure commit is present
        r.git.fetch("--all", "--tags")

        logger.info("try to find the pinned commit for %s ...", target)
        commit = get_post_build_pinned_commit(target)
        logger.info("Done.found the pinned commit %s for %s ...", commit, target)

        logger.info("checkout %s", commit)
        r.git.checkout(commit)
        # Only update submodules if present

        if not update_submodules:
            logger.info("skipping submodule update.")
        else:
            if r.submodules:
                logger.info(" update repo submodules ....")
                for sm in r.submodules:
                    sm.update(init=True, recursive=True, progress=PrintProgress())
            else:
                logger.info(
                    "No submodules found for repo %s, skipping submodule update.", repo
                )
        logger.info("Checked out %s at %s in  %s", target, commit, dst)
        return r
    except GitCommandError as e:
        logger.error("Git operation failed: %v", e)


def get_post_build_pinned_commit(name: str, prefix=".github/ci_commit_pins") -> str:
    path = Path(prefix) / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Pin file not found: {path}")
    return path.read_text(encoding="utf-8").strip()
