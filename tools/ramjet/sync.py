"""Your working tree as a commit, and the bundles and tars that carry it to Modal."""

import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

from tools.ramjet.common import git
from tools.ramjet.logger import log


def snapshot_tree() -> str:
    """Commit the working tree, untracked files included, without touching
    HEAD, the index or the stash."""
    index = git("rev-parse", "--git-path", "index")
    with tempfile.NamedTemporaryFile() as scratch:
        shutil.copyfile(index, scratch.name)  # keeps stat data: only edits get hashed
        env = {**os.environ, "GIT_INDEX_FILE": scratch.name}
        # git add --all on its own runs git status in every submodule to see if it is
        # dirty, which takes seconds; ls-files finds the changed paths without that
        changed = git("ls-files", "-z", "-m", "-o", "--exclude-standard", env=env)
        if changed:  # an empty list of paths would add everything
            paths = ["--pathspec-from-file=-", "--pathspec-file-nul"]
            git("--literal-pathspecs", "add", "--all", *paths, env=env, stdin=changed)
        tree = git("write-tree", env=env)
    return git("commit-tree", "--no-gpg-sign", tree, "-p", "HEAD", "-m", "ramjet")


def bundle(commit: str, since: str, ramjet_dir: Path) -> Path:
    """Pack only what the builder lacks: usually a few kilobytes."""
    path = ramjet_dir / "build.bundle"
    git("update-ref", "refs/worktree/ramjet", commit)  # refs/worktree/ is per worktree
    git("bundle", "create", str(path), "refs/worktree/ramjet", f"^{since}")
    return path


def diff(since: str, commit: str) -> dict[str, str]:
    """Paths that differ between two commits, each with its status: A, M, T or D."""
    out = git("diff", "--name-status", "--no-renames", "-z", since, commit)
    fields = out.split("\0")[:-1]
    return dict(zip(fields[1::2], fields[::2]))


def archive(commit: str, paths: list[str], path: Path) -> None:
    """These paths as of the commit: all a runner needs for a change the build doesn't
    read."""
    git("archive", "--format=tar", "-o", str(path), commit, "--", *paths)


def merge_base(commit: str) -> str:
    """Where the commit left pytorch/pytorch's main, as your clone last fetched it."""
    for remote in git("remote").split():
        url = git("remote", "get-url", remote)
        if re.search(r"github\.com[:/]pytorch/pytorch(\.git)?$", url):
            return git("merge-base", commit, f"{remote}/main")
    log.error("no git remote points at github.com/pytorch/pytorch")
    sys.exit(1)
