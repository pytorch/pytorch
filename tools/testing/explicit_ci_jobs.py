#!/usr/bin/env python3

from __future__ import annotations

import argparse
import fnmatch
import subprocess
import textwrap
from pathlib import Path


REPO_ROOT = Path(__file__).parents[2]
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"


def commit_ci(files: list[str], message: str) -> None:
    # Check that there are no other modified files than the ones edited by this
    # tool
    stdout = subprocess.run(
        ["git", "status", "--porcelain"], stdout=subprocess.PIPE
    ).stdout.decode()
    for line in stdout.split("\n"):
        if line == "":
            continue
        if line[0] != " ":
            raise RuntimeError(
                f"Refusing to commit while other changes are already staged: {line}"
            )

    # Make the commit
    subprocess.run(["git", "add"] + files)
    subprocess.run(["git", "commit", "-m", message])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="delete GitHub Actions workflow files that don't match a filter"
    )
    parser.add_argument(
        "--filter-gha", help="keep only these github actions (glob match)", default=""
    )
    parser.add_argument(
        "--make-commit",
        action="store_true",
        help="add change to git with to a do-not-merge commit",
    )
    args = parser.parse_args()

    touched_files: list[Path] = []

    if args.filter_gha:
        for relative_file in WORKFLOWS_DIR.iterdir():
            path = REPO_ROOT.joinpath(relative_file)
            if not fnmatch.fnmatch(path.name, args.filter_gha):
                touched_files.append(path)
                path.resolve().unlink()

    if args.make_commit:
        message = textwrap.dedent(
            """
        [skip ci][do not merge] Filter GitHub Actions workflow files

        See [Run Specific CI Jobs](https://github.com/pytorch/pytorch/blob/master/CONTRIBUTING.md#run-specific-ci-jobs) for details.
        """
        ).strip()
        commit_ci([str(f.relative_to(REPO_ROOT)) for f in touched_files], message)
