#!/usr/bin/env python3

from gitutils import get_git_repo_dir, GitRepo
from typing import Any


def parse_args() -> Any:
    from argparse import ArgumentParser
    parser = ArgumentParser("Merge PR/branch into default branch")
    parser.add_argument("--sync-branch", default="sync")
    parser.add_argument("--default-branch", type=str, default="main")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = GitRepo(get_git_repo_dir(), debug=args.debug)
    repo.cherry_pick_commits(args.sync_branch, args.default_branch)
    repo.push(args.default_branch, args.dry_run)


if __name__ == '__main__':
    main()
