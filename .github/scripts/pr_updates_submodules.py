#!/usr/bin/env python3
# Checks if PR updates submodules by querying GitHub GraphQL

from typing import Any, List
from trymerge import GitHubPR

def gh_get_pr_updated_submodules(owner: str, name: str, number: int) -> List[str]:
    pr = GitHubPR(owner, name, number)
    files = pr.get_changed_files()
    submodules = pr.get_submodules()
    return [f for f in files if f in submodules]

def parse_args() -> Any:
    from argparse import ArgumentParser
    parser = ArgumentParser("Check if PR updates submodules")
    parser.add_argument("--owner", type=str, default="pytorch")
    parser.add_argument("--name", type=str, default="pytorch")
    parser.add_argument("pr_num", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(gh_get_pr_updated_submodules(name=args.name, owner=args.owner, number=args.pr_num))


if __name__ == "__main__":
    main()
