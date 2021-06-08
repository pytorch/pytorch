#!/usr/bin/env python3

import os
import subprocess
from signal import SIG_DFL, SIGPIPE, signal
from typing import List

def git_lines(*args: str) -> List[str]:
    cmd = ['git']
    cmd.extend(args)
    return subprocess.check_output(cmd, encoding='ascii').splitlines()


def main() -> None:
    commits = git_lines('rev-list', 'master')
    for commit in commits:
        paths = git_lines(
            # https://stackoverflow.com/a/424142
            'diff-tree', '--no-commit-id', '--name-only', '-r', commit,
        )
        exts = set(os.path.splitext(p)[1] for p in paths)
        print(commit, exts, flush=True)


if __name__ == '__main__':
    signal(SIGPIPE, SIG_DFL)  # https://stackoverflow.com/a/30091579
    main()
