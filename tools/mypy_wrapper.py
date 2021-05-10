#!/usr/bin/env python3

"""
This module is meant to be run as a script (see the docstring of main
below) and passed the filename of any Python file in this repo, to
typecheck that file using only the subset of our mypy configs that apply
to it.

Since editors (e.g. VS Code) can be configured to use this wrapper
script in lieu of mypy itself, the idea is that this can be used to get
inline mypy results while developing, and have at least some degree of
assurance that those inline results match up with what you would get
from running the mypy lint from the .github/workflows/lint.yml file.

See also these wiki pages:

- https://github.com/pytorch/pytorch/wiki/Guide-for-adding-type-annotations-to-PyTorch
- https://github.com/pytorch/pytorch/wiki/Lint-as-you-type
"""

import sys
from collections import defaultdict
from configparser import ConfigParser
from pathlib import Path, PurePath, PurePosixPath
from typing import Any, Dict, List, Optional, Set, Tuple

import mypy.api
# not part of the public API, but this is the easiest way to ensure that
# we agree with what mypy actually does
import mypy.config_parser


def read_config(config_path: Path) -> Set[str]:
    config = ConfigParser()
    config.read(config_path)
    # hopefully on Windows this gives posix paths
    return set(mypy.config_parser.split_and_match_files(
        config['mypy']['files'],
    ))


def config_files() -> Dict[str, Set[str]]:
    return {str(ini): read_config(ini) for ini in Path().glob('mypy*.ini')}


def split_path(path: str) -> List[str]:
    pure = PurePosixPath(path)
    return [str(p.name) for p in list(reversed(pure.parents))[1:] + [pure]]


# mypy doesn't support recursive types yet
# https://github.com/python/mypy/issues/731
Trie = Dict[Optional[str], Any]


def make_trie(configs: Dict[str, Set[str]]) -> Trie:
    trie: Trie = {}
    for ini, files in configs.items():
        for f in files:
            inner = trie
            for segment in split_path(f):
                inner = inner.setdefault(segment, {})
            inner.setdefault(None, set()).add(ini)
    return trie


def lookup(trie: Trie, filename: str) -> Set[str]:
    configs = set()
    inner = trie
    for segment in split_path(filename):
        inner = inner.get(segment, {})
        configs |= inner.get(None, set())
    return configs


def make_plan(
    *,
    configs: Dict[str, Set[str]],
    files: List[str]
) -> Dict[str, List[str]]:
    trie = make_trie(configs)
    plan = defaultdict(list)
    for filename in files:
        for config in lookup(trie, filename):
            plan[config].append(filename)
    return plan


def run(*, args: List[str], files: List[str]) -> Tuple[int, List[str]]:
    repo_root = Path.cwd()
    plan = make_plan(configs=config_files(), files=[
        PurePath(f).relative_to(repo_root).as_posix() for f in files
    ])
    mypy_results = [
        mypy.api.run(
            # insert custom flags after args to avoid being overridden
            # by existing flags in args
            args + [
                # don't special-case the last line
                '--no-error-summary',
                f'--config-file={config}',
            ] + filtered
        )
        # by construction, filtered must be nonempty
        for config, filtered in plan.items()
    ]
    return (
        # assume all mypy exit codes are nonnegative
        # https://github.com/python/mypy/issues/6003
        max(
            [exit_code for _, _, exit_code in mypy_results],
            default=0,
        ),
        list(dict.fromkeys(  # remove duplicates, retain order
            item
            # assume stderr is empty
            # https://github.com/python/mypy/issues/1051
            for stdout, _, _ in mypy_results
            for item in stdout.splitlines()
        )),
    )


def main(args: List[str]) -> None:
    """
    Run mypy on one Python file using the correct config file(s).

    This function assumes the following preconditions hold:

    - the cwd is set to the root of this cloned repo
    - args is a valid list of CLI arguments that could be passed to mypy
    - some of args are absolute paths to files to typecheck
    - all the other args are config flags for mypy, rather than files

    These assumptions hold, for instance, when mypy is run automatically
    by VS Code's Python extension, so in your clone of this repository,
    you could modify your .vscode/settings.json to look something like
    this (assuming you use a conda environment named "pytorch"):

        {
          "python.linting.enabled": true,
          "python.linting.mypyEnabled": true,
          "python.linting.mypyPath":
            "${env:HOME}/miniconda3/envs/pytorch/bin/python",
          "python.linting.mypyArgs": [
            "${workspaceFolder}/tools/mypy_wrapper.py"
          ]
        }

    More generally, this should work for any editor sets the cwd to the
    repo root, runs mypy on individual files via their absolute paths,
    and allows you to set the path to the mypy executable.
    """
    repo_root = str(Path.cwd())
    exit_code, mypy_issues = run(
        args=[arg for arg in args if not arg.startswith(repo_root)],
        files=[arg for arg in args if arg.startswith(repo_root)],
    )
    for issue in mypy_issues:
        print(issue)
    sys.exit(exit_code)


if __name__ == '__main__':
    main(sys.argv[1:])
