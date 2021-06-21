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
    """
    Return the set of `files` in the `mypy` ini file at config_path.
    """
    config = ConfigParser()
    config.read(config_path)
    # hopefully on Windows this gives posix paths
    return set(mypy.config_parser.split_and_match_files(
        config['mypy']['files'],
    ))


# see tools/test/test_mypy_wrapper.py for examples of many of the
# following functions


def config_files() -> Dict[str, Set[str]]:
    """
    Return a dict from all our `mypy` ini filenames to their `files`.
    """
    return {str(ini): read_config(ini) for ini in Path().glob('mypy*.ini')}


def split_path(path: str) -> List[str]:
    """
    Split a relative (not absolute) POSIX path into its segments.
    """
    pure = PurePosixPath(path)
    return [str(p.name) for p in list(reversed(pure.parents))[1:] + [pure]]


# mypy doesn't support recursive types yet
# https://github.com/python/mypy/issues/731

# but if it did, the `Any` here would be `Union[Set[str], 'Trie']`,
# although that is not completely accurate: specifically, every `None`
# key must map to a `Set[str]`, and every `str` key must map to a `Trie`
Trie = Dict[Optional[str], Any]


def make_trie(configs: Dict[str, Set[str]]) -> Trie:
    """
    Return a trie from path prefixes to their `mypy` configs.

    Specifically, each layer of the trie represents a segment of a POSIX
    path relative to the root of this repo. If you follow a path down
    the trie and reach a `None` key, that `None` maps to the (nonempty)
    set of keys in `configs` which explicitly include that path.
    """
    trie: Trie = {}
    for ini, files in configs.items():
        for f in files:
            inner = trie
            for segment in split_path(f):
                inner = inner.setdefault(segment, {})
            inner.setdefault(None, set()).add(ini)
    return trie


def lookup(trie: Trie, filename: str) -> Set[str]:
    """
    Return the configs in `trie` that include a prefix of `filename`.

    A path is included by a config if any of its ancestors are included
    by the wildcard-expanded version of that config's `files`. Thus,
    this function follows `filename`'s path down the `trie` and
    accumulates all the configs it finds along the way.
    """
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
    """
    Return a dict from config names to the files to run them with.

    The keys of the returned dict are a subset of the keys of `configs`.
    The list of files in each value of returned dict should contain a
    nonempty subset of the given `files`, in the same order as `files`.
    """
    trie = make_trie(configs)
    plan = defaultdict(list)
    for filename in files:
        for config in lookup(trie, filename):
            plan[config].append(filename)
    return plan


def run(
    *,
    args: List[str],
    files: List[str],
) -> Tuple[int, List[str], List[str]]:
    """
    Return the exit code and list of output lines from running `mypy`.

    The given `args` are passed verbatim to `mypy`. The `files` (each of
    which must be an absolute path) are converted to relative paths
    (that is, relative to the root of this repo) and then classified
    according to which ones need to be run with each `mypy` config.
    Thus, `mypy` may be run zero, one, or multiple times, but it will be
    run at most once for each `mypy` config used by this repo.
    """
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
            for stdout, _, _ in mypy_results
            for item in stdout.splitlines()
        )),
        [stderr for _, stderr, _ in mypy_results],
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
    exit_code, mypy_issues, stderrs = run(
        args=[arg for arg in args if not arg.startswith(repo_root)],
        files=[arg for arg in args if arg.startswith(repo_root)],
    )
    for issue in mypy_issues:
        print(issue)
    for stderr in stderrs:
        print(stderr, end='', file=sys.stderr)
    sys.exit(exit_code)


if __name__ == '__main__':
    main(sys.argv[1:])
