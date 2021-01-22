#!/usr/bin/env python3

"""
This module serves two purposes:

- it holds the config_files function, which defines the set of subtests
  for the test_run_mypy test in test/test_type_hints.py
- it can be run as a script (see the docstring of main below) and passed
  the filename of any Python file in this repo, to typecheck that file
  using only the subset of our mypy configs that apply to it

Since editors (e.g. VS Code) can be configured to use this wrapper
script in lieu of mypy itself, the idea is that this can be used to get
inline mypy results while developing, and have at least some degree of
assurance that those inline results match up with what you would get
from running the TestTypeHints test suite in CI.

See also these wiki pages:

- https://github.com/pytorch/pytorch/wiki/Guide-for-adding-type-annotations-to-PyTorch
- https://github.com/pytorch/pytorch/wiki/Lint-as-you-type
"""

import re
import sys
from configparser import ConfigParser
from itertools import chain
from pathlib import Path
from typing import List, Set

# don't import any files that live in the PyTorch repo, since this is
# meant to work as a standalone script

try:
    import mypy.api
except ImportError:
    # let test/test_type_hints.py import this even if mypy is absent
    pass


def config_files() -> Set[str]:
    """
    Return a set of the names of all the PyTorch mypy config files.
    """
    return {
        'mypy.ini',
        'mypy-strict.ini',
    }


def repo_root() -> Path:
    """
    This script assumes that the cwd is the PyTorch repository root.
    """
    return Path.cwd()


def glob(*, pattern: str, filename: str) -> bool:
    """
    Return True iff the filename matches the (mypy ini) glob pattern.
    """
    full_pattern = str(repo_root() / pattern)
    path = Path(filename)
    return any(
        prefix.resolve().match(full_pattern)
        for prefix in chain([path], path.parents)
    )


def in_files(*, ini: str, py: str) -> bool:
    """
    Return True iff the py file is included in the ini file's "files".
    """
    config = ConfigParser()
    config.read(repo_root() / ini)
    return any(
        glob(pattern=pattern, filename=py)
        for pattern in re.split(r',\s*', config['mypy']['files'].strip())
    )


def main(args: List[str]) -> None:
    """
    Run mypy on one Python file using the correct config file(s).

    This function assumes the following about its input:

    - args is a valid list of CLI arguments that could be passed to mypy
    - the last element of args is the path of a file to typecheck
    - all the other args are config flags for mypy, rather than files

    These assumptions hold, for instance, when mypy is run automatically
    by VS Code's Python extension, so in your clone of this repository,
    you could modify your .vscode/settings.json to look like this:

        {
          "python.linting.enabled": true,
          "python.linting.mypyEnabled": true,
          "python.linting.mypyPath":
            "/path/to/pytorch/torch/testing/_internal/mypy_wrapper.py",
          "python.linting.mypyArgs": [
            "--show-column-numbers"
          ]
        }

    More generally, this should work for any editor that runs mypy on
    one file at a time (setting the cwd to the repo root) and allows you
    to set the path to the mypy executable.
    """
    if not args:
        sys.exit('The PyTorch mypy wrapper must be passed exactly one file.')
    configs = [f for f in config_files() if in_files(ini=f, py=args[-1])]
    mypy_results = [
        mypy.api.run(
            # insert right before args[-1] to avoid being overridden
            # by existing flags in args[:-1]
            args[:-1] + [
                '--no-error-summary',
                f'--config-file={config}',
                args[-1],
            ]
        )
        for config in configs
    ]
    mypy_issues = [
        item
        # assume stderr is empty
        # https://github.com/python/mypy/issues/1051
        for stdout, _, _ in mypy_results
        for item in stdout.splitlines()
    ]
    for issue in mypy_issues:
        print(issue)
    # assume all mypy exit codes are nonnegative
    # https://github.com/python/mypy/issues/6003
    sys.exit(max(
        [exit_code for _, _, exit_code in mypy_results],
        default=0,
    ))


if __name__ == '__main__':
    main(sys.argv[1:])
