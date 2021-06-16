#!/usr/bin/env python3

import argparse
import re
import subprocess
from signal import SIG_DFL, SIGPIPE, signal
from typing import Dict, List, Optional, Set


def sieve(paths: Set[str], regexes: List[str]) -> Set[str]:
    return {
        path for path in paths
        if any(re.match(regex, path) for regex in regexes)
    }


Classification = Dict[str, bool]


def classify(paths: Set[str]) -> Optional[Classification]:
    classified = {}
    if sieve(paths, [
        r'.*\.(cpp|cu|h)',
        r'third_party/(fbgemm|ideep)',
        r'.jenkins/pytorch',
        r'tools/autograd',
        r'tools/codegen',
        r'.circleci/config.yml',
    ]):
        classified['build'] = True
    if sieve(paths, [r'(test|torch)/.*\.py']):
        classified['test'] = True
    if sieve(paths, [r'docs/source/.*\.rst']):
        classified['docs'] = True
    if sieve(paths, [r'.circleci/scripts/windows_cud(a|nn)_install.sh']):
        classified['windows'] = True
    if sieve(paths, [r'.circleci/scripts/binary_populate_env.sh']):
        classified['binary'] = True
    if classified:
        return classified
    left = paths - sieve(paths, [
        r'.github/pytorch-circleci-labels.yml',
        r'.github/workflows/lint.yml',
        r'.github/workflows/update_disabled_tests.yml',
    ])
    if not left:
        return {'build': False}
    return None


def git_lines(*args: str) -> List[str]:
    cmd = ['git']
    cmd.extend(args)
    return subprocess.check_output(cmd, encoding='ascii').splitlines()


def out(x: str) -> None:
    print(x, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--commit', default='master')
    args = parser.parse_args()

    commits = git_lines('rev-list', args.commit)
    for commit in commits:
        paths = git_lines(
            # https://stackoverflow.com/a/424142
            'diff-tree', '--no-commit-id', '--name-only', '-r', commit,
        )
        classified = classify(set(paths))
        if classified is None:
            out(f"- sha: '{commit}'")
            out('  paths:')
            for path in paths:
                out(f'    - {path}')
        else:
            out(''.join([f"- {{sha: '{commit}'"] + [
                f', {k}: {"yes" if v else "no"}'
                for k, v in classified.items()
            ] + ['}']))


if __name__ == '__main__':
    signal(SIGPIPE, SIG_DFL)  # https://stackoverflow.com/a/30091579
    try:
        main()
    except KeyboardInterrupt:
        pass
