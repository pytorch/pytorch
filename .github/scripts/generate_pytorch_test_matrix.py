#!/usr/bin/env python3

"""Generates a matrix to be utilized through github actions

Will output a matrix to represent our testing configurations, which is currently
dictated by just sharding.

"""

import json
import os
import re
from typing import Dict

from typing_extensions import TypedDict


class Config(TypedDict):
    num_shards: int
    runner: str


def get_disabled_issues() -> str:
    pr_body = os.getenv('PR_BODY', '')
    # The below regex is meant to match all *case-insensitive* keywords that
    # GitHub has delineated would link PRs to issues, more details here:
    # https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue.
    # E.g., "Close #62851", "fixES #62851" and "RESOLVED #62851" would all match, but not
    # "closes  #62851" --> extra space, "fixing #62851" --> not a keyword, nor "fix 62851" --> no #
    regex = '(?i)(Close(d|s)?|Resolve(d|s)?|Fix(ed|es)?) #([0-9]+)'
    issue_numbers = [x[4] for x in re.findall(regex, pr_body)]
    return ','.join(issue_numbers)


def main() -> None:
    TEST_RUNNER_TYPE = os.getenv('TEST_RUNNER_TYPE')
    assert TEST_RUNNER_TYPE is not None
    ON_PULL_REQUEST = os.getenv('GITHUB_HEAD_REF')
    NUM_TEST_SHARDS_ON_PULL_REQUEST = os.getenv('NUM_TEST_SHARDS_ON_PULL_REQUEST')
    NUM_TEST_SHARDS = int(os.getenv('NUM_TEST_SHARDS', '1'))
    if ON_PULL_REQUEST and NUM_TEST_SHARDS_ON_PULL_REQUEST:
        NUM_TEST_SHARDS = int(NUM_TEST_SHARDS_ON_PULL_REQUEST)
    MULTIGPU_RUNNER_TYPE = os.getenv('MULTIGPU_RUNNER_TYPE')
    NOGPU_RUNNER_TYPE = os.getenv('NOGPU_RUNNER_TYPE')
    configs: Dict[str, Config] = {}
    if os.getenv('ENABLE_JIT_LEGACY_TEST'):
        configs['jit_legacy'] = {'num_shards': 1, 'runner': TEST_RUNNER_TYPE}
    if MULTIGPU_RUNNER_TYPE is not None and os.getenv('ENABLE_MULTIGPU_TEST'):
        configs['multigpu'] = {'num_shards': 1, 'runner': MULTIGPU_RUNNER_TYPE}
    if NOGPU_RUNNER_TYPE is not None and os.getenv('ENABLE_NOGPU_NO_AVX_TEST'):
        configs['nogpu_NO_AVX'] = {'num_shards': 1, 'runner': NOGPU_RUNNER_TYPE}
    if NOGPU_RUNNER_TYPE is not None and os.getenv('ENABLE_NOGPU_NO_AVX2_TEST'):
        configs['nogpu_NO_AVX2'] = {'num_shards': 1, 'runner': NOGPU_RUNNER_TYPE}
    if os.getenv('ENABLE_DISTRIBUTED_TEST'):
        configs['distributed'] = {'num_shards': 1, 'runner': TEST_RUNNER_TYPE}
    if os.getenv('ENABLE_SLOW_TEST'):
        configs['slow'] = {'num_shards': 1, 'runner': TEST_RUNNER_TYPE}
    if os.getenv('ENABLE_DOCS_TEST'):
        configs['docs_test'] = {'num_shards': 1, 'runner': TEST_RUNNER_TYPE}
    if os.getenv('ENABLE_BACKWARDS_COMPAT_TEST'):
        configs['backwards_compat'] = {'num_shards': 1, 'runner': TEST_RUNNER_TYPE}
    if os.getenv('ENABLE_XLA_TEST'):
        configs['xla'] = {'num_shards': 1, 'runner': TEST_RUNNER_TYPE}
    if os.getenv('ENABLE_NOARCH_TEST'):
        configs['noarch'] = {'num_shards': 1, 'runner': TEST_RUNNER_TYPE}
    matrix = {
        'include': [
            {
                'config': 'default',
                'shard': shard,
                'num_shards': NUM_TEST_SHARDS,
                'runner': TEST_RUNNER_TYPE,
            }
            for shard in range(1, NUM_TEST_SHARDS + 1)
        ] + [
            {
                'config': name,
                'shard': shard,
                'num_shards': config['num_shards'],
                'runner': config['runner'],
            }
            for name, config in configs.items()
            for shard in range(1, config['num_shards'] + 1)
        ]
    }
    render_matrix = {'config': list(dict.fromkeys(x['config'] for x in matrix['include']))}
    print(json.dumps({'matrix': matrix, 'render-matrix': render_matrix}, indent=2))
    print(f'::set-output name=matrix::{json.dumps(matrix)}')
    print(f'::set-output name=render-matrix::{json.dumps(render_matrix)}')
    print(f'::set-output name=ignore-disabled-issues::{get_disabled_issues()}')


if __name__ == "__main__":
    main()
