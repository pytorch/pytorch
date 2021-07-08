#!/usr/bin/env python3

"""Generates a matrix to be utilized through github actions

Will output a matrix to represent our testing configurations, which is currently
dictated by just sharding.

"""

import json
import os
from typing import Dict

from typing_extensions import TypedDict


class Config(TypedDict):
    num_shards: int
    runner: str


def main() -> None:
    TEST_RUNNER_TYPE = os.getenv('TEST_RUNNER_TYPE')
    NUM_TEST_SHARDS = int(os.getenv('NUM_TEST_SHARDS', '1'))
    MULTIGPU_RUNNER_TYPE = os.getenv('MULTIGPU_RUNNER_TYPE')
    configs: Dict[str, Config] = {}
    if MULTIGPU_RUNNER_TYPE is not None and os.getenv('ENABLE_MULTIGPU_TEST'):
        configs['multigpu'] = {'num_shards': 1, 'runner': MULTIGPU_RUNNER_TYPE}
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


if __name__ == "__main__":
    main()
