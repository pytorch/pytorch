#!/usr/bin/env python3

"""Generates a matrix to be utilized through github actions

Will output a matrix to represent our testing configurations, which is currently
dictated by just sharding.

"""

import json
import os


def main() -> None:
    NUM_TEST_SHARDS = int(os.getenv('NUM_TEST_SHARDS', '1'))
    shards_per_config = {}
    if os.getenv('ENABLE_MULTIGPU_TEST'):
        shards_per_config['multigpu'] = 1
    print(json.dumps({
        'include': [
            {
                'config': 'default',
                'shard': shard,
                'num_shards': NUM_TEST_SHARDS,
            }
            for shard in range(1, NUM_TEST_SHARDS + 1)
        ] + [
            {
                'config': config,
                'shard': shard,
                'num_shards': num_shards,
            }
            for config, num_shards in shards_per_config.items()
            for shard in range(1, num_shards + 1)
        ]
    }))


if __name__ == "__main__":
    main()
