#!/usr/bin/env python3

"""Generates a matrix to be utilized through github actions

Will output a matrix to represent our testing configurations, which is currently
dictated by just sharding.

"""

import json
import os
from typing import List


NUM_TEST_SHARDS = int(os.getenv('NUM_TEST_SHARDS', '1'))

def generate_sharding_list() -> List[int]:
    return list(range(1, NUM_TEST_SHARDS + 1))


def main() -> None:
    print(json.dumps(
        {
            'test_config': generate_sharding_list()
        },
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
