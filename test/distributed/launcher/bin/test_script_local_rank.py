#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="test script")

    parser.add_argument(
        "--local_rank",
        type=int,
        required=True,
        help="The rank of the node for multi-node distributed " "training",
    )

    return parser.parse_args()


def main():
    print("Start execution")
    args = parse_args()
    expected_rank = int(os.environ["LOCAL_RANK"])
    actual_rank = args.local_rank
    if expected_rank != actual_rank:
        raise RuntimeError(
            "Parameters passed: --local_rank that has different value "
            f"from env var: expected: {expected_rank}, got: {actual_rank}"
        )
    print("End execution")


if __name__ == "__main__":
    main()
