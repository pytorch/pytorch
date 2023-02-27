#!/usr/bin/env python3
# Owner(s): ["oncall: r2p"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="test script")

    parser.add_argument(
        "--fail",
        default=False,
        action="store_true",
        help="forces the script to throw a RuntimeError",
    )

    # file is used for assertions
    parser.add_argument(
        "--touch-file-dir",
        "--touch_file_dir",
        type=str,
        help="dir to touch a file with global rank as the filename",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    env_vars = [
        "LOCAL_RANK",
        "RANK",
        "GROUP_RANK",
        "ROLE_RANK",
        "ROLE_NAME",
        "LOCAL_WORLD_SIZE",
        "WORLD_SIZE",
        "ROLE_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "TORCHELASTIC_RESTART_COUNT",
        "TORCHELASTIC_MAX_RESTARTS",
        "TORCHELASTIC_RUN_ID",
        "OMP_NUM_THREADS",
        "TEST_SENTINEL_PARENT",
        "TORCHELASTIC_ERROR_FILE",
    ]

    print("Distributed env vars set by agent:")
    for env_var in env_vars:
        value = os.environ[env_var]
        print(f"{env_var} = {value}")

    if args.fail:
        raise RuntimeError("raising exception since --fail flag was set")
    else:
        file = os.path.join(args.touch_file_dir, os.environ["RANK"])
        Path(file).touch()
        print(f"Success, created {file}")


if __name__ == "__main__":
    main()
