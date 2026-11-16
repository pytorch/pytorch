#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test binary, raises a RuntimeError")
    parser.add_argument("--raises", type=bool, default=False)
    parser.add_argument("msg", type=str)
    args = parser.parse_args()

    rank = int(os.environ["RANK"])

    if args.raises:
        raise RuntimeError(f"raised from {rank}")
    else:
        print(f"{args.msg} from {rank}")
