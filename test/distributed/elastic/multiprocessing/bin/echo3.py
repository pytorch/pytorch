#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import ctypes
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="test binary, triggers a segfault (SIGSEGV)"
    )
    parser.add_argument("--segfault", type=bool, default=False)
    parser.add_argument("msg", type=str)
    args = parser.parse_args()

    rank = int(os.environ["RANK"])

    if args.segfault:
        ctypes.string_at(0)
    else:
        print(f"{args.msg} from {rank}")
