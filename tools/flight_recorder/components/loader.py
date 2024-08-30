# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import gc
import os
import pickle
import time
from typing import Any, Dict, List, Union


def read_dump(prefix: str, filename: str) -> Dict[str, Union[str, int, List[Any]]]:
    basename = os.path.basename(filename)
    assert (
        basename.find(prefix) == 0
    ), f"args.prefix ({prefix}) must match the beginning of each filename ({basename})"
    rank = int(basename[len(prefix) :])
    host_name = f"host_rank{rank}"

    with open(filename, "rb") as infile:
        dump = pickle.load(infile)

    entries = dump["entries"]
    version = dump["version"]
    pg_config = dump["pg_config"]

    return {
        "host_name": host_name,
        "rank": rank,
        "entries": entries,
        "version": version,
        "pg_config": pg_config,
    }


def read_dir(prefix: str, folder: str) -> Dict[str, Dict[str, Any]]:
    gc.disable()
    details = {}
    t0 = time.time()
    for root, _, files in os.walk(folder):
        for f in files:
            details[f] = read_dump(prefix, os.path.join(root, f))
    tb = time.time()
    print(f"loaded {len(files)} files in {tb - t0}s")
    return details
