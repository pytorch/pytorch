# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import gc
import os
import pickle
import re
import time
import typing
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple, Union

from tools.flight_recorder.components.fr_logger import FlightRecorderLogger


logger: FlightRecorderLogger = FlightRecorderLogger()


def read_dump(prefix: str, filename: str) -> Dict[str, Union[str, int, List[Any]]]:
    basename = os.path.basename(filename)

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


exp = re.compile(r"([\w\-\_]*?)(\d+)$")


def _determine_prefix(files: List[str]) -> str:
    """If the user doesn't specify a prefix, but does pass a dir full of similarly-prefixed files, we should be able to
    infer the common prefix most of the time.  But if we can't confidently infer, just fall back to requring the user
    to specify it
    """
    possible_prefixes: typing.DefaultDict[str, Set[int]] = defaultdict(set)
    for f in files:
        m = exp.search(f)
        if m:
            p, r = m.groups()
            possible_prefixes[p].add(int(r))
    if len(possible_prefixes) == 1:
        prefix = next(iter(possible_prefixes))
        logger.debug("Inferred common prefix %s", prefix)
        return prefix
    else:
        raise ValueError(
            "Unable to automatically determine the common prefix for the trace file names. "
            "Please specify --prefix argument manually"
        )


def read_dir(args: argparse.Namespace) -> Tuple[Dict[str, Dict[str, Any]], str]:
    gc.disable()
    prefix = args.prefix
    details = {}
    t0 = time.time()
    version = ""
    filecount = 0
    assert os.path.isdir(args.trace_dir), f"folder {args.trace_dir} does not exist"
    for root, _, files in os.walk(args.trace_dir):
        if prefix is None:
            prefix = _determine_prefix(files)
        for f in files:
            if f.find(prefix) != 0:
                continue
            details[f] = read_dump(prefix, os.path.join(root, f))
            filecount += 1
            if not version:
                version = str(details[f]["version"])
    tb = time.time()
    assert (
        len(details) > 0
    ), f"no files loaded from {args.trace_dir} with prefix {prefix}"
    logger.debug("loaded %s files in %ss", filecount, tb - t0)
    return details, version
