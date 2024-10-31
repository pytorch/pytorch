#!/usr/bin/env python3
"""Flight Recorder Trace Analyzer

This script primarily merges data from individual flight recorder buffers from individual ranks in a
PyTorch Distributed program into a flattened database format that can be used for further analysis.

However as part of the merging process, it is necessary to perform some analysis in order to match operators
on one rank with corresponding operators on other ranks and register them as one 'collective' entry.  During this
process, a significant amount of useful information can already be extracted such as where the first mismatch occurs
in cases of desync (when not all ranks issue a compatible collective in a particular process group).


Not Yet Implemented
- TODO- tracebacks aren't implemented

Known Issues
- Flight Recorder buffer sequence_id information is not sufficient to match collectives and coalseced collectives
  unless we have the trace data from the beginning of the program.  To enable confident analysis of trace buffers that
  do not start from zero (and to simplify the script's matching logic) we need to add more information to the recorder.
- Currently, the script omits checking the 'status' of collectives.  We can look for the first 'non completed'
  collective easily enough and report that.

Usage
python fr_trace.py <dump dir containing trace files> [-o <output file>]

- Omitting the optional output file will still yield analysis information to stdout
- The output file is a pickle of the flat DB, which may change in format in the future.
- This script is versioned so that we can ensure our future changes to flight recorder are backwards compatible.
"""

import pickle
from typing import Optional, Sequence

from tools.flight_recorder.components.builder import build_db
from tools.flight_recorder.components.config_manager import JobConfig
from tools.flight_recorder.components.loader import read_dir
from tools.flight_recorder.components.types import types


def main(args: Optional[Sequence[str]] = None) -> None:
    config = JobConfig()
    args = config.parse_args(args)
    assert args.trace_dir, "Trace directory trace_dir is required"
    details, version = read_dir(args.prefix, args.trace_dir)
    db = build_db(details, args, version)
    if args.output:
        with open(args.output, "wb") as f:
            pickle.dump((types, db), f)


if __name__ == "__main__":
    main()
