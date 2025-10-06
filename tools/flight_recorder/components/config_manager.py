# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
from collections.abc import Sequence
from typing import Optional

from tools.flight_recorder.components.fr_logger import FlightRecorderLogger


logger: FlightRecorderLogger = FlightRecorderLogger()


class JobConfig:
    """
    A helper class to manage the script configuration.
    """

    def __init__(self: "JobConfig"):
        self.parser = argparse.ArgumentParser(
            description="PyTorch Flight recorder analyzing script."
        )
        self.parser.add_argument(
            "trace_dir",
            nargs="?",
            help="Directory containing one trace file per rank, named with <prefix>_<rank>.",
        )
        self.parser.add_argument(
            "--selected-ranks",
            default=None,
            nargs="+",
            type=int,
            help="List of ranks we want to show traces for.",
        )
        self.parser.add_argument(
            "--allow-incomplete-ranks",
            action="store_true",
            help=(
                "FR trace require all ranks to have dumps for analysis. "
                "This flag allows best-effort partial analysis of results "
                "and printing of collected data."
            ),
        )
        self.parser.add_argument(
            "--pg-filters",
            default=None,
            nargs="+",
            type=str,
            help=(
                "List of filter strings, it could be pg name or pg desc. "
                "If specified, only show traces for the given pg."
            ),
        )
        self.parser.add_argument("-o", "--output", default=None)
        self.parser.add_argument(
            "-p",
            "--prefix",
            help=(
                "Common filename prefix to strip such that rank can be extracted. "
                "If not specified, will attempt to infer a common prefix."
            ),
            default=None,
        )
        self.parser.add_argument("-j", "--just_print_entries", action="store_true")
        self.parser.add_argument("-v", "--verbose", action="store_true")
        self.parser.add_argument("--print_stack_trace", action="store_true")
        self.parser.add_argument(
            "--mismatch_cap",
            type=int,
            default=10,
            help="Maximum number of mismatches we print (from earliest).",
        )

    def parse_args(
        self: "JobConfig", args: Optional[Sequence[str]]
    ) -> argparse.Namespace:
        args = self.parser.parse_args(args)
        if args.selected_ranks is not None:
            assert args.just_print_entries, (
                "Not support selecting ranks without printing entries"
            )
        if args.pg_filters is not None:
            assert args.just_print_entries, (
                "Not support selecting pg filters without printing entries"
            )
        if args.verbose:
            logger.set_log_level(logging.DEBUG)
        return args
