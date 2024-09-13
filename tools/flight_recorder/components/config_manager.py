# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from typing import Optional, Sequence


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
            "--pg-filters",
            default=None,
            nargs="+",
            type=str,
            help="List of filter strings",
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

    def parse_args(
        self: "JobConfig", args: Optional[Sequence[str]]
    ) -> argparse.Namespace:
        args = self.parser.parse_args(args)
        if args.selected_ranks is not None:
            assert (
                args.just_print_entries
            ), "Not support selecting ranks without printing entries"
        if args.pg_filters is not None:
            assert (
                args.just_print_entries
            ), "Not support selecting pg filters without printing entries"
        return args
