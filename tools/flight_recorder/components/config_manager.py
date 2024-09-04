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
            "-d", "--dir", help="Directory with flight recorder dumps"
        )
        self.parser.add_argument("-o", "--output", default=None)
        self.parser.add_argument(
            "-p",
            "--prefix",
            help="prefix to strip such that rank can be extracted",
            default="rank_",
        )
        self.parser.add_argument("-j", "--just_print_entries", action="store_true")
        self.parser.add_argument("-v", "--verbose", action="store_true")

    def parse_args(
        self: "JobConfig", args: Optional[Sequence[str]]
    ) -> argparse.Namespace:
        return self.parser.parse_args(args)
