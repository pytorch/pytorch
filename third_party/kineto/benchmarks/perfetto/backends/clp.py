# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse

from .common import TraceAnalysis


class CLPTraceAnalysis(TraceAnalysis):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
