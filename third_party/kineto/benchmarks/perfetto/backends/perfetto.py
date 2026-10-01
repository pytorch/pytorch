# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from typing import List

from perfetto.trace_processor import TraceProcessor

from .common import TraceAnalysis


class PerfettoTraceAnalysis(TraceAnalysis):
    name = "perfetto"

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)

    def load(self, input_file_path: str):
        self.tp = TraceProcessor(input_file_path)

    def search_gemm_kernels(self) -> List[str]:
        query = "SELECT DISTINCT(name) FROM slice WHERE name like '%sm90_xmma_gemm_%' ORDER BY ts"
        query_result = [str(x) for x in self.tp.query(query)]
        return query_result

    def select_kernels(self):
        query = "SELECT ts, dur, name FROM slice WHERE category == 'kernel' ORDER BY ts limit 30"
        query_result = [str(x) for x in self.tp.query(query)]
        return query_result

    def group_kernels(self):
        query = "SELECT name, sum(dur), avg(dur), count(*) as occ FROM slice WHERE category == 'kernel' GROUP BY name ORDER BY occ DESC"

        query_result = [str(x) for x in self.tp.query(query)]
        return query_result
