# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from .clp import CLPTraceAnalysis
from .common import DEFAULT_METRICS  # noqa: F401
from .perfetto import PerfettoTraceAnalysis

AVAILABLE_BACKENDS = {
    "perfetto": PerfettoTraceAnalysis,
    "clp": CLPTraceAnalysis,
}

AVAILABLE_TASKS = [
    "load",
    "search_gemm_kernels",
    "select_kernels",
    "group_kernels",
]

for name in AVAILABLE_BACKENDS:
    analysis = AVAILABLE_BACKENDS[name]
    analysis.name = name
