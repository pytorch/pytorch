# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import subprocess
import sys

from . import BENCHMARK_ROOT, s3_utils


TRACES = [
    "torchbench_traces.tar.gz",
]


def download_traces_from_s3():
    """Download trace to benchmarks/trace_analysis/.data"""
    for trace in TRACES:
        s3_utils.checkout_s3_data(trace, decompress=True)


def install_deps(requirements_txt="requirements.txt"):
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-r",
            os.path.join(BENCHMARK_ROOT, requirements_txt),
        ]
    )


if __name__ == "__main__":
    install_deps()
    download_traces_from_s3()
