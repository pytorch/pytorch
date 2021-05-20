#!/usr/bin/env python3

import os
import sys

# Force the Future class to always inspect its values and ensure they don't
# reside on unexpected devices. This only matters for CUDA tests, where it can
# catch missing CUDA synchronization early and clearly. We don't enable it all
# the time (yet?) because of perf concerns, and because it could introduce
# regressions when uninspectable types are used with CPU-only Futures.
os.environ["PYTORCH_FUTURE_ALWAYS_EXTRACT_DATAPTRS"] = "1"

import torch.distributed as dist

if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed.rpc.tensorpipe_rpc_agent_test_fixture import (
    TensorPipeRpcAgentTestFixture,
)
from torch.testing._internal.distributed.rpc_utils import (
    GENERIC_CUDA_TESTS,
    TENSORPIPE_CUDA_TESTS,
    MultiProcess,
    generate_tests,
)


globals().update(
    generate_tests(
        "TensorPipe",
        TensorPipeRpcAgentTestFixture,
        GENERIC_CUDA_TESTS + TENSORPIPE_CUDA_TESTS,
        MultiProcess.SPAWN,
        __name__,
    )
)


if __name__ == "__main__":
    run_tests()
