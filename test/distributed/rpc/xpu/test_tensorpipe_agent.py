#!/usr/bin/env python3
# Owner(s): ["oncall: distributed"]

import sys

import torch.distributed as dist


if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

import torch
from torch.testing._internal.common_utils import run_tests, IS_WINDOWS
from torch.testing._internal.distributed.rpc.tensorpipe_rpc_agent_test_fixture import (
    TensorPipeRpcAgentTestFixture,
)
from torch.testing._internal.distributed.rpc_utils import (
    generate_tests,
    GENERIC_DEVICE_TESTS,
    TENSORPIPE_DEVICE_TESTS,
)


if torch.xpu.is_available() and not IS_WINDOWS:
    torch._C._accelerator_setAllocatorSettings("expandable_segments:False")

globals().update(
    generate_tests(
        "XpuTensorPipe",
        TensorPipeRpcAgentTestFixture,
        GENERIC_DEVICE_TESTS + TENSORPIPE_DEVICE_TESTS,
        __name__,
    )
)


if __name__ == "__main__":
    run_tests()
