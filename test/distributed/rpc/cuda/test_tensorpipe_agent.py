#!/usr/bin/env python3
# Owner(s): ["oncall: distributed"]

import sys
import unittest

import torch.distributed as dist


if not dist.is_available():
    raise unittest.SkipTest("Distributed not available, skipping tests")

import torch
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed.rpc.tensorpipe_rpc_agent_test_fixture import (
    TensorPipeRpcAgentTestFixture,
)
from torch.testing._internal.distributed.rpc_utils import (
    generate_tests,
    GENERIC_CUDA_TESTS,
    TENSORPIPE_CUDA_TESTS,
)


if torch.cuda.is_available():
    torch.cuda.memory._set_allocator_settings("expandable_segments:False")

globals().update(
    generate_tests(
        "TensorPipe",
        TensorPipeRpcAgentTestFixture,
        GENERIC_CUDA_TESTS + TENSORPIPE_CUDA_TESTS,
        __name__,
    )
)


if __name__ == "__main__":
    run_tests()
