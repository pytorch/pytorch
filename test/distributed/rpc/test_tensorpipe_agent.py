#!/usr/bin/env python3
# Owner(s): ["oncall: distributed"]

import sys

import torch
import torch.distributed as dist


if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

from torch.testing._internal.common_utils import (
    HardwareClassification,
    IS_CI,
    run_tests,
)
from torch.testing._internal.distributed.rpc.tensorpipe_rpc_agent_test_fixture import (
    TensorPipeRpcAgentTestFixture,
)
from torch.testing._internal.distributed.rpc_utils import (
    generate_tests,
    GENERIC_TESTS,
    TENSORPIPE_TESTS,
)


# On CircleCI these tests are already run on CPU jobs, thus to save resources do
# not run them on GPU jobs, since they wouldn't provide additional test signal.
if not (IS_CI and torch.accelerator.is_available()):
    _generated_tests = generate_tests(
        "TensorPipe",
        TensorPipeRpcAgentTestFixture,
        GENERIC_TESTS + TENSORPIPE_TESTS,
        __name__,
    )
    for _cls in _generated_tests.values():
        _cls.hw_classification = HardwareClassification.CPU
    globals().update(_generated_tests)


if __name__ == "__main__":
    run_tests()
