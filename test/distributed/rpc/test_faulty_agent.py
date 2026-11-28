#!/usr/bin/env python3
# Owner(s): ["oncall: distributed"]

import sys

import torch
import torch.distributed as dist


if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

from torch.testing._internal.common_utils import IS_CI, run_tests, serialTestFile
from torch.testing._internal.distributed.rpc.faulty_rpc_agent_test_fixture import (
    FaultyRpcAgentTestFixture,
)
from torch.testing._internal.distributed.rpc_utils import (
    FAULTY_AGENT_TESTS,
    generate_tests,
)


# No other test runs beside those of this file: test/run_test.py runs every file with
# "distributed" in its name so.
pytestmark = serialTestFile()


# On CircleCI these tests are already run on CPU jobs, thus to save resources do
# not run them on GPU jobs, since they wouldn't provide additional test signal.
if not (IS_CI and torch.cuda.is_available()):
    globals().update(
        generate_tests(
            "Faulty",
            FaultyRpcAgentTestFixture,
            FAULTY_AGENT_TESTS,
            __name__,
        )
    )


if __name__ == "__main__":
    run_tests()
