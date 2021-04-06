#!/usr/bin/env python3

import sys
import torch.distributed as dist

if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed.rpc.faulty_rpc_agent_test_fixture import (
    FaultyRpcAgentTestFixture,
)
from torch.testing._internal.distributed.rpc_utils import (
    FAULTY_AGENT_TESTS,
    MultiProcess,
    generate_tests,
)


globals().update(
    generate_tests(
        "Faulty",
        FaultyRpcAgentTestFixture,
        FAULTY_AGENT_TESTS,
        MultiProcess.SPAWN,
        __name__,
    )
)


if __name__ == "__main__":
    run_tests()
