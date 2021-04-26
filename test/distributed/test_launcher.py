import os
import sys
import unittest
from contextlib import closing

import torch.distributed as dist
import torch.distributed.launch as launch
from torch.distributed.elastic.utils import get_socket_with_port

if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

from torch.testing._internal.common_utils import (
    TEST_WITH_ASAN,
    TEST_WITH_TSAN,
    TestCase,
    run_tests,
)


def path(script):
    return os.path.join(os.path.dirname(__file__), script)


@unittest.skipIf(
    TEST_WITH_ASAN or TEST_WITH_TSAN,
    "Skip ASAN as torch + multiprocessing spawn have known issues",
)
class TestDistirbutedLaunch(TestCase):
    def test_launch_user_script(self):
        nnodes = 1
        nproc_per_node = 4
        world_size = nnodes * nproc_per_node
        sock = get_socket_with_port()
        with closing(sock):
            master_port = sock.getsockname()[1]
        args = [
            f"--nnodes={nnodes}",
            f"--nproc_per_node={nproc_per_node}",
            "--monitor_interval=1",
            "--start_method=fork",
            "--master_addr=localhost",
            f"--master_port={master_port}",
            "--node_rank=0",
            "--use_env",
            path("bin/test_script.py"),
        ]
        launch.main(args)


if __name__ == "__main__":
    run_tests()
