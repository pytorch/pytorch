# Owner(s): ["oncall: distributed"]

import os
import sys
from contextlib import closing

import torch.distributed as dist
import torch.distributed.launch as launch
from torch.distributed.elastic.utils import get_socket_with_port

if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

from torch.testing._internal.common_utils import (
    TEST_WITH_DEV_DBG_ASAN,
    TestCase,
    run_tests,
)


def path(script):
    return os.path.join(os.path.dirname(__file__), script)


if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip ASAN as torch + multiprocessing spawn have known issues", file=sys.stderr
    )
    sys.exit(0)


class TestDistributedLaunch(TestCase):
    def test_launch_user_script(self):
        nnodes = 1
        nproc_per_node = 4
        world_size = nnodes * nproc_per_node
        sock = get_socket_with_port()
        with closing(sock):
            master_port = sock.getsockname()[1]
        args = [
            f"--nnodes={nnodes}",
            f"--nproc-per-node={nproc_per_node}",
            "--monitor-interval=1",
            "--start-method=spawn",
            "--master-addr=localhost",
            f"--master-port={master_port}",
            "--node-rank=0",
            "--use-env",
            path("bin/test_script.py"),
        ]
        launch.main(args)


if __name__ == "__main__":
    run_tests()
