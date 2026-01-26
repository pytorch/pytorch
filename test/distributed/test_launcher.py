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
    run_tests,
    TEST_WITH_DEV_DBG_ASAN,
    TestCase,
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


class TestLauncherApi(TestCase):
    def test_get_entrypoint_name_with_empty_strings(self):
        """
        Test that _get_entrypoint_name handles empty strings in args list.
        Previously, accessing arg[0] on an empty string would raise IndexError.
        """
        from torch.distributed.launcher.api import _get_entrypoint_name

        result = _get_entrypoint_name(sys.executable, ["", "-u", "script.py"])
        self.assertEqual(result, "script.py")

        result = _get_entrypoint_name(sys.executable, ["", "", "another.py"])
        self.assertEqual(result, "another.py")

        result = _get_entrypoint_name(sys.executable, ["", ""])
        self.assertEqual(result, "")

        result = _get_entrypoint_name(sys.executable, ["-u", "", "test.py"])
        self.assertEqual(result, "test.py")


if __name__ == "__main__":
    run_tests()
