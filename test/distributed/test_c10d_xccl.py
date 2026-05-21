# Owner(s): ["module: distributed"]

import os
import socket
import sys
import warnings
from datetime import timedelta

import torch
import torch.distributed as c10d


if not c10d.is_available() or not c10d.is_xccl_available():
    print("c10d XCCL not available, skipping tests", file=sys.stderr)
    sys.exit(0)

import torch.distributed as dist
from torch.testing._internal.common_distributed import (
    requires_accelerator_dist_backend,
)
from torch.testing._internal.common_utils import (
    run_tests,
    TestCase,
)


def _free_port() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return str(s.getsockname()[1])


class ProcessGroupXCCLTest(TestCase):
    @requires_accelerator_dist_backend(["xccl"])
    def test_xccl_set_pg_timeout_api(self):
        """
        Test _set_pg_timeout API for XCCL backend.

        Prior to this fix, ``_set_pg_timeout`` had device-conditional
        branches only for cpu (gloo) and cuda (nccl/gloo/torchcomms);
        an xpu/xccl process group fell through to the "Set timeout is
        now only supported for either nccl or gloo." warning and the
        per-PG timeout was silently never propagated to the backend.

        This test verifies:
        1. ``_set_pg_timeout`` on an xccl PG does NOT emit the warning;
        2. the call reaches ``ProcessGroupXCCL.set_timeout`` and
           updates ``backend.options._timeout`` to the new value.
        """
        if not torch.xpu.is_available():
            self.skipTest("requires Intel XPU")

        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", _free_port())
        os.environ["WORLD_SIZE"] = "1"
        os.environ["RANK"] = "0"
        torch.xpu.set_device(0)

        dist.init_process_group(
            backend="xccl",
            init_method="env://",
            timeout=timedelta(seconds=50),
        )
        try:
            pg = dist.distributed_c10d._get_default_group()
            backend = pg._get_backend(torch.device("xpu"))
            self.assertEqual(backend.options._timeout, timedelta(seconds=50))

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                c10d.distributed_c10d._set_pg_timeout(
                    timedelta(milliseconds=1), pg
                )

            unsupported_warnings = [
                w
                for w in caught
                if "Set timeout is now only supported" in str(w.message)
            ]
            self.assertEqual(
                unsupported_warnings,
                [],
                "_set_pg_timeout on an xccl PG should not emit the "
                "'only supported for nccl or gloo' warning",
            )
            self.assertEqual(
                backend.options._timeout, timedelta(milliseconds=1)
            )
        finally:
            dist.destroy_process_group()


if __name__ == "__main__":
    run_tests()
