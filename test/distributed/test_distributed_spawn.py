# Owner(s): ["oncall: distributed"]

import os
import sys

import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import Backend

_PRIOR_FP32_PRECISION: bool | str | None = None


def setUpModule():
    global _PRIOR_FP32_PRECISION
    device_type = getattr(torch.accelerator.current_accelerator(), "type", "cpu")
    device_module = torch.get_device_module(device_type)
    if hasattr(device_module, "matmul") and hasattr(device_module.matmul, "fp32_precision"):
        _PRIOR_FP32_PRECISION = device_module.matmul.fp32_precision
        device_module.matmul.allow_tf32 = False


def tearDownModule():
    global _PRIOR_FP32_PRECISION
    device_type = getattr(torch.accelerator.current_accelerator(), "type", "cpu")
    device_module = torch.get_device_module(device_type)
    if _PRIOR_FP32_PRECISION is not None and hasattr(device_module, "matmul") and hasattr(device_module.matmul, "fp32_precision"):
        device_module.matmul.fp32_precision = _PRIOR_FP32_PRECISION
        _PRIOR_FP32_PRECISION = None


if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

from torch.testing._internal.common_utils import HardwareClassification, run_tests, TEST_WITH_DEV_DBG_ASAN
from torch.testing._internal.distributed.distributed_test import (
    DistributedTest,
    TestDistBackend,
)


if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)

_allowed_backends = tuple(Backend.backend_list)

if (
    "BACKEND" not in os.environ
    or "WORLD_SIZE" not in os.environ
    or "TEMP_DIR" not in os.environ
):
    # TODO can we actually have `run_tests.py` emit the complete instructions when it prints a repro command?
    raise RuntimeError(
        "Missing expected env vars for `test_distributed_spawn.py`.  Please ensure to specify the following:\n"
        f"'BACKEND' = one of {_allowed_backends}\n"
        f"'WORLD_SIZE' = int >= 2\n"
        "'TEMP_DIR' specifying a directory containing a barrier file named 'barrier'.\n\n"
        f"e.g.\ntouch /tmp/barrier && TEMP_DIR=/tmp BACKEND='nccl' WORLD_SIZE=2 python {__file__}",
    )

BACKEND = os.environ["BACKEND"]

if BACKEND in _allowed_backends:

    class TestDistBackendWithSpawn(TestDistBackend, DistributedTest._DistTestBase):
        hw_classification = HardwareClassification.GENERIC

        def setUp(self):
            super().setUp()
            self._spawn_processes()
            device_type = getattr(torch.accelerator.current_accelerator(), "type", "cpu")
            device_module = torch.get_device_module(device_type)
            if hasattr(device_module, "cudnn") and device_module.is_available():
                device_module.cudnn.flags(enabled=True, allow_tf32=False).__enter__()

else:
    print(f"Invalid backend {BACKEND}. Tests will not be run!")


if __name__ == "__main__":
    run_tests()
