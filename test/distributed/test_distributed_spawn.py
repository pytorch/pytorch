# Owner(s): ["oncall: distributed"]

import os
import sys


if os.environ.get("BACKEND") == "nccl":
    os.environ["TORCH_DIST_USE_NCCL2"] = "0"

import torch
import torch.distributed as dist


_PRIOR_FP32_PRECISION: tuple[str, ...] | None = None


def setUpModule():
    global _PRIOR_FP32_PRECISION
    # allow_tf32 writes both the legacy Float32MatmulPrecision enum and the
    # backend-specific fp32_precision, so snapshot and restore all of it.
    _PRIOR_FP32_PRECISION = _snapshot_fp32_precision()
    torch.backends.cuda.matmul.allow_tf32 = False


def tearDownModule():
    global _PRIOR_FP32_PRECISION
    if _PRIOR_FP32_PRECISION is not None:
        _restore_fp32_precision(_PRIOR_FP32_PRECISION)
        _PRIOR_FP32_PRECISION = None


if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

from torch.testing._internal.common_utils import (
    _restore_fp32_precision,
    _snapshot_fp32_precision,
    HardwareClassification,
    run_tests,
    TEST_WITH_DEV_DBG_ASAN,
)
from torch.testing._internal.common_device_type import instantiate_device_type_tests
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

# Entrance gating only: the current accelerator's device type is probed inline
# at import time (before the instantiation framework generates per-device
# variants) solely to accept its registered communication backends — tests
# themselves always use the framework-injected device type. Third-party
# backends advertise their supported devices in the public
# Backend.backend_capability map (populated via Backend.register_backend).
_allowed_backends = ("gloo", "nccl", "ucc") + tuple(
    backend
    for backend, devices in dist.Backend.backend_capability.items()
    if (acc := torch.accelerator.current_accelerator()) is not None
    and acc.type in devices
    and backend not in ("gloo", "nccl", "ucc")
)
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


def _backend_supports_device(backend, device_type):
    # Authoritative device support matrix from Backend.backend_capability
    # (e.g. gloo supports cpu/cuda; accelerator vendors register their own
    # backends with the devices they support).
    return device_type in dist.Backend.backend_capability.get(backend, ())


if BACKEND in _allowed_backends:

    class TestDistBackendWithSpawn(TestDistBackend, DistributedTest._DistTestBase):
        hw_classification = HardwareClassification.GENERIC

        def setUp(self):
            super().setUp()
            self._spawn_processes()

    class TestDistBackendWithSpawnAccel(
        TestDistBackend, DistributedTest._AcceleratorDistTestBase
    ):
        hw_classification = HardwareClassification.ACCELERATOR

        def setUp(self):
            super().setUp()
            if not _backend_supports_device(BACKEND, self.device_type):
                self.skipTest(
                    f"The {BACKEND} backend does not support {self.device_type} tensors"
                )
            self._spawn_processes()
            if self.device_type == "cuda":
                torch.backends.cudnn.flags(enabled=True, allow_tf32=False).__enter__()

    class TestDistBackendWithSpawnNccl(
        TestDistBackend, DistributedTest._NcclDistTestBase
    ):
        hw_classification = HardwareClassification.CUDA

        def setUp(self):
            super().setUp()
            self._spawn_processes()
            torch.backends.cudnn.flags(enabled=True, allow_tf32=False).__enter__()

    # instantiate_device_type_tests only inspects the generic class's own
    # __dict__, so promote the tests inherited from _AcceleratorDistTestBase
    # into TestDistBackendWithSpawnAccel before instantiation. The original
    # entries are removed from the base so the un-wrapped (un-suffixed) tests
    # are not collected twice via the variant MRO.
    _accel_base = DistributedTest._AcceleratorDistTestBase
    for _name in [
        _n for _n, _fn in _accel_base.__dict__.items() if _n.startswith("test_")
    ]:
        setattr(TestDistBackendWithSpawnAccel, _name, getattr(_accel_base, _name))
        delattr(_accel_base, _name)

    # Instantiates per-accelerator variants (e.g. ...AccelCUDA / ...AccelPRIVATEUSE1)
    # of the device-injected tests; CPU is covered by the GENERIC class above.
    instantiate_device_type_tests(
        TestDistBackendWithSpawnAccel, globals(), except_for=["cpu"], allow_xpu=True
    )

else:
    print(f"Invalid backend {BACKEND}. Tests will not be run!")


if __name__ == "__main__":
    run_tests()
