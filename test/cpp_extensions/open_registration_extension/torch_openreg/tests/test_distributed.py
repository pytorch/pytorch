# Owner(s): ["module: PrivateUse1"]

import os
import sys
import tempfile


os.environ["BACKEND"] = "occl"
os.environ["WORLD_SIZE"] = "2"

# Provide a TEMP_DIR compatible with Barrier helpers if not already present.
if "TEMP_DIR" not in os.environ:
    _tmp_dir = tempfile.TemporaryDirectory()
    os.environ["TEMP_DIR"] = _tmp_dir.name
    os.makedirs(os.path.join(_tmp_dir.name, "barrier"), exist_ok=True)
    os.makedirs(os.path.join(_tmp_dir.name, "test_dir"), exist_ok=True)
    os.makedirs(os.path.join(_tmp_dir.name, "init_dir"), exist_ok=True)
    os.environ["INIT_METHOD"] = (
        f"file://{os.path.join(_tmp_dir.name, 'init_dir', 'shared_init_file')}"
    )


import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import _get_default_group
from torch.testing._internal.common_distributed import (
    cleanup_temp_dir,
    initialize_temp_directories,
    MultiProcessTestCase,
)
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed.distributed_test import (
    Barrier,
    DistributedTest,
    TestDistBackend,
)


if not (dist.is_available() and "occl" in dist.Backend.backend_list):
    print("torch.distributed OCCL backend unavailable, skipping tests", file=sys.stderr)
    sys.exit(0)


class TestProcessGroupOCCL(TestDistBackend, DistributedTest._DistTestBase):
    # OCCL collectives are dummy; we only validate registration and basic API.

    def setUp(self):
        # Bypass TestDistBackend.setUp (which expects many inherited tests) and
        # perform the minimal MultiProcessTestCase setup plus temp dir wiring.
        MultiProcessTestCase.setUp(self)
        initialize_temp_directories()
        Barrier.init()
        self.skip_return_code_checks = []
        # Spawn subprocesses for multi-process execution when running as main.
        if self.rank == self.MAIN_PROCESS_RANK:
            self._spawn_processes()

    def tearDown(self):
        cleanup_temp_dir()
        super().tearDown()

    def test_occl_backend_registration_and_default_group(self) -> None:
        # Backend is registered and available in the backend list.
        self.assertTrue(dist.is_available())
        self.assertIn("occl", dist.Backend.backend_list)

        # Default group should already be initialized for OCCL.
        self.assertTrue(dist.is_initialized())
        self.assertEqual(dist.get_backend(), "occl")

        pg = _get_default_group()
        self.assertEqual(dist.get_backend(pg), "occl")
        self.assertEqual(dist.get_world_size(), 2)
        self.assertEqual(dist.get_rank(), self.rank)

    def test_occl_allreduce(self) -> None:
        pg = _get_default_group()
        device = torch.device("openreg")

        tensors = [torch.rand(1, device=device)]
        fut = pg.allreduce(tensors).get_future()
        fut.wait()
        self.assertTrue(fut.done())

        # OCCL is currently a dummy backend and all communication methods return None.
        res = fut.value()
        self.assertEqual(res, None)


# Drop inherited torch distributed backend tests; we only need the OCCL smoke test.
_allowed_tests = {
    "test_occl_backend_registration_and_default_group",
    "test_occl_allreduce",
}
for _name in dir(TestProcessGroupOCCL):
    if _name.startswith("test_") and _name not in _allowed_tests:
        setattr(TestProcessGroupOCCL, _name, None)


if __name__ == "__main__":
    run_tests()
