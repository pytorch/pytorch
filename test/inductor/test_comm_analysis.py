# Owner(s): ["module: inductor"]

import sys
import unittest
import warnings

import torch
import torch.distributed as dist


try:
    from torch.testing._internal.common_distributed import requires_accelerator_dist_backend
except ImportError:
    print("common_distributed not importable, skipping tests", file=sys.stderr)
    sys.exit(0)

from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    HardwareClassification,
    run_tests,
    TestCase,
    TEST_PRIVATEUSE1,
)


def _get_all_gather_node(group_size, group_name):
    """Trace a simple all_gather function and return the collective FX node."""

    def func(inp, group_size, group_name):
        out = torch.ops._c10d_functional.all_gather_into_tensor(
            inp, group_size, group_name
        )
        wait = torch.ops._c10d_functional.wait_tensor(out)
        return wait

    gm = make_fx(func)(torch.ones(4, 4), group_size, group_name)
    for n in gm.graph.nodes:
        if n.op == "call_function" and "all_gather_into_tensor" in str(n.target):
            return n
    raise RuntimeError("No all_gather_into_tensor node found in traced graph")


class TestNcclEstimateDeviceResolution(TestCase):
    """
    Tests for the device resolution fix in _nccl_estimate() inside
    estimate_nccl_collective_runtime_from_fx_node.
    """
    hw_classification = HardwareClassification.ACCELERATOR

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        backend = dist.get_default_backend_for_device(cls.device_type)
        if not dist.is_available() or not dist.is_backend_available(backend):
            raise unittest.SkipTest(f"c10d {backend} not available, skipping tests")

    def _init_pg(self, backend, world_size=2):
        from torch.testing._internal.distributed.fake_pg import FakeStore

        store = FakeStore()
        dist.init_process_group(
            backend=backend, rank=0, world_size=world_size, store=store
        )
        pg = dist.group.WORLD
        group_name = "test_comm_analysis"
        torch._C._distributed_c10d._register_process_group(group_name, pg)
        return pg, group_name, pg.size()

    def _init_pg_real_store(self, backend, world_size=1):
        store = dist.HashStore()
        dist.init_process_group(
            backend=backend, rank=0, world_size=world_size, store=store
        )
        pg = dist.group.WORLD
        group_name = "test_comm_analysis"
        torch._C._distributed_c10d._register_process_group(group_name, pg)
        return pg, group_name, pg.size()

    def _destroy_pg(self):
        dist.destroy_process_group()

    @unittest.skipif(TEST_PRIVATEUSE1, "PU1 not support")
    def test_fake_backend_falls_back_to_analytical(self, device):
        """FAKE backend: _nccl_estimate returns None, falls back to analytical formula."""
        pg, group_name, group_size = self._init_pg("fake")
        try:
            node = _get_all_gather_node(group_size, group_name)
            from torch._inductor.comm_analysis import (
                estimate_nccl_collective_runtime_from_fx_node,
            )

            est_ms = estimate_nccl_collective_runtime_from_fx_node(
                node, use_nccl_estimator=True
            )
            self.assertGreater(est_ms, 0)

            est_ms_analytical = estimate_nccl_collective_runtime_from_fx_node(
                node, use_nccl_estimator=False
            )
            self.assertEqual(est_ms, est_ms_analytical)
        finally:
            self._destroy_pg()

    @requires_accelerator_dist_backend()
    def test_multi_backend_pg_resolves_to_xccl(self, device):
        """
        Multi-backend PG ("cpu:gloo,cuda:nccl"): We should resolve to the cuda device's backend.
        """
        torch.accelerator.set_device_index(0)
        backend_str = dist.get_default_backend_for_device(device)
        multi_backend = f"cpu:gloo,{torch.device(device).type}:{backend_str}"
        pg, group_name, group_size = self._init_pg_real_store(multi_backend)
        try:
            from torch.distributed.distributed_c10d import _get_pg_default_device

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                default_device = _get_pg_default_device(pg)
            self.assertEqual(default_device, torch.device("cpu"))

            dist_backend = pg._get_backend(torch.device(device))
            self.assertTrue(dist_backend._supports_time_estimate)

            gloo_backend = pg._get_backend(torch.device("cpu"))
            self.assertFalse(gloo_backend._supports_time_estimate)
        finally:
            self._destroy_pg()

    @requires_accelerator_dist_backend()
    def test_single_xccl_backend_resolves_correctly(self, device):
        """Single NCCL backend PG: cuda device resolves to NCCL with time estimation."""
        torch.accelerator.set_device_index(0)
        backend_str = dist.get_default_backend_for_device(device)
        pg, group_name, group_size = self._init_pg_real_store(backend_str)
        try:
            backend = pg._get_backend(torch.device(device))
            self.assertTrue(backend._supports_time_estimate)
        finally:
            self._destroy_pg()


instantiate_device_type_tests(
    TestNcclEstimateDeviceResolution, globals(), except_for="cpu"
)

if __name__ == "__main__":
    run_tests()
