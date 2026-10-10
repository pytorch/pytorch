# Owner(s): ["module: inductor"]

import sys
import unittest
import warnings

import torch
import torch.distributed as dist


if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

try:
    from torch.testing._internal.common_distributed import (
        requires_accelerator_dist_backend,
    )
except ImportError:
    print("common_distributed not importable, skipping tests", file=sys.stderr)
    sys.exit(0)

from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_utils import run_tests, skipIfXpu, TestCase


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"
device_module = torch.get_device_module(device_type)


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

    def test_fake_backend_falls_back_to_analytical(self):
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

    @requires_accelerator_dist_backend(["nccl", "xccl"])
    @unittest.skipUnless(torch.accelerator.is_available(), "requires an accelerator")
    @skipIfXpu(
        msg="XCCL does not implement supportsTimeEstimation(): "
        "https://github.com/intel/torch-xpu-ops/issues/5405"
    )
    def test_multi_backend_pg_resolves_to_nccl(self):
        """
        Multi-backend PG ("cpu:gloo,cuda:nccl"): We should resolve to the accelerator's backend.
        """
        device_module.set_device(0)
        accel_backend = dist.get_default_backend_for_device(device_type)
        pg, group_name, group_size = self._init_pg_real_store(
            f"cpu:gloo,{device_type}:{accel_backend}"
        )
        try:
            from torch.distributed.distributed_c10d import _get_pg_default_device

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                default_device = _get_pg_default_device(pg)
            self.assertEqual(default_device, torch.device("cpu"))

            nccl_backend = pg._get_backend(torch.device(device_type))
            self.assertTrue(nccl_backend._supports_time_estimate)

            gloo_backend = pg._get_backend(torch.device("cpu"))
            self.assertFalse(gloo_backend._supports_time_estimate)
        finally:
            self._destroy_pg()

    @requires_accelerator_dist_backend(["nccl", "xccl"])
    @unittest.skipUnless(torch.accelerator.is_available(), "requires an accelerator")
    @skipIfXpu(
        msg="XCCL does not implement supportsTimeEstimation(): "
        "https://github.com/intel/torch-xpu-ops/issues/5405"
    )
    def test_single_nccl_backend_resolves_correctly(self):
        """Single NCCL backend PG: accelerator device resolves to a backend with time estimation."""
        device_module.set_device(0)
        pg, group_name, group_size = self._init_pg_real_store(
            dist.get_default_backend_for_device(device_type)
        )
        try:
            backend = pg._get_backend(torch.device(device_type))
            self.assertTrue(backend._supports_time_estimate)
        finally:
            self._destroy_pg()


if __name__ == "__main__":
    run_tests()
