# Owner(s): ["module: inductor"]

import sys
import unittest
import warnings

import torch
import torch.distributed as dist


if not dist.is_available() or not dist.is_nccl_available():
    print("c10d NCCL not available, skipping tests", file=sys.stderr)
    sys.exit(0)

try:
    from torch.testing._internal.common_distributed import requires_nccl
except ImportError:
    print("common_distributed not importable, skipping tests", file=sys.stderr)
    sys.exit(0)

from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import run_tests, TestCase


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

    @requires_nccl()
    @unittest.skipUnless(TEST_CUDA, "requires CUDA")
    def test_multi_backend_pg_resolves_to_nccl(self):
        """
        Multi-backend PG ("cpu:gloo,cuda:nccl"): We should resolve to the cuda device's backend.
        """
        torch.cuda.set_device(0)
        pg, group_name, group_size = self._init_pg_real_store("cpu:gloo,cuda:nccl")
        try:
            from torch.distributed.distributed_c10d import _get_pg_default_device

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                default_device = _get_pg_default_device(pg)
            self.assertEqual(default_device, torch.device("cpu"))

            nccl_backend = pg._get_backend(torch.device("cuda"))
            self.assertTrue(nccl_backend._supports_time_estimate)

            gloo_backend = pg._get_backend(torch.device("cpu"))
            self.assertFalse(gloo_backend._supports_time_estimate)
        finally:
            self._destroy_pg()

    @requires_nccl()
    @unittest.skipUnless(TEST_CUDA, "requires CUDA")
    def test_single_nccl_backend_resolves_correctly(self):
        """Single NCCL backend PG: cuda device resolves to NCCL with time estimation."""
        torch.cuda.set_device(0)
        pg, group_name, group_size = self._init_pg_real_store("nccl")
        try:
            backend = pg._get_backend(torch.device("cuda"))
            self.assertTrue(backend._supports_time_estimate)
        finally:
            self._destroy_pg()


class TestCollectiveCostEstimatorRegistry(TestCase):
    """Tests for backend-specific collective cost estimator registration."""

    def test_registered_estimator_takes_precedence(self):
        from torch._inductor.comm_analysis import (
            NCCL_COLL,
            estimate_nccl_collective_runtime_impl,
            register_collective_cost_estimator,
        )

        register_collective_cost_estimator("cuda", lambda *args: 42.0)
        try:
            self.assertEqual(
                estimate_nccl_collective_runtime_impl(
                    1024, 8, NCCL_COLL.ALL_REDUCE, device_type="cuda"
                ),
                42.0,
            )
        finally:
            register_collective_cost_estimator("cuda", None)
        # After unregistering, the built-in analytical model is used again.
        self.assertGreater(
            estimate_nccl_collective_runtime_impl(
                1024, 8, NCCL_COLL.ALL_REDUCE, device_type="cuda"
            ),
            0,
        )

    def test_trivial_inputs_short_circuit_before_registry(self):
        # group_size <= 1 and UNSUPPORTED collectives return 0 without
        # consulting the registry.
        from torch._inductor.comm_analysis import (
            NCCL_COLL,
            estimate_nccl_collective_runtime_impl,
            register_collective_cost_estimator,
        )

        calls = []

        def estimator(tensor_storage_size_bytes, group_size, coll):  # type: ignore[no-untyped-def]
            calls.append((tensor_storage_size_bytes, group_size, coll))
            return 42.0

        register_collective_cost_estimator("cuda", estimator)
        try:
            self.assertEqual(
                estimate_nccl_collective_runtime_impl(
                    1024, 1, NCCL_COLL.ALL_REDUCE, device_type="cuda"
                ),
                0,
            )
            self.assertEqual(
                estimate_nccl_collective_runtime_impl(
                    1024, 8, NCCL_COLL.UNSUPPORTED, device_type="cuda"
                ),
                0,
            )
        finally:
            register_collective_cost_estimator("cuda", None)
        self.assertEqual(calls, [])

    def test_unregistered_non_cuda_is_unsupported(self):
        # Device types without a registered estimator have no calibrated cost
        # model: estimation returns None so callers can disable cost-based
        # optimizations until a backend-specific estimator is registered.
        from torch._inductor.comm_analysis import (
            NCCL_COLL,
            estimate_nccl_collective_runtime_impl,
            has_collective_cost_model,
        )

        self.assertIsNone(
            estimate_nccl_collective_runtime_impl(
                1024, 8, NCCL_COLL.ALL_REDUCE, device_type="npu"
            )
        )
        self.assertFalse(has_collective_cost_model("npu"))
        # CUDA keeps the built-in NCCL-calibrated analytical model.
        self.assertTrue(has_collective_cost_model("cuda"))
        self.assertIsNotNone(
            estimate_nccl_collective_runtime_impl(
                1024, 8, NCCL_COLL.ALL_REDUCE, device_type="cuda"
            )
        )

    def test_has_collective_cost_model_with_registered_estimator(self):
        from torch._inductor.comm_analysis import (
            has_collective_cost_model,
            register_collective_cost_estimator,
        )

        self.assertFalse(has_collective_cost_model("npu"))
        register_collective_cost_estimator("npu", lambda *args: 1.0)
        try:
            self.assertTrue(has_collective_cost_model("npu"))
        finally:
            register_collective_cost_estimator("npu", None)
        self.assertFalse(has_collective_cost_model("npu"))


if __name__ == "__main__":
    run_tests()
