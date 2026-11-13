# Owner(s): ["module: inductor"]

import contextlib
import sys
import unittest
import warnings
from types import SimpleNamespace
from unittest import mock

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


class TestRocmCommAnalysis(TestCase):
    def setUp(self):
        super().setUp()
        from torch._inductor.comm_analysis import get_gpu_type

        self.addCleanup(get_gpu_type.cache_clear)
        for patcher in (
            mock.patch.object(torch.version, "hip", "7.0"),
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(torch.cuda, "device_count", return_value=2),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    @contextlib.contextmanager
    def _on_arch(self, arch, major):
        from torch._inductor.comm_analysis import get_gpu_type

        props = SimpleNamespace(gcnArchName=arch, major=major)
        get_gpu_type.cache_clear()
        with mock.patch.object(torch.cuda, "get_device_properties", return_value=props):
            yield
        get_gpu_type.cache_clear()

    def test_cdna_keeps_gpu_model(self):
        from torch._inductor.comm_analysis import get_gpu_type, NVIDIA_GPU_TYPE

        for arch in ("gfx90a:sramecc+:xnack-", "gfx942:sramecc+:xnack-", "gfx950"):
            with self.subTest(arch=arch), self._on_arch(arch, 9):
                self.assertEqual(get_gpu_type(), NVIDIA_GPU_TYPE.HOPPER)

    def test_rdna_has_no_model(self):
        from torch._inductor.comm_analysis import (
            compute_min_saturation_bytes,
            detect_interconnect,
            get_gpu_type,
            get_intra_node_bw,
            InterconnectType,
            NCCL_COLL,
        )

        for arch, major in (("gfx1030", 10), ("gfx1100", 11), ("gfx1201", 12)):
            with self.subTest(arch=arch), self._on_arch(arch, major):
                self.assertEqual(
                    (get_gpu_type(), detect_interconnect(2), get_intra_node_bw()),
                    (None, InterconnectType.UNKNOWN, 0.0),
                )
                self.assertEqual(
                    compute_min_saturation_bytes(2, NCCL_COLL.ALL_GATHER), 0
                )

    def test_rdna_uses_configured_bandwidth(self):
        from torch._inductor.comm_analysis import (
            compute_min_saturation_bytes,
            estimate_nccl_collective_runtime_impl,
            NCCL_COLL,
        )

        def estimates():
            return (
                estimate_nccl_collective_runtime_impl(
                    64 * 1024 * 1024, 2, NCCL_COLL.ALL_GATHER
                ),
                compute_min_saturation_bytes(2, NCCL_COLL.ALL_GATHER),
            )

        with self._on_arch("gfx1100", 11):
            self.assertEqual(estimates(), (0, 0))
            with torch._inductor.config.patch(intra_node_bw=50):
                slow_time, low_sat = estimates()
            with torch._inductor.config.patch(intra_node_bw=100):
                fast_time, high_sat = estimates()
        self.assertGreater(slow_time, fast_time)
        self.assertGreater(fast_time, 0)
        self.assertGreater(high_sat, low_sat)
        self.assertGreater(low_sat, 0)


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
                get_gpu_type,
            )

            est_ms = estimate_nccl_collective_runtime_from_fx_node(
                node, use_nccl_estimator=True
            )
            if get_gpu_type() is None:
                self.assertEqual(est_ms, 0)
            else:
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


if __name__ == "__main__":
    run_tests()
