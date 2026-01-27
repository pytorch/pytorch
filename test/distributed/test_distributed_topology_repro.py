"""
Tests for distributed topology extraction and repro generation.

This tests Goal 1: When running with tlparse and distributed, correctly set up
the right mesh that existed during the run, with FakeProcesses support.
"""

import io
import unittest

import torch
import torch.distributed as dist
from torch.testing._internal.common_utils import run_tests, TestCase


class TestDistributedTopologyRepro(TestCase):
    """Test distributed topology extraction and fx_graph_runnable generation."""

    def setUp(self):
        # Initialize fake distributed if not already initialized
        if not dist.is_initialized():
            from torch.testing._internal.distributed.fake_pg import FakeStore

            store = FakeStore()
            dist.init_process_group(backend="fake", rank=0, world_size=4, store=store)
            self._initialized_dist = True
        else:
            self._initialized_dist = False

    def tearDown(self):
        if self._initialized_dist and dist.is_initialized():
            dist.destroy_process_group()

    def test_extract_distributed_topology_with_collectives(self):
        """Test that distributed topology is correctly extracted from a graph with collectives."""
        from torch._subclasses import FakeTensorMode
        from torch.fx.experimental.proxy_tensor import make_fx
        from torch._dynamo.repro.after_aot import _extract_distributed_topology

        def f(x):
            return torch.ops._c10d_functional.all_reduce(x, "sum", "default")

        with FakeTensorMode():
            fake_x = torch.randn(4, 4)
            gm = make_fx(f)(fake_x)

        topology = _extract_distributed_topology(gm)

        self.assertTrue(topology["has_distributed_ops"])
        self.assertIn("default", topology["process_groups"])
        self.assertEqual(topology["process_groups"]["default"]["size"], 4)
        self.assertEqual(topology["process_groups"]["default"]["rank"], 0)

    def test_extract_distributed_topology_no_collectives(self):
        """Test that topology extraction works for graphs without collectives."""
        from torch._subclasses import FakeTensorMode
        from torch.fx.experimental.proxy_tensor import make_fx
        from torch._dynamo.repro.after_aot import _extract_distributed_topology

        def f(x):
            return x * 2

        with FakeTensorMode():
            fake_x = torch.randn(4, 4)
            gm = make_fx(f)(fake_x)

        topology = _extract_distributed_topology(gm)

        self.assertFalse(topology["has_distributed_ops"])
        self.assertEqual(topology["process_groups"], {})

    def test_generate_distributed_init_code(self):
        """Test that distributed init code is generated correctly."""
        from torch._dynamo.repro.after_aot import _generate_distributed_init_code

        topology = {
            "has_distributed_ops": True,
            "process_groups": {
                "default": {"size": 8, "rank": 2},
            },
            "device_meshes": [],
        }

        code = _generate_distributed_init_code(topology, use_fake_processes=True)

        # Check that the code contains expected elements
        self.assertIn("FakeStore", code)
        self.assertIn("dist.init_process_group", code)
        self.assertIn('backend="fake"', code)
        self.assertIn("REPRO_RANK", code)
        self.assertIn("REPRO_WORLD_SIZE", code)
        # Check that it uses the correct world size from topology
        self.assertIn('"8"', code)

    def test_generate_distributed_init_code_with_device_mesh(self):
        """Test that DeviceMesh setup code is generated when meshes are present."""
        from torch._dynamo.repro.after_aot import _generate_distributed_init_code

        topology = {
            "has_distributed_ops": True,
            "process_groups": {"default": {"size": 4, "rank": 0}},
            "device_meshes": [
                {
                    "device_type": "cuda",
                    "mesh": [[0, 1], [2, 3]],
                    "mesh_dim_names": ("dp", "tp"),
                }
            ],
        }

        code = _generate_distributed_init_code(topology, use_fake_processes=True)

        self.assertIn("DeviceMesh", code)
        self.assertIn('"cuda"', code)
        self.assertIn("[[0, 1], [2, 3]]", code)
        self.assertIn("('dp', 'tp')", code)

    def test_save_graph_repro_with_distributed(self):
        """Test that save_graph_repro generates proper repro with distributed ops."""
        from torch._subclasses import FakeTensorMode
        from torch.fx.experimental.proxy_tensor import make_fx
        from torch._dynamo.repro.after_aot import save_graph_repro

        def f(x):
            return torch.ops._c10d_functional.all_reduce(x, "sum", "default")

        with FakeTensorMode():
            fake_x = torch.randn(4, 4)
            gm = make_fx(f)(fake_x)

        fd = io.StringIO()
        save_graph_repro(fd, gm, [fake_x], "inductor")
        repro_code = fd.getvalue()

        # Check that the repro includes distributed setup
        self.assertIn("import torch.distributed as dist", repro_code)
        self.assertIn("FakeStore", repro_code)
        self.assertIn("dist.init_process_group", repro_code)
        self.assertIn("dist.destroy_process_group", repro_code)


if __name__ == "__main__":
    run_tests()
