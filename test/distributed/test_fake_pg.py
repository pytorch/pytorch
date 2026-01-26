# Owner(s): ["oncall: distributed"]

import sys
import unittest

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
import torch.nn as nn
from torch._C._distributed_c10d import FakeProcessGroup
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DeviceMesh, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    RowwiseParallel,
)
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing import FileCheck
from torch.testing._internal.common_distributed import HAS_ACCELERATOR
from torch.testing._internal.common_fsdp import get_devtype
from torch.testing._internal.common_utils import run_tests, skipIfHpu, TestCase
from torch.testing._internal.distributed._tensor.common_dtensor import MLPModule
from torch.testing._internal.distributed.fake_pg import FakeStore
from torch.utils._python_dispatch import TorchDispatchMode


if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

device_type = get_devtype().type


class TestFakePG(TestCase):
    def tearDown(self):
        super().tearDown()
        try:
            dist.destroy_process_group()
        except AssertionError:
            pass

    def test_all_reduce(self):
        dist.init_process_group(backend="fake", rank=1, world_size=2)

        output = torch.ones(3, 3) * dist.get_rank()
        dist.all_reduce(output)
        self.assertEqual(tuple(output.shape), (3, 3))

    def test_allgather(self):
        dist.init_process_group(backend="fake", rank=1, world_size=2)

        input_tensor = torch.ones(3, 3) * dist.get_rank()
        output_tensors = [torch.empty_like(input_tensor) for _ in range(2)]
        dist.all_gather(output_tensors, input_tensor)
        for out_tensor in output_tensors:
            self.assertEqual(tuple(out_tensor.shape), (3, 3))

    def test_reduce_scatter(self):
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=1, world_size=2, store=store)

        to_reduce_scatter = [torch.ones(3, 3) * rank for rank in range(2)]
        output_tensor = torch.empty(3, 3)

        dist.reduce_scatter(output_tensor, to_reduce_scatter)
        self.assertEqual(tuple(output_tensor.shape), (3, 3))

    @unittest.skipIf(not HAS_ACCELERATOR, "No accelerator")
    def test_construct_fsdp(self):
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)
        FSDP(nn.Linear(2, 3, device=device_type))

    @skipIfHpu
    @unittest.skipIf(not HAS_ACCELERATOR, "No accelerator")
    def test_fsdp_fake_e2e(self):
        store = dist.HashStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)
        my_module = nn.Sequential(
            nn.Linear(2, 3, device=device_type),
            nn.ReLU(),
            nn.Linear(3, 2, device=device_type),
        )
        sharded_module = FSDP(my_module, use_orig_params=True)
        optim = torch.optim.Adam(sharded_module.parameters(), lr=0.0001)
        input = torch.randn(2, 2)
        x = sharded_module(input)
        loss = x.sum()
        loss.backward()
        optim.step()

    @skipIfHpu
    @unittest.skipIf(not HAS_ACCELERATOR, "No accelerator")
    def test_fake_pg_tracing(self):
        store = dist.HashStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)

        default_pg = dist.distributed_c10d._get_default_group()

        def allgather_fn(tensor):
            return funcol.all_gather_tensor(tensor, 0, default_pg)

        gm = make_fx(allgather_fn)(torch.randn(2, 2, device=device_type))
        FileCheck().check("all_gather").check("wait_tensor").run(str(gm.graph))

    def test_broadcast(self):
        dist.init_process_group(backend="fake", rank=0, world_size=2)

        # src == rank
        output = torch.ones(3, 3)
        dist.broadcast(output, src=0)
        self.assertEqual(tuple(output.shape), (3, 3))

        # src != rank
        output = torch.ones(3, 3)
        dist.broadcast(output, src=1)
        self.assertEqual(tuple(output.shape), (3, 3))

    def test_scatter(self):
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)

        # src == rank
        output = torch.ones(3, 3)
        to_scatter = [torch.ones(3, 3) * rank for rank in range(2)]
        dist.scatter(output, to_scatter)
        self.assertEqual(tuple(output.shape), (3, 3))

        # src != rank
        output = torch.ones(3, 3)
        dist.scatter(output, None, src=1)
        self.assertEqual(tuple(output.shape), (3, 3))

    def test_alltoall(self):
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)

        output_list = [torch.ones(3, 3) for _ in range(2)]
        input_list = [torch.ones(3, 3) for _ in range(2)]
        dist.all_to_all(output_list, input_list)
        self.assertEqual(len(output_list), 2)
        for output in output_list:
            self.assertEqual(tuple(output.shape), (3, 3))

    def test_alltoall_base(self):
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)

        out_tensor = torch.ones(3, 3)
        in_tensor = torch.ones(3, 3)
        output_split = [1, 1]
        input_split = [1, 1]
        dist.all_to_all_single(out_tensor, in_tensor, output_split, input_split)
        self.assertEqual(tuple(out_tensor.shape), (3, 3))

    def test_send(self):
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)

        tensor = torch.ones(3, 3)
        dist.send(tensor, 1)
        self.assertEqual(tuple(tensor.shape), (3, 3))

    def test_recv(self):
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)

        output = torch.ones(3, 3)
        dist.recv(output, 1)
        self.assertEqual(tuple(output.shape), (3, 3))

    @skipIfHpu
    @unittest.skipIf(not HAS_ACCELERATOR, "No accelerator")
    def test_fsdp_tp_fake_e2e(self):
        world_size = 4
        tp_size = 2

        store = dist.HashStore()
        dist.init_process_group(
            backend="fake", rank=0, world_size=world_size, store=store
        )

        device_mesh = DeviceMesh(
            device_type, torch.arange(0, world_size).view(-1, tp_size)
        )
        device_mesh = init_device_mesh(
            device_type, (world_size // tp_size, tp_size), mesh_dim_names=["dp", "tp"]
        )

        sequence_parallelize_plan = {
            "net1": ColwiseParallel(input_layouts=Shard(0)),
            "net2": RowwiseParallel(output_layouts=Shard(0)),
        }
        pairwise_parallelize_plan = {
            "net1": ColwiseParallel(),
            "net2": RowwiseParallel(),
        }
        for parallel_plan in [sequence_parallelize_plan, pairwise_parallelize_plan]:
            my_module = parallelize_module(
                MLPModule(device=device_type),
                device_mesh["tp"],
                parallel_plan,
            )

            sharded_module = FSDP(
                my_module, use_orig_params=True, device_mesh=device_mesh["dp"]
            )
            optim = torch.optim.Adam(sharded_module.parameters(), lr=0.0001)

            for i in range(10):
                dp_rank = dist.get_rank()
                torch.manual_seed(i + dp_rank)
                input = torch.randn(20, 10, device=f"{device_type}:{dp_rank}")
                x = sharded_module(input)
                loss = x.sum()
                loss.backward()
                optim.step()

    def test_error_on_collective(self):
        from torch.testing._internal.distributed.fake_pg import FakeStore

        # Test with error_on_collective=False (default behavior)
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)

        # These should work normally
        tensor = torch.ones(3, 3)
        dist.all_reduce(tensor)
        self.assertEqual(tuple(tensor.shape), (3, 3))

        dist.destroy_process_group()

        # Test with error_on_collective=True
        from torch._C._distributed_c10d import FakeProcessGroup

        options = FakeProcessGroup.Options()
        options.error_on_collective = True

        store = FakeStore()
        dist.init_process_group(
            backend="fake", rank=0, world_size=2, store=store, pg_options=options
        )

        # These should now raise errors
        tensor = torch.ones(3, 3)
        with self.assertRaisesRegex(
            RuntimeError, "FakeProcessGroup collective operation error"
        ):
            dist.all_reduce(tensor)

        with self.assertRaisesRegex(
            RuntimeError, "FakeProcessGroup collective operation error"
        ):
            output_tensors = [torch.empty_like(tensor) for _ in range(2)]
            dist.all_gather(output_tensors, tensor)

        with self.assertRaisesRegex(
            RuntimeError, "FakeProcessGroup collective operation error"
        ):
            dist.broadcast(tensor, src=0)

        with self.assertRaisesRegex(
            RuntimeError, "FakeProcessGroup collective operation error"
        ):
            dist.barrier()

    def test_fake_process_group_direct_usage_error(self):
        class SimpleTensorMode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if kwargs is None:
                    kwargs = {}
                return func(*args, **kwargs)

        with self.assertRaisesRegex(TypeError, r"No constructor defined"):
            fake_pg = FakeProcessGroup(rank=0, world_size=3)

            with SimpleTensorMode():
                tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
                dist.all_reduce(tensor, group=fake_pg)

    def test_fake_process_group_proper_usage_dispatch(self):
        class SimpleTensorMode(TorchDispatchMode):
            def __init__(self):
                self.ops = []

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                self.ops.append(str(func))
                if kwargs is None:
                    kwargs = {}
                return func(*args, **kwargs)

        fake_store = FakeStore()
        dist.init_process_group("fake", store=fake_store, rank=0, world_size=3)

        with SimpleTensorMode() as mode:
            tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            dist.all_reduce(tensor)

        op_names = [str(op) for op in mode.ops]
        self.assertIn("aten.lift_fresh.default", op_names)
        self.assertIn("c10d.allreduce_.default", op_names)


class TestCrossBackendProcessGroupNaming(TestCase):
    """
    Tests for cross-backend process group naming consistency.

    These tests verify that DeviceMesh creates canonical process group names that
    are consistent between fake and real backends. This is critical for the
    cross-backend precompilation workflow where:
    1. Precompile with fake backend (single process, fake PGs)
    2. Load and run with real NCCL backend (multi-process, real PGs)

    The canonical naming ensures that process group references in compiled
    artifacts can be resolved correctly regardless of which backend was used
    during compilation.
    """

    def setUp(self):
        super().setUp()
        # Ensure no process group is initialized from previous tests
        if dist.is_initialized():
            dist.destroy_process_group()
        # Clear the C++ GroupRegistry completely (including aliases from previous tests)
        torch._C._distributed_c10d._unregister_all_process_groups()
        # Clear device mesh resources
        from torch.distributed.device_mesh import _mesh_resources

        _mesh_resources.mesh_stack.clear()

    def tearDown(self):
        super().tearDown()
        try:
            dist.destroy_process_group()
        except AssertionError:
            pass
        # Clear the C++ GroupRegistry completely
        torch._C._distributed_c10d._unregister_all_process_groups()
        # Clear device mesh resources
        from torch.distributed.device_mesh import _mesh_resources

        _mesh_resources.mesh_stack.clear()

    def test_canonical_naming_with_fake_backend(self):
        """
        Test that DeviceMesh creates canonical names with fake backend.

        This verifies that process groups created with fake backend get canonical
        names registered as aliases, which is required for cross-backend precompile.
        """
        world_size = 4
        store = FakeStore()
        dist.init_process_group(
            backend="fake", rank=0, world_size=world_size, store=store
        )

        # Create a 2D mesh (simulating FSDP x TP)
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=["fsdp", "tp"])

        # Verify that the mesh stores canonical names that are resolvable
        # The canonical name format is: mesh_mesh_{dim_name}_{hash(first_subgroup_ranks)}
        self.assertEqual(len(mesh._dim_group_names), 2)

        # All canonical names stored in mesh._dim_group_names should be resolvable
        for dim_idx, dim_name in enumerate(["fsdp", "tp"]):
            canonical_name = mesh._dim_group_names[dim_idx]
            # Verify the canonical name follows the expected pattern
            self.assertTrue(
                canonical_name.startswith(f"mesh_mesh_{dim_name}_"),
                f"Canonical name {canonical_name} doesn't start with mesh_mesh_{dim_name}_",
            )

            # Verify the canonical name is resolvable
            # This is critical for cross-backend precompile: the canonical name
            # is baked into compiled artifacts and must be resolvable at load time.
            try:
                pg = dist.distributed_c10d._resolve_process_group(canonical_name)
                self.assertIsNotNone(pg)
            except LookupError:
                self.fail(
                    f"Failed to resolve canonical name: {canonical_name}. "
                    "This would break cross-backend precompilation."
                )

    def test_canonical_names_are_hash_based(self):
        """
        Test that canonical names use hash-based naming for consistency.

        The canonical name should be deterministic based on the mesh dimensions
        and rank groups, ensuring that fake backend and real backend produce
        compatible names.
        """
        world_size = 4
        store = FakeStore()
        dist.init_process_group(
            backend="fake", rank=0, world_size=world_size, store=store
        )

        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=["fsdp", "tp"])

        # Canonical names should contain a hash suffix (numeric)
        for canonical_name in mesh._dim_group_names:
            # Name format: mesh_mesh_{dim_name}_{hash}
            parts = canonical_name.split("_")
            self.assertGreaterEqual(len(parts), 4)
            hash_part = parts[-1]
            # The hash should be numeric
            self.assertTrue(
                hash_part.isdigit(),
                f"Hash part '{hash_part}' in canonical name '{canonical_name}' "
                "should be numeric",
            )

    @skipIfHpu
    @unittest.skipIf(not HAS_ACCELERATOR, "No accelerator")
    def test_fsdp2_with_fake_backend_creates_canonical_names(self):
        """
        Test that FSDP2 with fake backend creates resolvable canonical PG names.

        This is an end-to-end test that verifies the FSDP2 + DeviceMesh workflow
        creates process groups with canonical names that would be resolvable
        when loading precompiled artifacts.

        Note: We use a 2D mesh (fsdp, tp) because a 1D mesh that spans the entire
        world reuses the default process group without canonical naming (an
        optimization that avoids creating unnecessary groups).
        """
        from torch.distributed._composable.fsdp import fully_shard

        world_size = 4
        store = dist.HashStore()
        dist.init_process_group(
            backend="fake", rank=0, world_size=world_size, store=store
        )

        # Create 2D mesh for FSDP + TP (2x2)
        # This ensures we go through the canonical naming path
        mesh = init_device_mesh(
            device_type, (2, 2), mesh_dim_names=["fsdp", "tp"]
        )

        # Create and shard a simple model using the fsdp submesh
        model = nn.Sequential(
            nn.Linear(10, 10, device=device_type),
            nn.ReLU(),
            nn.Linear(10, 10, device=device_type),
        )
        fully_shard(model, mesh=mesh["fsdp"])

        # Verify canonical names for both dimensions are stored and resolvable
        self.assertEqual(len(mesh._dim_group_names), 2)

        for dim_idx, dim_name in enumerate(["fsdp", "tp"]):
            canonical_name = mesh._dim_group_names[dim_idx]
            self.assertTrue(
                canonical_name.startswith(f"mesh_mesh_{dim_name}_"),
                f"Canonical name {canonical_name} doesn't match expected pattern "
                f"mesh_mesh_{dim_name}_*",
            )

            try:
                pg = dist.distributed_c10d._resolve_process_group(canonical_name)
                self.assertIsNotNone(pg)
            except LookupError:
                self.fail(
                    f"Failed to resolve canonical name: {canonical_name}. "
                    "FSDP2 precompiled artifacts would fail to load with real backend."
                )

        # Run a forward pass to ensure everything works
        input_tensor = torch.randn(2, 10, device=device_type)
        output = model(input_tensor)
        self.assertEqual(output.shape, (2, 10))


if __name__ == "__main__":
    run_tests()
