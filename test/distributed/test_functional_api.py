# Owner(s): ["oncall: distributed"]

import sys
import unittest
from functools import partial, wraps

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as ft_c
import torch.distributed.distributed_c10d as c10d
import torch.distributed.tensor as dt
from functorch import make_fx
from torch._inductor.utils import run_and_get_code
from torch.testing import FileCheck
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.inductor_utils import HAS_GPU


if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

from torch.testing._internal.common_distributed import (
    DistributedTestBase,
    MultiThreadedTestCase,
    requires_accelerator_dist_backend,
    TEST_SKIPS,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfHpu,
    TEST_CUDA,
    TEST_HPU,
    TEST_XPU,
    TestCase,
)


# NOTE: Instructions for adding new device types to this test file
#
# This test file contains two types of tests:
# 1. Tests that run on both CPUs and accelerators
# 2. Tests that run only on accelerators
#
# We use two variables to manage device types:
# - `devices`: A list containing device types for both CPU and accelerator tests
# - `DEVICE`: A string containing only the accelerator type for accelerator-only tests
#
# To add a new device type:
# 1. Add a new `elif` statement in the if-else ladder below
# 2. Check for the presence of your device (e.g., TEST_NEW_DEVICE)
# 3. Append your device type to the `devices` list
# 4. Assign your device type string to `DEVICE`
#
# Example:
# elif TEST_NEW_DEVICE:
#     devices.append("new_device")
#     DEVICE = "new_device"

DEVICE = "cuda"
devices = ["cpu"]
if TEST_HPU:
    devices.append("hpu")
    DEVICE = "hpu"
elif TEST_XPU:
    devices.append("xpu")
    DEVICE = "xpu"
elif TEST_CUDA:
    devices.append("cuda")


def new_subgroups(group_size: int, pg_tag=None):
    world_size = dist.get_world_size()
    subgroups = []
    cur_subgroup = None

    for subgroup_id in range(world_size // group_size):
        start_rank = subgroup_id * group_size
        end_rank = start_rank + group_size
        ranks_in_subgroup = list(range(start_rank, end_rank))
        subgroup = c10d._new_group_with_tag(
            ranks=ranks_in_subgroup,
            pg_tag=pg_tag,
        )
        subgroups.append(subgroup)

        rank = dist.get_rank()
        if rank in ranks_in_subgroup:
            cur_subgroup = subgroup

    return cur_subgroup, subgroups


@skipIfHpu
class TestExpand(MultiThreadedTestCase):
    @property
    def world_size(self):
        return 4

    def setUp(self):
        super().setUp()
        self._spawn_threads()

    def test_expand_1d_rank_list(self):
        tag, rankset, group_size = ft_c._expand_group([0, 1, 2, 3])
        self.assertEqual("", tag)
        self.assertEqual([0, 1, 2, 3], rankset)
        self.assertEqual(4, group_size)

        tag, rankset, group_size = ft_c._expand_group([0, 1, 2, 3], "bla")
        self.assertEqual("bla", tag)

    def test_expand_2d_rank_list(self):
        tag, rankset, group_size = ft_c._expand_group([[0, 1], [2, 3]])
        self.assertEqual("", tag)
        self.assertEqual([0, 1, 2, 3], rankset)
        self.assertEqual(2, group_size)

        tag, rankset, group_size = ft_c._expand_group([[0, 1], [2, 3]], "blu")
        self.assertEqual("blu", tag)

        with self.assertRaisesRegex(ValueError, "group sizes must be identical"):
            ft_c._expand_group([[0], [1, 2, 3]])

    def test_expand_process_group(self):
        tag, rankset, group_size = ft_c._expand_group(dist.group.WORLD)
        self.assertEqual(c10d._get_group_tag(dist.group.WORLD), tag)
        self.assertEqual([0, 1, 2, 3], rankset)
        self.assertEqual(4, group_size)

        tag, rankset, group_size = ft_c._expand_group(dist.group.WORLD, "bla")
        self.assertEqual("bla", tag)

        my_pg, _ = new_subgroups(group_size=2)
        tag, rankset, group_size = ft_c._expand_group(my_pg)
        self.assertEqual(c10d._get_group_tag(my_pg), tag)
        self.assertEqual(dist.get_process_group_ranks(my_pg), rankset)
        self.assertEqual(2, group_size)

        my_pg = None
        for i in range(dist.get_world_size()):
            group = c10d._new_group_with_tag([i], pg_tag="my_pg")
            if i == dist.get_rank():
                my_pg = group
        tag, rankset, group_size = ft_c._expand_group(my_pg)
        self.assertEqual("my_pg", tag)
        self.assertEqual([dist.get_rank()], rankset)
        self.assertEqual(1, group_size)

        tag, rankset, group_size = ft_c._expand_group(my_pg, "bla")
        self.assertEqual("bla", tag)

    def test_expand_device_mesh(self):
        mesh = dt.DeviceMesh("cpu", torch.arange(4))
        tag, rankset, group_size = ft_c._expand_group(mesh)
        self.assertEqual(c10d._get_group_tag(mesh.get_group(mesh_dim=0)), tag)
        self.assertEqual([0, 1, 2, 3], rankset)
        self.assertEqual(4, group_size)

        mesh = dt.DeviceMesh("cpu", torch.arange(4))
        tag, rankset, group_size = ft_c._expand_group(mesh)
        self.assertEqual(c10d._get_group_tag(mesh.get_group(mesh_dim=0)), tag)
        self.assertEqual([0, 1, 2, 3], rankset)
        self.assertEqual(4, group_size)

    def test_expand_device_mesh_tuple(self):
        mesh = dt.DeviceMesh("cpu", torch.arange(4).view(2, 2))
        with self.assertRaisesRegex(AssertionError, "Only 1D mesh"):
            tag, rankset, group_size = ft_c._expand_group(mesh)

        tag, rankset, group_size = ft_c._expand_group((mesh, 0))
        self.assertEqual(c10d._get_group_tag(mesh.get_group(mesh_dim=0)), tag)
        expected_rankset = [0, 2] if dist.get_rank() in [0, 2] else [1, 3]
        self.assertEqual(expected_rankset, rankset)
        self.assertEqual(2, group_size)

        tag, rankset, group_size = ft_c._expand_group((mesh, 1))
        expected_rankset = [0, 1] if dist.get_rank() in [0, 1] else [2, 3]
        self.assertEqual(c10d._get_group_tag(mesh.get_group(mesh_dim=1)), tag)
        self.assertEqual(expected_rankset, rankset)
        self.assertEqual(2, group_size)


@skipIfHpu
class TestPgTag(MultiThreadedTestCase):
    @property
    def world_size(self):
        return 4

    def setUp(self):
        super().setUp()
        self._spawn_threads()

    """
    The behavior we want is as follow:

    - rankset+tag will always result in the same PG.
    Do we enforce this by failing creation of new PGs or returning existing ones?
        Return existing one.

    - default tag gives existing behavior.
        This means we should create duplicates.
    - _expand_group on _default-tagged pg should always resolve to it
        This mean we can't depend on empty tag + rankset.
    """

    def test_pg_creation_with_tag(self):
        my_group, _ = new_subgroups(group_size=2, pg_tag="blu")
        my_group2, _ = new_subgroups(group_size=2, pg_tag="blu")
        self.assertEqual(my_group, my_group2)

        my_group3, _ = new_subgroups(group_size=2, pg_tag="blu2")
        self.assertNotEqual(my_group, my_group3)

        my_group4, _ = new_subgroups(group_size=2)
        self.assertNotEqual(my_group, my_group4)

        my_group5, _ = new_subgroups(group_size=2)
        self.assertNotEqual(my_group4, my_group5)

    def test_pg_lookup_roundtrip(self):
        pg_tag0, _ = new_subgroups(group_size=2, pg_tag="blu")
        pg_tag1, _ = new_subgroups(group_size=2, pg_tag="blu2")
        pg_notag0, _ = new_subgroups(group_size=2)
        pg_notag1, _ = new_subgroups(group_size=2)

        def roundtrip(pg):
            tag, rankset, _ = ft_c._expand_group(pg)
            return c10d._find_pg_by_ranks_and_tag(tag, rankset)

        self.assertEqual(pg_tag0, roundtrip(pg_tag0))
        self.assertEqual(pg_tag1, roundtrip(pg_tag1))
        self.assertEqual(pg_notag0, roundtrip(pg_notag0))
        self.assertEqual(pg_notag1, roundtrip(pg_notag1))

    def test_pg_lookup_with_tag(self):
        pg_tag0, _ = new_subgroups(group_size=2, pg_tag="blu")
        pg_tag1, _ = new_subgroups(group_size=2, pg_tag="bla")
        pg_notag0, _ = new_subgroups(group_size=2)

        def roundtrip(pg, pg_tag):
            tag, rankset, _ = ft_c._expand_group(pg, pg_tag)
            return c10d._find_pg_by_ranks_and_tag(tag, rankset)

        self.assertEqual(pg_tag0, roundtrip(pg_tag1, "blu"))
        self.assertEqual(pg_tag0, roundtrip(pg_notag0, "blu"))
        # Cannot erase the tag of a PG
        self.assertEqual(pg_tag0, roundtrip(pg_tag0, ""))

    def test_find_or_create_pg(self):
        pg = c10d._find_or_create_pg_by_ranks_and_tag("blu", [0, 1, 2, 3], 2)
        pg_tag0, _ = new_subgroups(group_size=2, pg_tag="blu")
        self.assertEqual(pg, pg_tag0)

    def test_find_root_pg(self):
        pg = c10d._find_pg_by_ranks_and_tag("", [0, 1, 2, 3])
        self.assertEqual(dist.group.WORLD, pg)


@instantiate_parametrized_tests
@skipIfHpu
class TestTraceableCollectives(MultiThreadedTestCase):
    @property
    def world_size(self):
        return 4

    def setUp(self):
        super().setUp()
        self._spawn_threads()

    @parametrize("device", devices)
    def test_broadcast(self, device):
        if device != "cpu":
            if torch.accelerator.device_count() < self.world_size:
                self.skipTest("Not enough accelerator devices")
            torch.accelerator.set_device_index(dist.get_rank())

        if dist.get_rank() == 0:
            tensor = torch.ones([4], device=device)
        else:
            tensor = torch.zeros([4], device=device)

        mesh = dt.DeviceMesh(device, torch.arange(4))
        res = ft_c.broadcast(tensor, 0, mesh)
        self.assertEqual(res, torch.ones([4], device=device))

    @parametrize("device", devices)
    def test_all_reduce_eager(self, device):
        if device != "cpu":
            if torch.accelerator.device_count() < self.world_size:
                self.skipTest("Not enough accelerator devices")
            torch.accelerator.set_device_index(dist.get_rank())

        tensor = torch.ones([4], device=device)
        mesh = dt.DeviceMesh(device, torch.arange(4))

        res = ft_c.all_reduce(tensor, "sum", mesh)
        self.assertEqual(res, torch.tensor([4, 4, 4, 4], dtype=torch.float))

        mesh = dt.DeviceMesh(device, torch.arange(4).view(2, 2))
        res2 = ft_c.all_reduce(tensor, "sum", (mesh, 1))
        self.assertEqual(res2, torch.tensor([2, 2, 2, 2], dtype=torch.float))

    @parametrize("device", devices)
    def test_all_reduce_coalesced_eager(self, device):
        if device != "cpu":
            if torch.accelerator.device_count() < self.world_size:
                self.skipTest("Not enough accelerator devices")
            torch.accelerator.set_device_index(dist.get_rank())

        t0 = torch.ones([4], device=device)
        t1 = torch.ones([6], device=device) + 2
        mesh = dt.DeviceMesh(device, torch.arange(4))

        res = ft_c.all_reduce_coalesced([t0, t1], "sum", mesh)
        self.assertEqual(res[0], t0 * 4)
        self.assertEqual(res[1], t1 * 4)

    @parametrize("device", devices)
    def test_all_gather_tensor(self, device):
        if device != "cpu":
            if torch.accelerator.device_count() < self.world_size:
                self.skipTest("Not enough accelerator devices")
            torch.accelerator.set_device_index(dist.get_rank())

        # testing 1d/2d mesh
        mesh_1d = dt.DeviceMesh(device, torch.arange(self.world_size))
        mesh_2d = dt.DeviceMesh(device, torch.arange(self.world_size).view(2, 2))
        for mesh in [mesh_1d, mesh_2d]:
            dims_to_gather = [0, 1, 2]
            for dim in dims_to_gather:
                output_size = [3, 3, 3]
                output_size[dim] *= mesh.size(0)
                # each rank have its own tensor, all_gather gives a bigger tensor
                local_tensor = torch.ones([3, 3, 3], device=device)
                gathered_tensor = ft_c.all_gather_tensor(
                    local_tensor, gather_dim=dim, group=(mesh, 0)
                )
                self.assertEqual(gathered_tensor, torch.ones(output_size))

    @parametrize("device", devices)
    def test_all_gather_into_tensor_coalesced(self, device):
        if device != "cpu":
            if torch.accelerator.device_count() < self.world_size:
                self.skipTest("Not enough accelerator devices")
            torch.accelerator.set_device_index(dist.get_rank())

        tensors = [torch.ones([4], device=device), torch.ones([4], device=device) + 1]
        mesh = dt.DeviceMesh(device, torch.arange(4))

        res = ft_c.all_gather_into_tensor_coalesced(tensors, mesh)
        self.assertEqual(2, len(res))
        self.assertEqual(torch.ones([4 * dist.get_world_size()], device=device), res[0])
        self.assertEqual(
            torch.ones([4 * dist.get_world_size()], device=device) + 1, res[1]
        )

    @parametrize("device", devices)
    def test_reduce_scatter_tensor(self, device):
        if device != "cpu":
            if torch.accelerator.device_count() < self.world_size:
                self.skipTest("Not enough accelerator devices")
            torch.accelerator.set_device_index(dist.get_rank())

        # testing 1d/2d mesh
        mesh_1d = dt.DeviceMesh(device, torch.arange(self.world_size))
        mesh_2d = dt.DeviceMesh(device, torch.arange(self.world_size).view(2, 2))
        for mesh in [mesh_1d, mesh_2d]:
            dims_to_scatter = [0, 1]
            for dim in dims_to_scatter:
                group_size = mesh.size(0)
                input_size = [3, 3]
                output_size = [3, 3]
                output_size[dim] *= group_size
                input_tensor = torch.ones(output_size, device=device)
                res_num = 1 * group_size
                rs_tensor = ft_c.reduce_scatter_tensor(
                    input_tensor, "sum", scatter_dim=dim, group=(mesh, 0)
                )
                self.assertEqual(rs_tensor, torch.ones(input_size) * res_num)

    @parametrize("device", devices)
    def test_reduce_scatter_into_tensor_coalesced(self, device):
        if device != "cpu":
            if torch.accelerator.device_count() < self.world_size:
                self.skipTest("Not enough accelerator devices")
            torch.accelerator.set_device_index(dist.get_rank())
        tensors = [
            torch.ones([4], dtype=torch.int64, device=device),
            torch.ones([4], dtype=torch.int64, device=device) + 1,
        ]
        mesh = dt.DeviceMesh(device, torch.arange(4))

        res = ft_c.reduce_scatter_tensor_coalesced(tensors, "sum", [0, 0], mesh)
        self.assertEqual(2, len(res))
        self.assertEqual(torch.tensor([4], device=device), res[0])
        self.assertEqual(torch.tensor([8], device=device), res[1])


class TestMetaCollectives(TestCase):
    def test_all_reduce(self):
        x = torch.rand((2, 3, 4), device="meta")
        out = ft_c.all_reduce(x, "sum", "0")
        self.assertEqual(x.size(), out.size())


@skipIfHpu
class TestGradCollectives(MultiThreadedTestCase):
    @property
    def world_size(self):
        return 2

    def setUp(self):
        super().setUp()
        self._spawn_threads()

    def test_all_reduce(self):
        x = torch.rand([4], requires_grad=True)
        y = torch.rand([4], requires_grad=True)
        out = ft_c.all_reduce(x, "sum", dist.group.WORLD)
        (out + y).sum().backward()
        self.assertIsNotNone(x.grad)


class TestMakeFx(TestCase):
    def setUp(self):
        # make_fx is not thread-safe due to patching nd mutating global states
        # so create a fake_pg.
        self.rank = 0
        self.world_size = 2
        dist.init_process_group(
            backend="fake",
            world_size=self.world_size,
            rank=self.rank,
        )

    def tearDown(self):
        super().tearDown()

        self.assertFalse(torch.fx._symbolic_trace.is_fx_tracing())

    def test_all_reduce_tracing(self):
        def allred(input):
            return ft_c.all_reduce(input, "sum", group=dist.group.WORLD) + 1

        graph = make_fx(allred)(torch.rand(4))
        FileCheck().check("all_reduce").check("wait_tensor").run(str(graph.graph))

        mesh = dt.DeviceMesh("cpu", torch.arange(self.world_size))

        def allred_mesh(input):
            return ft_c.all_reduce(input, "sum", mesh) + 1

        mesh_graph = make_fx(allred_mesh)(torch.rand(4))
        FileCheck().check_not("get_attr").check("wait_tensor").run(
            str(mesh_graph.graph)
        )

        def allred_mesh_dim(input):
            return ft_c.all_reduce(input, "sum", (mesh, 0)) + 1

        mesh_dim_graph = make_fx(allred_mesh_dim)(torch.rand(4))
        FileCheck().check_not("get_attr").check("wait_tensor").run(
            str(mesh_dim_graph.graph)
        )


BACKEND = dist.Backend.NCCL if torch.cuda.is_available() else dist.Backend.GLOO

# Adding support for HCCL backend
# To add a different backend
# add an elif to the same chain with a conditional checking for the device type (along the lines of TEST_HPU or TEST_CUDA)
# And then set the BACKEND variable appropriately.
if TEST_HPU:
    BACKEND = dist.Backend.HCCL
elif TEST_XPU:
    BACKEND = dist.Backend.XCCL


# allows you to check for multiple accelerator irrespective of device type
# to add new device types to this check simply follow the same format
# and append an elif with the conditional and appropriate device count function for your new device
def exit_if_lt_x_accelerators(x):
    if torch.accelerator.is_available():
        if torch.accelerator.device_count() < x:
            sys.exit(TEST_SKIPS[f"multi-gpu-{x}"].exit_code)


def with_comms(func=None):
    if func is None:
        return partial(with_comms)

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if (
            BACKEND == dist.Backend.NCCL or BACKEND == dist.Backend.XCCL
        ) and torch.accelerator.device_count() < self.world_size:
            sys.exit(TEST_SKIPS[f"multi-gpu-{self.world_size}"].exit_code)

        kwargs["device"] = DEVICE
        self.pg = self.create_pg(device=DEVICE)
        try:
            return func(self, *args, **kwargs)
        finally:
            torch.distributed.destroy_process_group()

    return wrapper


class TestCollectivesWithDistributedBackend(DistributedTestBase):
    @with_comms()
    def test_all_gather_into_tensor_coalesced(self, device):
        exit_if_lt_x_accelerators(self.world_size)
        tensors = [
            torch.ones([4], device=device),
            torch.ones([4], device=device) + 1,
        ]
        mesh = dt.DeviceMesh(device, torch.arange(self.world_size))

        res = ft_c.all_gather_into_tensor_coalesced(tensors, mesh)
        self.assertEqual(2, len(res))
        self.assertEqual(torch.ones([4 * dist.get_world_size()]), res[0])
        self.assertEqual(torch.ones([4 * dist.get_world_size()]) + 1, res[1])

    @with_comms()
    def test_all_to_all_single(self, device):
        mesh = dt.DeviceMesh(device, torch.arange(self.world_size))
        rank = dist.get_rank()

        row = self.world_size * (rank + 1) * (self.world_size + 1) / 2
        x = torch.ones(int(row), 5, device=device) * (rank + 1)
        split_sizes = [(i + 1) * (rank + 1) for i in range(self.world_size)]
        y = ft_c.all_to_all_single(
            x, output_split_sizes=split_sizes, input_split_sizes=split_sizes, group=mesh
        )
        expected = []
        for idx, tensor in enumerate(torch.split(x, split_sizes)):
            expected.append(torch.full_like(tensor, (idx + 1)))
        expected = torch.cat(expected)
        self.assertEqual(y, expected)

    @with_comms()
    def test_all_to_all_single_1d_input(self, device):
        mesh = dt.DeviceMesh(device, torch.arange(self.world_size))
        rank = dist.get_rank()

        row = self.world_size * (rank + 1) * (self.world_size + 1) / 2
        x = torch.ones(int(row), device=device) * (rank + 1)
        split_sizes = [(i + 1) * (rank + 1) for i in range(self.world_size)]
        y = ft_c.all_to_all_single(
            x, output_split_sizes=split_sizes, input_split_sizes=split_sizes, group=mesh
        )
        expected = []
        for idx, tensor in enumerate(torch.split(x, split_sizes)):
            expected.append(torch.full_like(tensor, (idx + 1)))
        expected = torch.cat(expected)
        self.assertEqual(y, expected)

    @with_comms()
    def test_all_to_all_single_split_sizes_none(self, device):
        mesh = dt.DeviceMesh(device, torch.arange(self.world_size))
        rank = dist.get_rank()

        x = torch.ones(self.world_size, self.world_size, device=device) * (rank + 1)
        y = ft_c.all_to_all_single(
            x, output_split_sizes=None, input_split_sizes=None, group=mesh
        )
        expected = []
        for idx, tensor in enumerate(torch.chunk(x, self.world_size)):
            expected.append(torch.full_like(tensor, (idx + 1)))
        expected = torch.cat(expected)
        self.assertEqual(y, expected)

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @requires_accelerator_dist_backend(["nccl", "xccl"])
    @with_comms()
    def test_tracing(self, device):
        def allreduce(t, pg):
            return ft_c.all_reduce(t, "sum", pg)

        compiled_allreduce = torch.compile(allreduce, fullgraph=True)
        compiled_allreduce(torch.randn(8, device=device), self.pg)

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    def test_tracing_with_fakepg(self, device=DEVICE):
        exit_if_lt_x_accelerators(self.world_size)

        def allreduce(t, pg):
            return ft_c.all_reduce(t, "sum", pg)

        compiled_allreduce = torch.compile(allreduce, fullgraph=True)  # noqa: F841
        dist.init_process_group(
            backend="fake",
            rank=0,
            world_size=8,
        )
        allreduce(torch.randn(8, device=device), pg=dist.group.WORLD)
        dist.destroy_process_group()

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @requires_accelerator_dist_backend(["nccl", "xccl"])
    @with_comms()
    def test_tracing_with_dce_code(self, device):
        if self.world_size > 2:
            return

        def func(batch, group, rank):
            ret = ft_c.permute_tensor(batch, [1, 0], group)
            if hasattr(ret, "wait"):
                ret = ret.wait()
            if rank == 0:
                return ret
            else:
                return batch * 5

        compiled_func = torch.compile(func)
        compiled_func(torch.ones((100,), device=device), self.process_group, self.rank)
        dist.barrier()


class TestDistributedBackendCollectivesWithWorldSize4(
    TestCollectivesWithDistributedBackend
):
    @property
    def world_size(self):
        return 4

    @with_comms()
    def test_permute_tensor_with_sub_group(self, device):
        exit_if_lt_x_accelerators(self.world_size)
        mesh_dim_names = ["dp", "tp"]

        mesh_2d = dt.init_device_mesh(
            device, (2, self.world_size // 2), mesh_dim_names=mesh_dim_names
        )

        for mesh_name in mesh_dim_names:
            mesh = mesh_2d[mesh_name]
            rank = mesh.get_local_rank()

            # rank0: [0., 1.], rank1: [2., 3.]
            send_tensor = torch.arange(2, dtype=torch.float32, device=device) + 2 * rank
            recvd_tensor = ft_c.permute_tensor(send_tensor, [1, 0], group=mesh)

            # rank0: [2., 3.], rank1: [0., 1.]
            expected = torch.arange(2, dtype=torch.float32, device=device) + 2 * (
                (rank - 1 + 2) % 2
            )
            self.assertEqual(
                recvd_tensor,
                expected,
                msg=f"Expected {expected} on {self.rank=} (local_rank={rank}), "
                f"but received {recvd_tensor} instead.",
            )


@instantiate_parametrized_tests
@skipIfHpu
class TestFunctionalAutograd(MultiThreadedTestCase):
    def setUp(self):
        super().setUp()
        self._spawn_threads()

    @property
    def world_size(self):
        return 2

    @parametrize("compile", [True, False])
    def test_all_to_all_single(self, compile: bool = True) -> None:
        group = dist.group.WORLD.group_name

        t = torch.ones((self.world_size, 2), requires_grad=True)

        def my_func(t: torch.Tensor, world_size: int) -> torch.Tensor:
            sizes = [1] * world_size
            t = t * 2
            assert t.requires_grad
            out = ft_c.all_to_all_single_autograd(t, sizes, sizes, group)
            out = out + 0
            return out

        if compile:
            compiled = torch.compile(my_func, fullgraph=True, backend="aot_eager")
        else:
            compiled = my_func

        out = compiled(t, self.world_size)
        self.assertEqual(out.shape, t.shape)
        self.assertEqual(out, torch.full_like(t, 2.0))
        self.assertIsNotNone(out.grad_fn)
        self.assertTrue(out.requires_grad)
        loss = out.sum()
        loss.backward()
        self.assertEqual(t.grad, torch.full_like(t, 2.0))

    def test_all_to_all_single_inductor(self) -> None:
        group = dist.group.WORLD.group_name

        t = torch.rand((self.world_size, 2), requires_grad=True)

        def my_func(t: torch.Tensor, world_size: int) -> torch.Tensor:
            sizes = [1] * world_size
            t = t * 10
            assert t.requires_grad
            out = ft_c.all_to_all_single_autograd(t, sizes, sizes, group)
            out = out + 2
            return out.sum()

        compiled = torch.compile(my_func, fullgraph=True)

        def run_with_backward():
            out = compiled(t, self.world_size)
            out.backward()

        _, codes = run_and_get_code(run_with_backward)
        for code in codes:
            assert_keywords = ["assert_size_stride", "assert_alignment"]
            filtered_lines = [
                line
                for line in code.splitlines()
                if not any(assert_key in line for assert_key in assert_keywords)
            ]
            code = "\n".join(filtered_lines)
            FileCheck().check_count(
                "_c10d_functional.all_to_all_single.default", 1, exactly=True
            ).check_count("_c10d_functional.wait_tensor.default", 1, exactly=True).run(
                code
            )

        self.assertIsNotNone(t.grad)

    @parametrize("compile", [True, False])
    def test_all_gather_tensor(self, compile: bool) -> None:
        group = dist.group.WORLD.group_name

        def my_func(t: torch.Tensor, dim: int) -> torch.Tensor:
            assert t.requires_grad
            out = ft_c.all_gather_tensor_autograd(
                t * 1.0,
                gather_dim=dim,
                group=group,
            )
            out = out * 1.0
            return out

        if compile:
            compiled = torch.compile(my_func, fullgraph=True, backend="aot_eager")
        else:
            compiled = my_func

        dims_to_gather = [0, 1, 2]
        for dim in dims_to_gather:
            output_size = [3, 3, 3]
            output_size[dim] *= self.world_size
            # each rank have its own tensor, all_gather gives a bigger tensor
            local_tensor = torch.ones([3, 3, 3], requires_grad=True)
            gathered_tensor = compiled(local_tensor, dim)
            self.assertEqual(gathered_tensor, torch.ones(output_size))

            gathered_tensor.sum().backward()
            self.assertEqual(
                local_tensor.grad,
                torch.full((3, 3, 3), fill_value=float(self.world_size)),
            )

    @parametrize("compile", [True, False])
    def test_reduce_scatter_tensor(self, compile: bool) -> None:
        group = dist.group.WORLD.group_name

        def my_func(t: torch.Tensor, dim: int) -> torch.Tensor:
            assert t.requires_grad
            rs_tensor = (
                ft_c.reduce_scatter_tensor_autograd(
                    input_tensor * 1.0, "sum", scatter_dim=dim, group=group
                )
                * 1.0
            )
            return rs_tensor

        if compile:
            compiled = torch.compile(my_func, fullgraph=True, backend="aot_eager")
        else:
            compiled = my_func

        dims_to_scatter = [0, 1]
        for dim in dims_to_scatter:
            group_size = self.world_size
            input_size = [3, 3]
            output_size = [3, 3]
            output_size[dim] *= group_size
            input_tensor = torch.ones(output_size, requires_grad=True)
            rs_tensor = compiled(input_tensor, dim)
            res_num = 1 * group_size
            self.assertEqual(rs_tensor, torch.ones(input_size) * res_num)
            rs_tensor.sum().backward()
            self.assertEqual(input_tensor.grad, torch.full(output_size, fill_value=1.0))


class TestFunctionalAutogradWithDistributedBackend(DistributedTestBase):
    @with_comms()
    def test_all_to_all_single(self, device) -> None:
        group = self.pg
        t = torch.ones((self.world_size, 2), requires_grad=True, device=device)

        sizes = [1] * self.world_size
        assert t.requires_grad
        out = ft_c.all_to_all_single_autograd(t * 2, sizes, sizes, group) + 0

        self.assertEqual(out.shape, t.shape)
        self.assertEqual(out, torch.full_like(t, 2.0))
        self.assertIsNotNone(out.grad_fn)
        self.assertTrue(out.requires_grad)
        loss = out.sum()
        loss.backward()
        self.assertEqual(t.grad, torch.full_like(t, 2.0))


# Update the supported devices in DEVICE
instantiate_device_type_tests(
    TestCollectivesWithDistributedBackend, globals(), only_for=DEVICE, allow_xpu=True
)
instantiate_device_type_tests(
    TestDistributedBackendCollectivesWithWorldSize4,
    globals(),
    only_for=DEVICE,
    allow_xpu=True,
)
instantiate_device_type_tests(
    TestFunctionalAutogradWithDistributedBackend,
    globals(),
    only_for=DEVICE,
    allow_xpu=True,
)

if __name__ == "__main__":
    run_tests()
