# Owner(s): ["oncall: distributed"]

import datetime
import os
import unittest
from unittest import mock

import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
from torch.distributed.distributed_c10d import _TORCHCOMM_AVAILABLE
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_distributed import C10dTorchCommsTestBase
from torch.testing._internal.common_utils import (
    find_free_port,
    parametrize,
    run_tests,
    subtest,
    TestCase,
)


@unittest.skipIf(not _TORCHCOMM_AVAILABLE, "TorchComms is not installed")
class TestC10dTorchCommsBasic(C10dTorchCommsTestBase):
    REDUCE_OPS = [
        subtest(dist.ReduceOp.SUM, name="SUM"),
        subtest(dist.ReduceOp.AVG, name="AVG"),
        subtest(dist.ReduceOp.MIN, name="MIN"),
        subtest(dist.ReduceOp.MAX, name="MAX"),
        subtest(dist.ReduceOp.PRODUCT, name="PRODUCT"),
    ]

    @property
    def _rank_value(self):
        return self.rank + 1

    def _requires_cuda(self):
        """Return True when the test variant is NOT cuda.

        MultiProcContinuousTest workers wrap unittest.SkipTest as RuntimeError,
        so @onlyCUDA / self.skipTest() poison the entire class.  Tests that
        need NCCL should call this and ``return`` early instead.
        """
        return self.device_type != "cuda"

    def _skip_if_product_overflows(self, op):
        if op == dist.ReduceOp.PRODUCT and self.world_size > 12:
            self.skipTest(
                f"world_size={self.world_size} > 12: PRODUCT is world_size! "
                "and only up to 12! is exactly representable in float32"
            )

    def _expected_reduce_result(self, op):
        """Return the expected scalar result for a rank+1 input reduced across all ranks."""
        total = sum(range(1, self.world_size + 1))
        if op == dist.ReduceOp.SUM:
            return total
        elif op == dist.ReduceOp.AVG:
            return total / self.world_size
        elif op == dist.ReduceOp.MIN:
            return 1
        elif op == dist.ReduceOp.MAX:
            return self.world_size
        elif op == dist.ReduceOp.PRODUCT:
            result = 1
            for i in range(1, self.world_size + 1):
                result *= i
            return result
        raise ValueError(f"Unsupported op: {op}")

    @parametrize("op", REDUCE_OPS)
    def test_allreduce(self, op):
        self._skip_if_product_overflows(op)
        tensor = torch.tensor([self._rank_value], dtype=torch.float32)
        dist.all_reduce(tensor, op=op, group=self.pg)
        self.assertEqual(tensor.item(), self._expected_reduce_result(op))

    def test_all_gather(self):
        input_tensor = torch.tensor([self._rank_value], dtype=torch.float32)
        gather_list = [torch.empty_like(input_tensor) for _ in range(self.world_size)]
        dist.all_gather(gather_list, input_tensor, group=self.pg)
        expected = list(range(1, self.world_size + 1))
        self.assertEqual([t.item() for t in gather_list], expected)

    def test_all_gather_into_tensor(self):
        input_tensor = torch.tensor([self._rank_value], dtype=torch.float32)
        output_tensor = torch.empty(self.world_size, dtype=torch.float32)
        dist.all_gather_single(output_tensor, input_tensor, group=self.pg)
        expected = list(range(1, self.world_size + 1))
        self.assertEqual([t.item() for t in output_tensor], expected)

    def test_broadcast(self):
        tensor = torch.tensor([self._rank_value], dtype=torch.float32)
        dist.broadcast(tensor, src=0, group=self.pg)
        self.assertEqual(tensor.item(), 1)

    def test_gather(self):
        tensor = torch.tensor([self._rank_value], dtype=torch.float32)
        gather_list = None
        if self.rank == 0:
            gather_list = [torch.empty_like(tensor) for _ in range(self.world_size)]
        dist.gather(tensor, gather_list=gather_list, dst=0, group=self.pg)
        if self.rank == 0:
            expected = list(range(1, self.world_size + 1))
            self.assertEqual([t.item() for t in gather_list], expected)

    def test_scatter(self):
        if self.rank == 0:
            scatter_list = [
                torch.tensor([i], dtype=torch.float32) for i in range(self.world_size)
            ]
        else:
            scatter_list = None
        tensor = torch.empty(1, dtype=torch.float32)
        dist.scatter(tensor, scatter_list=scatter_list, src=0, group=self.pg)
        self.assertEqual(tensor.item(), self.rank)

    @parametrize("op", REDUCE_OPS)
    def test_reduce(self, op):
        self._skip_if_product_overflows(op)
        input_tensor = torch.tensor([self._rank_value], dtype=torch.float32)
        dist.reduce(input_tensor, dst=0, op=op, group=self.pg)
        if self.rank == 0:
            self.assertEqual(input_tensor.item(), self._expected_reduce_result(op))

    @parametrize("op", REDUCE_OPS)
    def test_reduce_scatter(self, op):
        self._skip_if_product_overflows(op)
        input_tensor = [
            torch.tensor([self._rank_value], dtype=torch.float32)
            for _ in range(self.world_size)
        ]
        output_tensor = torch.empty(1, dtype=torch.float32)
        dist.reduce_scatter(output_tensor, input_tensor, op=op, group=self.pg)
        self.assertEqual(output_tensor.item(), self._expected_reduce_result(op))

    @parametrize("op", REDUCE_OPS)
    def test_reduce_scatter_tensor(self, op):
        self._skip_if_product_overflows(op)
        input_tensor = torch.full(
            (self.world_size,), self._rank_value, dtype=torch.float32
        )
        output_tensor = torch.empty(1, dtype=torch.float32)
        dist.reduce_scatter_single(output_tensor, input_tensor, op=op, group=self.pg)
        self.assertEqual(output_tensor.item(), self._expected_reduce_result(op))

    def test_all_to_all(self):
        input_tensor = [
            torch.tensor([self._rank_value], dtype=torch.float32)
            for _ in range(self.world_size)
        ]
        output_tensor = [
            torch.empty(1, dtype=torch.float32) for _ in range(self.world_size)
        ]
        dist.all_to_all(output_tensor, input_tensor, group=self.pg)
        expected = list(range(1, self.world_size + 1))
        self.assertEqual([t.item() for t in output_tensor], expected)

    def test_all_to_all_single(self):
        input_tensor = torch.full(
            (self.world_size,), self._rank_value, dtype=torch.float32
        )
        output_tensor = torch.empty([self.world_size], dtype=torch.float32)
        dist.all_to_all_single(output_tensor, input_tensor, group=self.pg)
        expected = list(range(1, self.world_size + 1))
        self.assertEqual([t.item() for t in output_tensor], expected)

    def test_all_to_all_single_with_split_sizes(self):
        # Each rank sends (rank + 1) elements to every other rank,
        # so rank r's input_split_sizes are all (rank + 1).
        input_split_sizes = [self._rank_value] * self.world_size
        # Rank r receives (sender_rank + 1) elements from each sender,
        # so output_split_sizes[i] = i + 1.
        output_split_sizes = [i + 1 for i in range(self.world_size)]

        input_tensor = torch.empty(sum(input_split_sizes), dtype=torch.float32)
        offset = 0
        for dst in range(self.world_size):
            input_tensor[offset : offset + input_split_sizes[dst]].fill_(
                self.rank + dst
            )
            offset += input_split_sizes[dst]

        output_tensor = torch.empty(sum(output_split_sizes), dtype=torch.float32)
        dist.all_to_all_single(
            output_tensor,
            input_tensor,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=self.pg,
        )

        # Verify: section from sender i should contain value (i + rank)
        offset = 0
        for src in range(self.world_size):
            section = output_tensor[offset : offset + output_split_sizes[src]]
            expected = torch.full_like(section, src + self.rank)
            self.assertTrue(
                torch.equal(section, expected),
                lambda msg: f"{msg}\nMismatch in section from rank {src}: got {section}, expected {expected}",
            )
            offset += output_split_sizes[src]

    def test_send_recv(self):
        send_rank = (self.rank + 1) % self.world_size
        recv_rank = (self.rank + self.world_size - 1) % self.world_size
        send_tensor = torch.tensor([self.rank], dtype=torch.float32)
        recv_tensor = torch.empty(1, dtype=torch.float32)
        if self.rank % 2 == 0:
            # Even ranks: send first, then receive
            dist.send(send_tensor, dst=send_rank, group=self.pg)
            dist.recv(recv_tensor, src=recv_rank, group=self.pg)
        else:
            # Odd ranks: receive first, then send
            dist.recv(recv_tensor, src=recv_rank, group=self.pg)
            dist.send(send_tensor, dst=send_rank, group=self.pg)
        # Each rank receives the rank number of the sender
        self.assertEqual(recv_tensor.item(), recv_rank)

    def test_barrier(self):
        dist.barrier(group=self.pg)
        # If we reach this point, the barrier succeeded without deadlock
        self.assertTrue(True)

    def test_new_group_delegates_to_split_group(self):
        # Under torchcomms, `new_group` routes through `split_group`. The
        # resulting subgroup must contain the requested ranks and be usable
        # for collectives.
        subg_ranks = list(range(self.world_size // 2))
        ng = dist.new_group(ranks=subg_ranks)

        if self.rank in subg_ranks:
            self.assertEqual(dist.get_process_group_ranks(ng), subg_ranks)
            tensor = torch.tensor([self._rank_value], dtype=torch.float32)
            dist.all_reduce(tensor, group=ng)
            self.assertEqual(tensor.item(), sum(r + 1 for r in subg_ranks))
        else:
            self.assertIs(ng, dist.GroupMember.NON_GROUP_MEMBER)

    def test_new_group_backend_none_narrows_to_default_device(self):
        ranks = list(range(self.world_size))
        ng = dist.new_group(ranks=ranks, backend=None)
        tensor = torch.tensor([self._rank_value], dtype=torch.float32)
        dist.all_reduce(tensor, group=ng)
        self.assertEqual(tensor.item(), sum(range(1, self.world_size + 1)))

    def test_new_group_bare_default_backend_is_auto_qualified(self):
        if self._requires_cuda():
            return
        ranks = list(range(self.world_size))
        ng = dist.new_group(ranks=ranks, backend="nccl")
        tensor = torch.tensor([self._rank_value], dtype=torch.float32)
        dist.all_reduce(tensor, group=ng)
        self.assertEqual(tensor.item(), sum(range(1, self.world_size + 1)))

    def test_new_group_qualified_backend_passes_through(self):
        if self._requires_cuda():
            return
        ranks = list(range(self.world_size))
        ng = dist.new_group(ranks=ranks, backend="cuda:nccl")
        tensor = torch.tensor([self._rank_value], dtype=torch.float32)
        dist.all_reduce(tensor, group=ng)
        self.assertEqual(tensor.item(), sum(range(1, self.world_size + 1)))

    def test_new_group_with_pg_options(self):
        if self._requires_cuda():
            return
        ranks = list(range(self.world_size))
        opts = dist.ProcessGroupNCCL.Options(is_high_priority_stream=True)
        opts.config.cga_cluster_size = 2
        opts.config.max_ctas = 16
        ng = dist.new_group(ranks=ranks, pg_options=opts)
        tensor = torch.tensor([self._rank_value], dtype=torch.float32)
        dist.all_reduce(tensor, group=ng)
        self.assertEqual(tensor.item(), sum(range(1, self.world_size + 1)))

    def test_new_group_sequential_pg_options_produce_distinct_groups(self):
        if self._requires_cuda():
            return
        ranks = list(range(self.world_size))
        opts_a = dist.ProcessGroupNCCL.Options(is_high_priority_stream=True)
        opts_a.config.cga_cluster_size = 2
        opts_b = dist.ProcessGroupNCCL.Options()
        opts_b.config.cga_cluster_size = 4
        g_a = dist.new_group(ranks=ranks, pg_options=opts_a)
        g_b = dist.new_group(ranks=ranks, pg_options=opts_b)
        self.assertNotEqual(g_a.group_name, g_b.group_name)


devices = ["cpu", "cuda", "xpu"]
instantiate_device_type_tests(
    TestC10dTorchCommsBasic, globals(), only_for=devices, allow_xpu=True
)


@unittest.skipIf(not _TORCHCOMM_AVAILABLE, "TorchComms is not installed")
class TestC10dTorchCommsInitAutoQualify(C10dTorchCommsTestBase):
    """Verify init_process_group auto-qualifies bare backends under torchcomms.

    Overrides ``_init_pg`` to pass ``device_id`` with bare ``"nccl"`` —
    the auto-qualify logic in ``init_process_group`` should expand it to
    ``"cpu:gloo,cuda:nccl"`` so both CPU and CUDA backends are available.
    """

    @classmethod
    def _init_pg(cls, rank, world_size, rdvz_file):
        torch.distributed.config.use_torchcomms = True
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["TORCHCOMM_STORE_PATH"] = rdvz_file
        os.environ["LOCAL_RANK"] = str(rank)

        store = dist.FileStore(rdvz_file, world_size)
        device_id = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(rank)

        dist.init_process_group(
            backend="nccl",
            world_size=world_size,
            rank=rank,
            store=store,
            device_id=device_id,
        )
        cls.pg = dist.distributed_c10d._get_default_group()
        torch.set_default_device(device_id)

    @property
    def _rank_value(self):
        return self.rank + 1

    def test_default_pg_has_cpu_backend(self):
        default_pg = dist.distributed_c10d._get_default_group()
        cpu_be = default_pg._get_backend(torch.device("cpu"))
        self.assertIsNotNone(cpu_be)

    def test_default_pg_has_cuda_backend(self):
        default_pg = dist.distributed_c10d._get_default_group()
        cuda_be = default_pg._get_backend(torch.device("cuda"))
        self.assertIsNotNone(cuda_be)

    def test_allreduce_on_auto_qualified_pg(self):
        tensor = torch.tensor([self._rank_value], dtype=torch.float32)
        dist.all_reduce(tensor, group=self.pg)
        self.assertEqual(tensor.item(), sum(range(1, self.world_size + 1)))

    def test_new_group_from_auto_qualified_parent(self):
        ranks = list(range(self.world_size))
        ng = dist.new_group(ranks=ranks)
        tensor = torch.tensor([self._rank_value], dtype=torch.float32)
        dist.all_reduce(tensor, group=ng)
        self.assertEqual(tensor.item(), sum(range(1, self.world_size + 1)))

    def test_bound_device_id_is_set(self):
        default_pg = dist.distributed_c10d._get_default_group()
        self.assertIsNotNone(default_pg.bound_device_id)
        self.assertEqual(default_pg.bound_device_id.type, "cuda")

    def test_new_group_nccl_lazy_builds_per_peer_group(self):
        # Passing backend="nccl-lazy" builds a per-peer, lazily-initialized
        # group (a dedicated comm + stream per send/recv peer) usable for P2P.
        # use_local_synchronization makes it members-only. The parent must be
        # device-bound (it is, in this class).
        ranks = list(range(self.world_size))
        g = dist.new_group(
            ranks=ranks, backend="nccl-lazy", use_local_synchronization=True
        )
        self.assertEqual(dist.get_process_group_ranks(g), ranks)
        # P2P round-trip over the lazy group: 0 -> 1 -> ... -> 0
        dev = torch.device(f"cuda:{self.rank}")
        send_to = (self.rank + 1) % self.world_size
        recv_from = (self.rank - 1) % self.world_size
        if self.rank % 2 == 0:
            dist.send(
                torch.full((4,), float(self.rank), device=dev), dst=send_to, group=g
            )
            r = torch.empty(4, device=dev)
            dist.recv(r, src=recv_from, group=g)
        else:
            r = torch.empty(4, device=dev)
            dist.recv(r, src=recv_from, group=g)
            dist.send(
                torch.full((4,), float(self.rank), device=dev), dst=send_to, group=g
            )
        self.assertEqual(r[0].item(), float(recv_from))

    def test_non_torchcomms_backend_falls_through_to_c10d(self):
        # Under torchcomms, a backend TorchComms does not own (registered the way
        # mooncake registers a custom c10d backend) must route through the normal
        # ProcessGroup path, not new_comm. Use a gloo-backed stand-in.
        def _creator(store, grank, gsize, timeout):
            return dist.ProcessGroupGloo(store, grank, gsize, timeout)

        name = "tc_gate_stub"
        if name not in dist.Backend.backend_list:
            dist.Backend.register_backend(name, _creator, devices=["cpu", "cuda"])

        ranks = list(range(self.world_size))
        g = dist.new_group(ranks=ranks, backend=name)
        be = g._get_backend(torch.device("cpu"))
        # The c10d creator ran (real ProcessGroupGloo), not a TorchComms wrapper.
        self.assertNotIn("BackendWrapper", type(be).__name__)
        t = torch.tensor([self._rank_value], dtype=torch.float32)
        dist.all_reduce(t, group=g)
        self.assertEqual(t.item(), sum(range(1, self.world_size + 1)))


instantiate_device_type_tests(
    TestC10dTorchCommsInitAutoQualify, globals(), only_for=["cuda"]
)


@unittest.skipIf(not _TORCHCOMM_AVAILABLE, "TorchComms is not installed")
class TestC10dTorchCommsMixedBackends(C10dTorchCommsTestBase):
    """Verify subgroup creation from a mixed-backend parent under torchcomms.

    The parent PG mixes ``cuda:nccl`` with ``cpu:fake``. Under torchcomms,
    ``new_group`` builds each subgroup directly via ``new_comm`` (a members-only
    store rendezvous), so a non-splittable device backend in the parent (here
    ``cpu:fake``) is no obstacle: members construct the subgroup and non-members
    return ``NON_GROUP_MEMBER`` without any parent split.
    """

    @classmethod
    def _init_pg(cls, rank, world_size, rdvz_file):
        torch.distributed.config.use_torchcomms = True
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["TORCHCOMM_STORE_PATH"] = rdvz_file
        os.environ["LOCAL_RANK"] = str(rank)

        store = dist.FileStore(rdvz_file, world_size)
        device_id = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(rank)

        dist.init_process_group(
            backend="cuda:nccl,cpu:fake",
            world_size=world_size,
            rank=rank,
            store=store,
            device_id=device_id,
        )
        cls.pg = dist.distributed_c10d._get_default_group()
        torch.set_default_device(device_id)

    @property
    def _rank_value(self):
        return self.rank + 1

    def test_mixed_backend_subgroup(self):
        subg_ranks = list(range(self.world_size // 2))
        ng = dist.new_group(ranks=subg_ranks)

        if self.rank in subg_ranks:
            self.assertEqual(dist.get_process_group_ranks(ng), subg_ranks)
            tensor = torch.tensor([self._rank_value], dtype=torch.float32)
            dist.all_reduce(tensor, group=ng)
            self.assertEqual(tensor.item(), sum(r + 1 for r in subg_ranks))
        else:
            self.assertIs(ng, dist.GroupMember.NON_GROUP_MEMBER)


instantiate_device_type_tests(
    TestC10dTorchCommsMixedBackends, globals(), only_for=["cuda"]
)


class TestTorchCommsHandlesBackend(TestCase):
    """Unit-test the pure routing logic of ``_torchcomms_handles_backend``.

    This decides whether a backend is one TorchComms can construct (so it goes
    through ``new_comm`` / ``split_group``) versus a custom c10d plugin such as
    ``mooncake`` that must fall through to the normal ProcessGroup path. The
    function depends only on ``_TORCHCOMM_AVAILABLE`` and the two
    ``is_backend_*`` callables, all of which we patch -- so these tests run even
    when torchcomms is not installed.
    """

    def _patch(self, available=True, built=(), registered=()):
        built, registered = set(built), set(registered)
        return mock.patch.multiple(
            c10d,
            _TORCHCOMM_AVAILABLE=available,
            _torchcomms_is_backend_built=lambda name: name in built,
            _torchcomms_is_backend_registered=lambda name: name in registered,
            create=True,
        )

    def test_unavailable_returns_false(self):
        with self._patch(available=False, built=["nccl"]):
            self.assertFalse(c10d._torchcomms_handles_backend("nccl"))
            self.assertFalse(c10d._torchcomms_handles_backend(None))

    def test_none_backend_inherits_parent(self):
        with self._patch(built=[]):
            self.assertTrue(c10d._torchcomms_handles_backend(None))

    def test_builtin_backend(self):
        with self._patch(built=["nccl"]):
            self.assertTrue(c10d._torchcomms_handles_backend("nccl"))

    def test_registered_backend(self):
        with self._patch(registered=["myadapter"]):
            self.assertTrue(c10d._torchcomms_handles_backend("myadapter"))

    def test_custom_backend_falls_through(self):
        with self._patch(built=["nccl"]):
            self.assertFalse(c10d._torchcomms_handles_backend("mooncake"))
            self.assertFalse(c10d._torchcomms_handles_backend("ucc"))

    def test_case_insensitive(self):
        with self._patch(built=["nccl"]):
            self.assertTrue(c10d._torchcomms_handles_backend("NCCL"))

    def test_nccl_lazy_is_own_backend(self):
        # nccl-lazy is a distinct built torchcomms backend, not aliased to nccl:
        # it is handled iff nccl-lazy itself is built/registered.
        with self._patch(built=["nccl-lazy"]):
            self.assertTrue(c10d._torchcomms_handles_backend("nccl-lazy"))
        with self._patch(built=["nccl"]):
            self.assertFalse(c10d._torchcomms_handles_backend("nccl-lazy"))
        with self._patch(built=[]):
            self.assertFalse(c10d._torchcomms_handles_backend("nccl-lazy"))

    def test_qualified_all_handled(self):
        with self._patch(built=["gloo", "nccl"]):
            self.assertTrue(c10d._torchcomms_handles_backend("cpu:gloo,cuda:nccl"))

    def test_qualified_one_unhandled_falls_through(self):
        with self._patch(built=["gloo"]):
            self.assertFalse(c10d._torchcomms_handles_backend("cpu:gloo,cuda:mooncake"))

    def test_empty_parts_are_skipped(self):
        with self._patch(built=["nccl"]):
            self.assertTrue(c10d._torchcomms_handles_backend("nccl,"))
            self.assertTrue(c10d._torchcomms_handles_backend(" nccl , "))


class TestC10dTorchCommsNewGroupHelper(TestCase):
    """Unit-test the TorchComms-specific branches of ``_new_process_group_helper``.

    These cover the three subgroup-init fixes: the device handed to ``new_comm``
    carries this rank's device index (``device_id``) rather than a device-type-only
    device; ``TORCHCOMM_RANK``/``TORCHCOMM_SIZE`` are seeded from the group's
    rank/size around the ``new_comm`` call and restored afterwards; and a
    non-member of a subgroup must NOT issue a no-color parent split under
    TorchComms. Everything TorchComms-specific is patched (``create=True``), so
    these run even when TorchComms is not installed.
    """

    TIMEOUT = datetime.timedelta(seconds=30)

    def _drive_member(self, *, backend, device_id, group_rank, group_size):
        """Drive the members path down to ``new_comm``.

        ``new_comm`` is mocked to capture its device argument and the live env,
        then abort with a sentinel so we never touch real comm machinery. Uses
        the default-group path (``global_ranks_in_group == []``) so no
        initialized world is required. Returns the captured dict.
        """
        captured = {}

        def fake_new_comm(backend_str, device, name=None, store=None, hints=None):
            captured["backend_str"] = backend_str
            captured["device"] = device
            captured["rank_env"] = os.environ.get("TORCHCOMM_RANK")
            captured["size_env"] = os.environ.get("TORCHCOMM_SIZE")
            raise RuntimeError("stop-after-new_comm")

        with mock.patch.multiple(
            c10d,
            _use_torchcomms_enabled=lambda: True,
            _torchcomms_handles_backend=lambda b: True,
            new_comm=fake_new_comm,
            create=True,
        ):
            with self.assertRaisesRegex(RuntimeError, "stop-after-new_comm"):
                c10d._new_process_group_helper(
                    group_size=group_size,
                    group_rank=group_rank,
                    global_ranks_in_group=[],
                    backend=backend,
                    store=dist.HashStore(),
                    group_name=c10d.GroupName(self.id()),
                    timeout=self.TIMEOUT,
                    device_id=device_id,
                )
        return captured

    def test_new_comm_gets_indexed_device_id(self):
        # A subgroup's group-local rank differs from the rank's physical device,
        # so new_comm must receive device_id (WITH index), not a device-type-only
        # device. group_rank (2) deliberately differs from the device index (3).
        cap = self._drive_member(
            backend="nccl",
            device_id=torch.device("cuda:3"),
            group_rank=2,
            group_size=4,
        )
        self.assertEqual(cap["device"], torch.device("cuda:3"))

    def test_new_comm_device_type_mismatch_not_overridden(self):
        # gloo maps to the cpu device; device_id is a cuda device, so the type
        # guard must leave cpu alone rather than substituting cuda:3.
        cap = self._drive_member(
            backend="gloo",
            device_id=torch.device("cuda:3"),
            group_rank=1,
            group_size=4,
        )
        self.assertEqual(cap["device"], torch.device("cpu"))

    def test_new_comm_without_device_id_keeps_type_only_device(self):
        # World-group path: no device_id bound, so the guard must not fire and
        # new_comm gets the device-type-only device from the backend map.
        cap = self._drive_member(
            backend="nccl",
            device_id=None,
            group_rank=0,
            group_size=4,
        )
        self.assertEqual(cap["device"], torch.device("cuda"))

    def test_torchcomm_rank_size_seeded_from_group(self):
        cap = self._drive_member(
            backend="nccl",
            device_id=torch.device("cuda:1"),
            group_rank=2,
            group_size=7,
        )
        self.assertEqual(cap["rank_env"], "2")
        self.assertEqual(cap["size_env"], "7")

    def test_torchcomm_rank_size_restored_after_call(self):
        saved = {k: os.environ.get(k) for k in ("TORCHCOMM_RANK", "TORCHCOMM_SIZE")}
        try:
            for k in ("TORCHCOMM_RANK", "TORCHCOMM_SIZE"):
                os.environ.pop(k, None)
            # Unset before -> unset after.
            self._drive_member(
                backend="nccl",
                device_id=torch.device("cuda:0"),
                group_rank=0,
                group_size=2,
            )
            self.assertIsNone(os.environ.get("TORCHCOMM_RANK"))
            self.assertIsNone(os.environ.get("TORCHCOMM_SIZE"))
            # Set before -> original values restored after.
            os.environ["TORCHCOMM_RANK"] = "99"
            os.environ["TORCHCOMM_SIZE"] = "77"
            self._drive_member(
                backend="nccl",
                device_id=torch.device("cuda:0"),
                group_rank=0,
                group_size=2,
            )
            self.assertEqual(os.environ["TORCHCOMM_RANK"], "99")
            self.assertEqual(os.environ["TORCHCOMM_SIZE"], "77")
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def _drive_non_member(self, *, torchcomms_enabled):
        """Drive the non-member early-return path of a subgroup.

        The default group is faked as initialized and device-bound with this
        rank (0) absent from the requested subgroup, so ``_new_process_group_helper``
        takes the ``NON_GROUP_MEMBER`` branch. Returns (result, split_source_mock).
        """
        split_src = mock.MagicMock()
        default = mock.MagicMock()
        default.rank.return_value = 0
        default.bound_device_id = torch.device("cuda:0")
        with mock.patch.multiple(
            c10d,
            _use_torchcomms_enabled=lambda: torchcomms_enabled,
            is_initialized=lambda: True,
            _get_default_group=lambda: default,
            _get_split_source=lambda pg: split_src,
        ):
            res = c10d._new_process_group_helper(
                group_size=2,
                group_rank=0,
                global_ranks_in_group=[1, 2],  # rank 0 is NOT a member
                backend="nccl",
                store=dist.HashStore(),
                group_name=c10d.GroupName(self.id()),
                timeout=self.TIMEOUT,
            )
        return res, split_src

    def test_non_member_skips_nocolor_split_under_torchcomms(self):
        res, split_src = self._drive_non_member(torchcomms_enabled=True)
        self.assertEqual(res, (dist.GroupMember.NON_GROUP_MEMBER, None))
        split_src.perform_nocolor_split.assert_not_called()

    def test_non_member_performs_nocolor_split_without_torchcomms(self):
        # Contrast: with TorchComms disabled the NCCL-style path still requires
        # non-members to issue the no-color split to stay in sync.
        res, split_src = self._drive_non_member(torchcomms_enabled=False)
        self.assertEqual(res, (dist.GroupMember.NON_GROUP_MEMBER, None))
        split_src.perform_nocolor_split.assert_called_once_with(torch.device("cuda:0"))


class TestC10dGroupNameHashSalt(TestCase):
    """Unit-test the collective-consistent group-name hash salt.

    ``_hash_ranks_to_str`` / ``_process_group_name`` are generic c10d helpers
    (not TorchComms-specific), but the divergence they can produce was surfaced
    by the TorchComms subgroup work. After an earlier asymmetric-membership
    ``new_group`` (e.g. ``new_group([0])``), member and non-member ranks hold
    different ``len(_world.pg_names)`` because only members register a PG.
    Salting the hash with that length makes ranks compute DIFFERENT names for the
    same later group; backends that use the group name as a rendezvous store
    prefix (e.g. Gloo split's ``connectFullMesh``) then key off mismatched
    prefixes and deadlock. The salt must instead be ``_world.group_count``, which
    every rank advances in lockstep -- ``_process_group_name`` increments it
    before the membership check, on BOTH the hashed and non-hashed paths.
    """

    def setUp(self):
        super().setUp()
        self._saved_group_count = c10d._world.group_count
        self._saved_pg_names = dict(c10d._world.pg_names)

    def tearDown(self):
        c10d._world.group_count = self._saved_group_count
        c10d._world.pg_names.clear()
        c10d._world.pg_names.update(self._saved_pg_names)
        super().tearDown()

    def _register_dummy_pgs(self, n):
        # Simulate a rank that registered extra PGs (asymmetric membership).
        for i in range(n):
            c10d._world.pg_names[object()] = c10d.GroupName(f"dummy_{i}")

    def test_hash_salt_uses_group_count(self):
        # Same ranks, different group_count -> different name (salt is live).
        ranks = [0, 1, 2, 3]
        c10d._world.group_count = 5
        name_a = c10d._hash_ranks_to_str(ranks)
        c10d._world.group_count = 6
        name_b = c10d._hash_ranks_to_str(ranks)
        self.assertNotEqual(name_a, name_b)

    def test_hash_independent_of_pg_names_length(self):
        # ROOT-CAUSE regression: the hash must NOT depend on len(_world.pg_names),
        # the quantity that diverges across ranks under asymmetric membership.
        ranks = [0, 1, 2, 3]
        c10d._world.group_count = 9
        c10d._world.pg_names.clear()
        name_few = c10d._hash_ranks_to_str(ranks)
        self._register_dummy_pgs(5)
        name_many = c10d._hash_ranks_to_str(ranks)
        self.assertEqual(name_few, name_many)

    def test_two_asymmetric_ranks_agree_on_name(self):
        # The deadlock scenario end-to-end through _process_group_name: two ranks
        # that made the SAME sequence of group-creation calls (equal group_count)
        # but registered a DIFFERENT number of PGs must compute the same name.
        ranks = [0, 1, 2, 3]
        # Rank that was a member of earlier subgroups: more pg_names.
        c10d._world.group_count = 3
        c10d._world.pg_names.clear()
        self._register_dummy_pgs(2)
        name_member = c10d._process_group_name(ranks, use_hashed_name=True)
        # Rank that was a non-member: fewer pg_names, but same group_count because
        # every rank advances it in lockstep regardless of membership.
        c10d._world.group_count = 3
        c10d._world.pg_names.clear()
        name_non_member = c10d._process_group_name(ranks, use_hashed_name=True)
        self.assertEqual(name_member, name_non_member)

    def test_process_group_name_increments_on_hashed_path(self):
        c10d._world.group_count = 10
        c10d._process_group_name([0, 1], use_hashed_name=True)
        self.assertEqual(c10d._world.group_count, 11)

    def test_process_group_name_increments_on_nonhashed_path(self):
        c10d._world.group_count = 20
        c10d._process_group_name([0, 1], use_hashed_name=False)
        self.assertEqual(c10d._world.group_count, 21)

    def test_nonhashed_name_is_group_count_before_increment(self):
        c10d._world.group_count = 42
        name = c10d._process_group_name([0, 1], use_hashed_name=False)
        self.assertEqual(name, "42")
        self.assertEqual(c10d._world.group_count, 43)

    def test_repeated_hashed_same_ranks_get_distinct_names(self):
        # Two distinct PGs over identical ranks must get distinct names; the salt
        # advancing on the hashed path (fixed by this commit) is what
        # disambiguates them. Before the fix the hashed path left group_count
        # unchanged, so back-to-back splits over the same ranks collided.
        c10d._world.group_count = 0
        first = c10d._process_group_name([2, 3], use_hashed_name=True)
        second = c10d._process_group_name([2, 3], use_hashed_name=True)
        self.assertNotEqual(first, second)


if __name__ == "__main__":
    run_tests()
