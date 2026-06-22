# Owner(s): ["oncall: distributed"]

import unittest

import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import _TORCHCOMM_AVAILABLE
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_distributed import C10dTorchCommsTestBase
from torch.testing._internal.common_utils import (
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
                f"Mismatch in section from rank {src}: got {section}, expected {expected}",
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

    def test_new_group_via_split_group_raises_on_unsupported_args(self):
        # `split_group` has a narrower surface than `new_group`; under
        # torchcomms the delegation must surface that mismatch instead of
        # silently falling back to the legacy path.
        ranks = list(range(self.world_size))
        with self.assertRaisesRegex(NotImplementedError, "use_local_synchronization"):
            dist.new_group(ranks=ranks, use_local_synchronization=True)
        with self.assertRaisesRegex(NotImplementedError, "sort_ranks"):
            dist.new_group(ranks=ranks, sort_ranks=False)

    # The next block of tests covers
    # ``new_group``'s torchcomms behaviors: auto-qualify of bare backends,
    # ``backend=None`` narrowing to the parent's default-device backend,
    # and the ``pg_options`` → ``_new_group_with_tag`` bypass dispatch
    # that lets NCCL Options translate to torchcomms hints applied at
    # comm construction.

    def test_new_group_backend_none_narrows_to_default_device(self, device):
        # ``new_group(backend=None)`` under torchcomms must auto-narrow to
        # the parent's default-device backend (e.g. ``"cuda:nccl"``) so
        # the subgroup doesn't pick up a free gloo comm.
        ng = dist.new_group(ranks=list(range(self.world_size)), backend=None)
        tensor = torch.tensor([self._rank_value], dtype=torch.float32, device=device)
        dist.all_reduce(tensor, group=ng)
        self.assertEqual(tensor.item(), sum(range(1, self.world_size + 1)))

    def test_new_group_bare_default_backend_is_auto_qualified(self, device):
        # Bare ``backend="nccl"`` (or whatever the parent's default-device
        # backend is) must be auto-qualified to ``"<device>:<backend>"``.
        ng = dist.new_group(
            ranks=list(range(self.world_size)),
            backend=self.backend(device),
        )
        tensor = torch.tensor([self._rank_value], dtype=torch.float32, device=device)
        dist.all_reduce(tensor, group=ng)
        self.assertEqual(tensor.item(), sum(range(1, self.world_size + 1)))

    def test_new_group_qualified_backend_passes_through(self, device):
        # An already-qualified backend (``"cuda:nccl"`` etc.) must pass
        # through unchanged.
        qualified = f"{device}:{self.backend(device)}"
        ng = dist.new_group(ranks=list(range(self.world_size)), backend=qualified)
        tensor = torch.tensor([self._rank_value], dtype=torch.float32, device=device)
        dist.all_reduce(tensor, group=ng)
        self.assertEqual(tensor.item(), sum(range(1, self.world_size + 1)))

    def test_new_group_pg_options_routes_through_helper_and_works(self, device):
        # ``pg_options=ProcessGroupNCCL.Options(...)`` whose fields translate
        # to torchcomms hints (high_priority_stream / cga_cluster_size /
        # max_ctas / min_ctas) must bypass ``split_group`` (which can't
        # consume NCCL Options) and route through ``_new_group_with_tag``
        # -> ``_new_process_group_helper`` so the hints reach
        # ``torchcomms.new_comm(hints=...)``. The resulting PG must be
        # functional for collectives.
        if "cuda" not in device:
            self.skipTest("pg_options→hints path is NCCL-specific")
        opts = dist.ProcessGroupNCCL.Options(is_high_priority_stream=True)
        opts.config.cga_cluster_size = 2
        opts.config.max_ctas = 16
        ng = dist.new_group(
            ranks=list(range(self.world_size)),
            pg_options=opts,
            group_desc="TEST_PG_OPTIONS",
        )
        tensor = torch.tensor([self._rank_value], dtype=torch.float32, device=device)
        dist.all_reduce(tensor, group=ng)
        self.assertEqual(tensor.item(), sum(range(1, self.world_size + 1)))

    def test_new_group_pg_options_with_backend_none_still_narrows(self, device):
        # Regression guard: the auto-qualify of ``backend=None`` must apply
        # even when ``pg_options`` takes the bypass path. Otherwise a
        # ``cpu:gloo,cuda:nccl`` parent would create an extra gloo
        # subgroup comm per call.
        if "cuda" not in device:
            self.skipTest("pg_options→hints path is NCCL-specific")
        opts = dist.ProcessGroupNCCL.Options(is_high_priority_stream=True)
        ng = dist.new_group(
            ranks=list(range(self.world_size)),
            backend=None,
            pg_options=opts,
            group_desc="TEST_PG_OPTIONS_NONE_BACKEND",
        )
        tensor = torch.tensor([self._rank_value], dtype=torch.float32, device=device)
        dist.all_reduce(tensor, group=ng)
        self.assertEqual(tensor.item(), sum(range(1, self.world_size + 1)))

    def test_new_group_sequential_pg_options_produce_distinct_pgs(self, device):
        # Two consecutive ``new_group`` calls with different ``pg_options``
        # must produce distinct, independently-usable PGs (no caching of
        # the underlying comm across calls with different hints).
        if "cuda" not in device:
            self.skipTest("pg_options→hints path is NCCL-specific")
        opts_a = dist.ProcessGroupNCCL.Options(is_high_priority_stream=True)
        opts_a.config.cga_cluster_size = 2
        opts_b = dist.ProcessGroupNCCL.Options()
        opts_b.config.cga_cluster_size = 4
        g_a = dist.new_group(
            ranks=list(range(self.world_size)),
            pg_options=opts_a,
            group_desc="SEQ_A",
        )
        g_b = dist.new_group(
            ranks=list(range(self.world_size)),
            pg_options=opts_b,
            group_desc="SEQ_B",
        )
        self.assertIsNot(g_a, g_b)
        self.assertNotEqual(g_a.group_name, g_b.group_name)
        for g in (g_a, g_b):
            tensor = torch.tensor(
                [self._rank_value], dtype=torch.float32, device=device
            )
            dist.all_reduce(tensor, group=g)
            self.assertEqual(tensor.item(), sum(range(1, self.world_size + 1)))


class TestNcclOptionsToTorchCommsHints(TestCase):
    """Pure-fn tests for
    ``torch.distributed.distributed_c10d._nccl_options_to_torchcomms_hints``.

    Verifies the ``ProcessGroupNCCL.Options`` → ``Dict[str, str]``
    translation against the hint names torchcomms actually accepts
    (``high_priority_stream`` / ``cga_cluster_size`` / ``max_ctas`` /
    ``min_ctas``), and that NCCL_CONFIG_UNDEF_INT (-2**31) sentinel
    values are dropped.
    """

    def test_none_returns_empty_dict(self):
        from torch.distributed.distributed_c10d import (
            _nccl_options_to_torchcomms_hints,
        )

        self.assertEqual(_nccl_options_to_torchcomms_hints(None), {})

    def test_default_options_drops_sentinels(self):
        from torch.distributed.distributed_c10d import (
            _nccl_options_to_torchcomms_hints,
        )

        opts = dist.ProcessGroupNCCL.Options()
        # Fresh Options carries NCCL_CONFIG_UNDEF_INT sentinels for all the
        # int fields and is_high_priority_stream=False; the translator
        # should emit nothing.
        self.assertEqual(_nccl_options_to_torchcomms_hints(opts), {})

    def test_high_priority_stream_emitted_as_string_true(self):
        from torch.distributed.distributed_c10d import (
            _nccl_options_to_torchcomms_hints,
        )

        opts = dist.ProcessGroupNCCL.Options(is_high_priority_stream=True)
        hints = _nccl_options_to_torchcomms_hints(opts)
        self.assertEqual(hints.get("high_priority_stream"), "true")

    def test_config_fields_round_trip(self):
        from torch.distributed.distributed_c10d import (
            _nccl_options_to_torchcomms_hints,
        )

        opts = dist.ProcessGroupNCCL.Options(is_high_priority_stream=True)
        opts.config.cga_cluster_size = 2
        opts.config.max_ctas = 16
        opts.config.min_ctas = 4
        hints = _nccl_options_to_torchcomms_hints(opts)
        self.assertEqual(hints["high_priority_stream"], "true")
        self.assertEqual(hints["cga_cluster_size"], "2")
        self.assertEqual(hints["max_ctas"], "16")
        self.assertEqual(hints["min_ctas"], "4")

    def test_partial_config_omits_unset_fields(self):
        from torch.distributed.distributed_c10d import (
            _nccl_options_to_torchcomms_hints,
        )

        opts = dist.ProcessGroupNCCL.Options()
        opts.config.cga_cluster_size = 4
        hints = _nccl_options_to_torchcomms_hints(opts)
        self.assertEqual(hints, {"cga_cluster_size": "4"})

    def test_attributeerror_on_non_nccl_options_returns_empty(self):
        from torch.distributed.distributed_c10d import (
            _nccl_options_to_torchcomms_hints,
        )

        class Bogus:
            pass

        self.assertEqual(_nccl_options_to_torchcomms_hints(Bogus()), {})


devices = ["cpu", "cuda", "xpu"]
instantiate_device_type_tests(
    TestC10dTorchCommsBasic, globals(), only_for=devices, allow_xpu=True
)

if __name__ == "__main__":
    run_tests()
