# Owner(s): ["oncall: distributed"]

import itertools
import sys

import torch
import torch.distributed as dist


if not dist.is_available():
    print("distributed package not available, skipping tests", file=sys.stderr)
    sys.exit(0)

from c10d_backend_common import (
    C10D_BACKENDS,
    C10dBackendTest,
    C10dBackendTestContinuous,
    instantiate_backend_tests,
)

from torch.testing._internal.common_distributed import MultiProcContinuousTest
from torch.testing._internal.common_utils import run_tests


ASYNC_OPS = (False, True)
COUNTS = (0, 4)
FLOAT_REDUCE_OPS = (
    dist.ReduceOp.SUM,
    dist.ReduceOp.PRODUCT,
    dist.ReduceOp.MIN,
    dist.ReduceOp.MAX,
    dist.ReduceOp.AVG,
)
INTEGER_REDUCE_OPS = (
    dist.ReduceOp.SUM,
    dist.ReduceOp.PRODUCT,
    dist.ReduceOp.MIN,
    dist.ReduceOp.MAX,
    dist.ReduceOp.BAND,
    dist.ReduceOp.BOR,
    dist.ReduceOp.BXOR,
)


class CollectivesTestMixin:
    def _value(self, rank, dtype):
        if dtype == torch.bool:
            return bool(rank % 2)
        return rank + 1

    def _tensor(self, count, dtype, rank=None):
        rank = self.rank if rank is None else rank
        value = self._value(rank, dtype)
        return torch.full((count,), value, dtype=torch.float32, device=self.device).to(
            dtype
        )

    def _expected_tensor(self, count, dtype, rank):
        return self._tensor(count, dtype, rank=rank)

    def _distinct_tensor(self, src, dst, shape, memory_format=torch.contiguous_format):
        # Positive values distinct per element, source and destination rank.
        base = torch.arange(1, torch.Size(shape).numel() + 1, device=self.device)
        tensor = (base + 100 * src + 1000 * dst).float().view(shape)
        return tensor.contiguous(memory_format=memory_format)

    def _expected_distinct_reduce(self, shape, op):
        # Reduction over sources of what each rank sent to this rank.
        inputs = torch.stack(
            [
                self._distinct_tensor(src, self.rank, shape)
                for src in range(self.world_size)
            ]
        )
        if op == dist.ReduceOp.SUM:
            return inputs.sum(0)
        if op == dist.ReduceOp.PRODUCT:
            return inputs.prod(0)
        if op == dist.ReduceOp.MIN:
            return inputs.amin(0)
        if op == dist.ReduceOp.MAX:
            return inputs.amax(0)
        if op == dist.ReduceOp.AVG:
            return inputs.mean(0)
        if op == dist.ReduceOp.PREMUL_SUM:
            return (inputs * 0.5).sum(0)
        raise AssertionError(f"Unhandled reduce op: {op}")

    def _sentinel(self, shape, memory_format=torch.contiguous_format):
        # Never a valid result: distinct inputs and their reductions are
        # positive.
        return torch.full(shape, -1.0, device=self.device).contiguous(
            memory_format=memory_format
        )

    def _memory_formats(self, mixed):
        # Mixed: even ranks use channels_last, odd ranks contiguous.
        if mixed and self.rank % 2:
            return torch.contiguous_format
        return torch.channels_last

    def _wait(self, work, async_op):
        if async_op:
            self.assertIsNotNone(work)
            work.wait()
        else:
            self.assertIsNone(work)

    def _reduce_ops(self, dtype):
        if dtype.is_floating_point:
            return FLOAT_REDUCE_OPS
        if self.supports_bitwise_reductions:
            return INTEGER_REDUCE_OPS
        return INTEGER_REDUCE_OPS[:4]

    def _expected_reduce_value(self, dtype, op):
        values = [self._value(rank, dtype) for rank in range(self.world_size)]
        if op == dist.ReduceOp.SUM:
            result = sum(values)
        elif op == dist.ReduceOp.PRODUCT:
            result = 1
            for value in values:
                result *= value
        elif op == dist.ReduceOp.MIN:
            result = min(values)
        elif op == dist.ReduceOp.MAX:
            result = max(values)
        elif op == dist.ReduceOp.AVG:
            result = sum(values) / self.world_size
        elif op == dist.ReduceOp.BAND:
            result = values[0]
            for value in values[1:]:
                result &= value
        elif op == dist.ReduceOp.BOR:
            result = values[0]
            for value in values[1:]:
                result |= value
        elif op == dist.ReduceOp.BXOR:
            result = values[0]
            for value in values[1:]:
                result ^= value
        else:
            raise AssertionError(f"Unhandled reduce op: {op}")
        return result

    def _expected_reduce(self, count, dtype, op):
        value = self._expected_reduce_value(dtype, op)
        return torch.full((count,), value, dtype=torch.float32, device=self.device).to(
            dtype
        )

    def _test_broadcast(self, count, dtype, async_op):
        tensor = self._tensor(count, dtype)
        work = dist.broadcast(tensor, src=0, async_op=async_op)
        self._wait(work, async_op)
        self.assertEqual(tensor, self._expected_tensor(count, dtype, rank=0))

    def _test_all_gather(self, count, dtype, async_op):
        tensor = self._tensor(count, dtype)
        output = [torch.empty_like(tensor) for _ in range(self.world_size)]
        work = dist.all_gather(output, tensor, async_op=async_op)
        self._wait(work, async_op)
        for rank, result in enumerate(output):
            self.assertEqual(result, self._expected_tensor(count, dtype, rank))

    def _test_all_gather_single(self, count, dtype, async_op):
        tensor = self._tensor(count, dtype)
        output = torch.empty(self.world_size * count, dtype=dtype, device=self.device)
        work = dist.all_gather_single(output, tensor, async_op=async_op)
        self._wait(work, async_op)
        expected = torch.cat(
            [
                self._expected_tensor(count, dtype, rank)
                for rank in range(self.world_size)
            ]
        )
        self.assertEqual(output, expected)

    def _test_gather(self, count, dtype, async_op):
        tensor = self._tensor(count, dtype)
        output = (
            [torch.empty_like(tensor) for _ in range(self.world_size)]
            if self.rank == 0
            else None
        )
        work = dist.gather(tensor, output, dst=0, async_op=async_op)
        self._wait(work, async_op)
        if self.rank == 0:
            for rank, result in enumerate(output):
                self.assertEqual(result, self._expected_tensor(count, dtype, rank))

    def _test_gather_single(self, count, dtype, async_op):
        tensor = self._tensor(count, dtype)
        expected = [
            self._expected_tensor(count, dtype, rank) for rank in range(self.world_size)
        ]
        for stacked in (False, True):
            with self.subTest(stacked=stacked):
                shape = (
                    (self.world_size, count) if stacked else (self.world_size * count,)
                )
                output = (
                    torch.empty(shape, dtype=dtype, device=self.device)
                    if self.rank == 0
                    else None
                )
                work = dist.gather_single(tensor, output, dst=0, async_op=async_op)
                self._wait(work, async_op)
                if self.rank == 0:
                    self.assertEqual(
                        output,
                        torch.stack(expected) if stacked else torch.cat(expected),
                    )

    def _test_scatter(self, count, dtype, async_op):
        output = torch.empty(count, dtype=dtype, device=self.device)
        inputs = (
            [
                self._expected_tensor(count, dtype, rank)
                for rank in range(self.world_size)
            ]
            if self.rank == 0
            else None
        )
        work = dist.scatter(output, inputs, src=0, async_op=async_op)
        self._wait(work, async_op)
        self.assertEqual(output, self._expected_tensor(count, dtype, self.rank))

    def _test_all_to_all(self, count, dtype, async_op):
        inputs = [
            self._expected_tensor(count, dtype, self.rank)
            for _ in range(self.world_size)
        ]
        outputs = [torch.empty_like(inputs[0]) for _ in range(self.world_size)]
        work = dist.all_to_all(outputs, inputs, async_op=async_op)
        self._wait(work, async_op)
        for rank, result in enumerate(outputs):
            self.assertEqual(result, self._expected_tensor(count, dtype, rank))

    def _test_all_to_all_single(self, count, dtype, async_op):
        inputs = torch.cat(
            [
                self._expected_tensor(count, dtype, self.rank)
                for _ in range(self.world_size)
            ]
        )
        output = torch.empty_like(inputs)
        work = dist.all_to_all_single(output, inputs, async_op=async_op)
        self._wait(work, async_op)
        expected = torch.cat(
            [
                self._expected_tensor(count, dtype, rank)
                for rank in range(self.world_size)
            ]
        )
        self.assertEqual(output, expected)

    def _test_transport_matrix(self, test):
        for count in COUNTS:
            for dtype in self.dtypes:
                for async_op in ASYNC_OPS:
                    with self.subTest(count=count, dtype=dtype, async_op=async_op):
                        test(count, dtype, async_op)


class AbstractCollectivesTest(CollectivesTestMixin, C10dBackendTestContinuous):
    def test_broadcast(self):
        self._test_transport_matrix(self._test_broadcast)

    def test_all_gather(self):
        self._test_transport_matrix(self._test_all_gather)

    def test_all_gather_uneven(self):
        if not self.supports_uneven_all_gather:
            self.skipTest(f"{self.backend_name} does not support uneven all-gather")
        sizes = [rank + 1 for rank in range(self.world_size)]
        for async_op in ASYNC_OPS:
            with self.subTest(async_op=async_op):
                input = self._tensor(sizes[self.rank], torch.float32)
                outputs = [torch.empty(size, device=self.device) for size in sizes]
                work = dist.all_gather(outputs, input, async_op=async_op)
                self._wait(work, async_op)
                for rank, output in enumerate(outputs):
                    self.assertEqual(
                        output,
                        torch.full_like(output, self._value(rank, output.dtype)),
                    )

    def test_all_gather_multidim(self):
        # Distinct per-element values catch misplaced or transposed rows.
        for async_op in ASYNC_OPS:
            with self.subTest(async_op=async_op):
                base = torch.arange(6, dtype=torch.float32, device=self.device)
                input = (base + 100 * self.rank).view(2, 3)
                outputs = [torch.empty_like(input) for _ in range(self.world_size)]
                work = dist.all_gather(outputs, input, async_op=async_op)
                self._wait(work, async_op)
                for rank, output in enumerate(outputs):
                    self.assertEqual(output, (base + 100 * rank).view(2, 3))

    def test_all_gather_channels_last(self):
        # Ranks with different memory formats must issue the same collective
        # and gather logical values.
        shape = (2, 3, 2, 2)
        for mixed in (False, True):
            memory_format = self._memory_formats(mixed)
            for async_op in ASYNC_OPS:
                with self.subTest(mixed=mixed, async_op=async_op):
                    input = self._distinct_tensor(self.rank, 0, shape, memory_format)
                    outputs = [
                        self._sentinel(shape, memory_format)
                        for _ in range(self.world_size)
                    ]
                    work = dist.all_gather(outputs, input, async_op=async_op)
                    self._wait(work, async_op)
                    for rank, output in enumerate(outputs):
                        self.assertEqual(output, self._distinct_tensor(rank, 0, shape))

    def test_all_gather_single(self):
        self._test_transport_matrix(self._test_all_gather_single)

    def test_gather(self):
        self._test_transport_matrix(self._test_gather)

    def test_gather_into_tensor(self):
        if not self.supports_gather_single:
            self.skipTest(f"{self.backend_name} does not support gather_single")
        tensor = self._tensor(4, torch.float32)
        output = (
            torch.empty(self.world_size * 4, device=self.device)
            if self.rank == 0
            else None
        )
        dist.gather_into_tensor(tensor, output, dst=0)
        if self.rank == 0:
            expected = torch.cat(
                [
                    self._expected_tensor(4, torch.float32, rank)
                    for rank in range(self.world_size)
                ]
            )
            self.assertEqual(output, expected)

    def test_scatter(self):
        self._test_transport_matrix(self._test_scatter)

    def test_all_to_all(self):
        self._test_transport_matrix(self._test_all_to_all)

    def test_all_to_all_single(self):
        self._test_transport_matrix(self._test_all_to_all_single)

    def test_all_to_all_single_split_sizes(self):
        for async_op in ASYNC_OPS:
            input_splits = [
                self.rank + destination + 1 for destination in range(self.world_size)
            ]
            output_splits = [
                source + self.rank + 1 for source in range(self.world_size)
            ]
            inputs = torch.cat(
                [
                    torch.full(
                        (count,),
                        self.rank,
                        dtype=torch.float32,
                        device=self.device,
                    )
                    for count in input_splits
                ]
            )
            output = torch.empty(sum(output_splits), device=self.device)
            work = dist.all_to_all_single(
                output,
                inputs,
                output_split_sizes=output_splits,
                input_split_sizes=input_splits,
                async_op=async_op,
            )
            self._wait(work, async_op)
            expected = torch.cat(
                [
                    torch.full(
                        (count,), source, dtype=torch.float32, device=self.device
                    )
                    for source, count in enumerate(output_splits)
                ]
            )
            self.assertEqual(output, expected)

    def test_all_to_all_single_independent_empty_split_sizes(self):
        input_count = self.rank + 1
        input = torch.full(
            (self.world_size * input_count,),
            self.rank,
            dtype=torch.float32,
            device=self.device,
        )
        output_splits = [rank + 1 for rank in range(self.world_size)]
        output = torch.empty(sum(output_splits), device=self.device)
        dist.all_to_all_single(
            output,
            input,
            output_split_sizes=output_splits,
            input_split_sizes=[],
        )
        expected = torch.cat(
            [
                torch.full((count,), rank, dtype=torch.float32, device=self.device)
                for rank, count in enumerate(output_splits)
            ]
        )
        self.assertEqual(output, expected)

        input_splits = [rank + 1 for rank in range(self.world_size)]
        input = torch.cat(
            [
                torch.full((count,), self.rank, dtype=torch.float32, device=self.device)
                for count in input_splits
            ]
        )
        output_count = self.rank + 1
        output = torch.empty(
            self.world_size * output_count, dtype=torch.float32, device=self.device
        )
        dist.all_to_all_single(
            output,
            input,
            output_split_sizes=[],
            input_split_sizes=input_splits,
        )
        expected = torch.cat(
            [
                torch.full(
                    (output_count,), rank, dtype=torch.float32, device=self.device
                )
                for rank in range(self.world_size)
            ]
        )
        self.assertEqual(output, expected)

    def test_all_reduce(self):
        for count in COUNTS:
            for dtype in self.dtypes:
                for op in self._reduce_ops(dtype):
                    for async_op in ASYNC_OPS:
                        with self.subTest(
                            count=count, dtype=dtype, op=op, async_op=async_op
                        ):
                            tensor = self._tensor(count, dtype)
                            work = dist.all_reduce(tensor, op=op, async_op=async_op)
                            self._wait(work, async_op)
                            self.assertEqual(
                                tensor, self._expected_reduce(count, dtype, op)
                            )

    def test_reduce(self):
        for count in COUNTS:
            for dtype in self.dtypes:
                for op in self._reduce_ops(dtype):
                    for async_op in ASYNC_OPS:
                        with self.subTest(
                            count=count, dtype=dtype, op=op, async_op=async_op
                        ):
                            tensor = self._tensor(count, dtype)
                            work = dist.reduce(tensor, dst=0, op=op, async_op=async_op)
                            self._wait(work, async_op)
                            if self.rank == 0:
                                self.assertEqual(
                                    tensor, self._expected_reduce(count, dtype, op)
                                )

    def test_reduce_scatter(self):
        for count in COUNTS:
            for dtype in self.dtypes:
                for op in self._reduce_ops(dtype):
                    for async_op in ASYNC_OPS:
                        with self.subTest(
                            count=count, dtype=dtype, op=op, async_op=async_op
                        ):
                            inputs = [
                                self._tensor(count, dtype)
                                for _ in range(self.world_size)
                            ]
                            output = torch.empty_like(inputs[0])
                            work = dist.reduce_scatter(
                                output, inputs, op=op, async_op=async_op
                            )
                            self._wait(work, async_op)
                            self.assertEqual(
                                output, self._expected_reduce(count, dtype, op)
                            )

    def test_reduce_scatter_uneven(self):
        sizes = [rank + 1 for rank in range(self.world_size)]
        for async_op in ASYNC_OPS:
            with self.subTest(async_op=async_op):
                inputs = [self._tensor(size, torch.float32) for size in sizes]
                output = torch.empty(
                    sizes[self.rank], dtype=torch.float32, device=self.device
                )
                work = dist.reduce_scatter(
                    output,
                    inputs,
                    op=dist.ReduceOp.SUM,
                    async_op=async_op,
                )
                self._wait(work, async_op)
                self.assertEqual(
                    output,
                    self._expected_reduce(
                        sizes[self.rank], torch.float32, dist.ReduceOp.SUM
                    ),
                )

    def _distinct_reduce_ops(self):
        ops = list(self._reduce_ops(torch.float32))
        if torch.float32 in self.premul_sum_dtypes:
            ops.append(dist.ReduceOp.PREMUL_SUM(0.5))
        return ops

    def test_reduce_scatter_distinct_inputs(self):
        # Distinct values per element and destination catch misplaced rows.
        shape = (2, 3)
        for op in self._distinct_reduce_ops():
            for async_op in ASYNC_OPS:
                with self.subTest(op=op, async_op=async_op):
                    inputs = [
                        self._distinct_tensor(self.rank, dst, shape)
                        for dst in range(self.world_size)
                    ]
                    output = self._sentinel(shape)
                    work = dist.reduce_scatter(output, inputs, op=op, async_op=async_op)
                    self._wait(work, async_op)
                    self.assertEqual(output, self._expected_distinct_reduce(shape, op))

    def test_reduce_scatter_channels_last(self):
        self._test_reduce_scatter_memory_format(mixed=False)

    def test_reduce_scatter_mixed_memory_formats(self):
        # Ranks with different memory formats must issue the same collective
        # and reduce logical values.
        self._test_reduce_scatter_memory_format(mixed=True)

    def _test_reduce_scatter_memory_format(self, mixed):
        shape = (2, 3, 2, 2)
        memory_format = self._memory_formats(mixed)
        for op in self._distinct_reduce_ops():
            for async_op in ASYNC_OPS:
                with self.subTest(op=op, async_op=async_op):
                    inputs = [
                        self._distinct_tensor(self.rank, dst, shape, memory_format)
                        for dst in range(self.world_size)
                    ]
                    output = self._sentinel(shape, memory_format)
                    work = dist.reduce_scatter(output, inputs, op=op, async_op=async_op)
                    self._wait(work, async_op)
                    self._assert_equal_on_all_ranks(
                        output, self._expected_distinct_reduce(shape, op)
                    )

    def _assert_equal_on_all_ranks(self, actual, expected):
        # Agree on the result before asserting, so a failure on one rank (an
        # expected failure) does not leave the others in the next collective.
        failure = None
        try:
            self.assertEqual(actual, expected)
        except AssertionError as e:
            failure = e
        ok = torch.tensor([failure is None], dtype=torch.int32, device=self.device)
        dist.all_reduce(ok, op=dist.ReduceOp.MIN)
        if failure is not None:
            raise failure
        self.assertEqual(ok.item(), 1, "result mismatch on another rank")

    def test_reduce_scatter_single(self):
        for count in COUNTS:
            for dtype in self.dtypes:
                for op in self._reduce_ops(dtype):
                    for async_op in ASYNC_OPS:
                        with self.subTest(
                            count=count, dtype=dtype, op=op, async_op=async_op
                        ):
                            inputs = torch.cat(
                                [
                                    self._tensor(count, dtype)
                                    for _ in range(self.world_size)
                                ]
                            )
                            output = torch.empty(count, dtype=dtype, device=self.device)
                            work = dist.reduce_scatter_single(
                                output, inputs, op=op, async_op=async_op
                            )
                            self._wait(work, async_op)
                            self.assertEqual(
                                output, self._expected_reduce(count, dtype, op)
                            )

    def test_premul_sum(self):
        if not self.premul_sum_dtypes:
            self.skipTest(f"{self.backend_name} does not support PREMUL_SUM")
        expected = sum(range(1, self.world_size + 1)) * 0.5
        for dtype in self.premul_sum_dtypes:
            for factor in (
                0.5,
                torch.tensor(0.5, dtype=dtype, device=self.device),
            ):
                with self.subTest(dtype=dtype, factor_type=type(factor).__name__):
                    tensor = torch.full(
                        (4,), float(self.rank + 1), dtype=dtype, device=self.device
                    )
                    dist.all_reduce(tensor, op=dist.ReduceOp.PREMUL_SUM(factor))
                    self.assertEqual(tensor, torch.full_like(tensor, expected))

    def test_barrier(self):
        for async_op in ASYNC_OPS:
            work = dist.barrier(async_op=async_op)
            self._wait(work, async_op)

    def test_all_reduce_coalesced(self):
        for dtype in self.dtypes:
            for async_op in ASYNC_OPS:
                with self.subTest(dtype=dtype, async_op=async_op):
                    tensors = [self._tensor(i + 1, dtype) for i in range(3)]
                    work = dist.all_reduce_coalesced(tensors, async_op=async_op)
                    self._wait(work, async_op)
                    for tensor in tensors:
                        self.assertEqual(
                            tensor,
                            self._expected_reduce(
                                tensor.numel(), dtype, dist.ReduceOp.SUM
                            ),
                        )

    def test_coalescing_manager(self):
        if not self.supports_coalescing:
            self.skipTest(f"{self.backend_name} does not support coalescing")

        for async_ops in ASYNC_OPS:
            tensors = [
                torch.full((i + 1,), float(self.rank + i), device=self.device)
                for i in range(3)
            ]
            with dist._coalescing_manager(
                device=self.device, async_ops=async_ops
            ) as cm:
                for tensor in tensors:
                    dist.all_reduce(tensor)
            self.assertEqual(len(cm.works), 1 if async_ops else 0)
            cm.wait()
            rank_sum = self.world_size * (self.world_size - 1) // 2
            for i, tensor in enumerate(tensors):
                expected_value = rank_sum + i * self.world_size
                self.assertEqual(tensor, torch.full_like(tensor, expected_value))

        inputs = [
            torch.full((i + 1,), float(self.rank + i), device=self.device)
            for i in range(3)
        ]
        for async_ops in ASYNC_OPS:
            gathered = [
                torch.empty(input.numel() * self.world_size, device=self.device)
                for input in inputs
            ]
            with dist._coalescing_manager(
                device=self.device, async_ops=async_ops
            ) as cm:
                for output, input in zip(gathered, inputs):
                    dist.all_gather_single(output, input)
            self.assertEqual(len(cm.works), 1 if async_ops else 0)
            cm.wait()
            for i, output in enumerate(gathered):
                expected = torch.arange(
                    i, self.world_size + i, dtype=output.dtype, device=self.device
                ).repeat_interleave(i + 1)
                self.assertEqual(output, expected)

        inputs = [
            torch.full(
                (self.world_size * (i + 1),),
                float(self.rank + i),
                device=self.device,
            )
            for i in range(3)
        ]
        for async_ops in ASYNC_OPS:
            outputs = [torch.empty(i + 1, device=self.device) for i in range(3)]
            with dist._coalescing_manager(
                device=self.device, async_ops=async_ops
            ) as cm:
                for output, input in zip(outputs, inputs):
                    dist.reduce_scatter_single(output, input)
            self.assertEqual(len(cm.works), 1 if async_ops else 0)
            cm.wait()
            rank_sum = self.world_size * (self.world_size - 1) // 2
            for i, output in enumerate(outputs):
                expected_value = rank_sum + i * self.world_size
                self.assertEqual(output, torch.full_like(output, expected_value))

    def _coalescing_window(self, async_op, p2p, side, signal_peer=None):
        pg = dist.distributed_c10d._get_default_group()
        send_peer = (self.rank + 1) % self.world_size
        recv_peer = (self.rank - 1) % self.world_size
        n = 4 * self.world_size
        a2a_in = torch.full((n,), float(self.rank), device=self.device)
        a2a_out = torch.empty_like(a2a_in)
        reduced = torch.full((4,), float(self.rank + 1), device=self.device)
        sent = torch.full((4,), float(self.rank), device=self.device)
        received = torch.empty_like(sent)
        current = torch.cuda.current_stream(self.device)
        stream = None
        if side:
            stream = torch.cuda.Stream(self.device)
            stream.wait_stream(current)
            current.synchronize()
        pg._start_coalescing(self.device)
        # A window may span streams; sync ops run on the stream they are issued on.
        with torch.cuda.stream(stream):
            dist.all_to_all_single(a2a_out, a2a_in, async_op=async_op)
            dist.all_reduce(reduced, async_op=async_op)
        if p2p:
            dist.isend(sent, send_peer)
            dist.irecv(received, recv_peer)
        work = pg._end_coalescing(self.device)
        # The legacy backend returns no work for an all-sync window.
        completed = work is not None and work.is_completed()
        if signal_peer is not None:
            signal_peer()
            # The peers have not issued anything yet, so the window cannot be done.
            self.assertFalse(completed)
            # Only wait() orders the current stream after side-stream ops.
            if stream is not None:
                self.assertTrue(current.query())
        if work is not None:
            work.wait()
        elif stream is not None:
            current.wait_stream(stream)

        ranks = torch.arange(self.world_size, device=self.device)
        self.assertEqual(a2a_out, ranks.repeat_interleave(4).float())
        expected = sum(range(1, self.world_size + 1))
        self.assertEqual(reduced, torch.full_like(reduced, float(expected)))
        if p2p:
            self.assertEqual(received, torch.full_like(received, float(recv_peer)))

    def test_coalescing_window_covers_all_ops(self):
        if not self.supports_coalescing:
            self.skipTest(f"{self.backend_name} does not support coalescing")
        store = dist.distributed_c10d._get_default_store()
        for async_op, p2p, side in itertools.product((False, True), repeat=3):
            # Warm up so NCCL connections exist and rank 0 can issue alone.
            self._coalescing_window(async_op, p2p, side)
            key = f"coalescing_window_rank0_issued_{async_op}_{p2p}_{side}"
            if self.rank != 0:
                store.wait([key])
                self._coalescing_window(async_op, p2p, side)
            else:
                self._coalescing_window(
                    async_op, p2p, side, lambda key=key: store.set(key, "1")
                )

    def test_float8_transport(self):
        if not self.float8_dtypes:
            self.skipTest(f"{self.backend_name} does not support Float8")
        if torch.cuda.get_device_capability(self.device) < (9, 0):
            self.skipTest("Float8 collectives require sm90 or newer")
        tests = (
            self._test_broadcast,
            self._test_all_gather,
            self._test_all_gather_single,
            self._test_gather,
            self._test_scatter,
            self._test_all_to_all,
            self._test_all_to_all_single,
        )
        for test in tests:
            for dtype in self.float8_dtypes:
                for async_op in ASYNC_OPS:
                    with self.subTest(
                        collective=test.__name__,
                        dtype=dtype,
                        async_op=async_op,
                    ):
                        test(4, dtype, async_op)

    def test_complex_collectives(self):
        for dtype in self.complex_dtypes:
            for async_op in ASYNC_OPS:
                with self.subTest(
                    collective="broadcast", dtype=dtype, async_op=async_op
                ):
                    tensor = self._tensor(4, dtype)
                    work = dist.broadcast(tensor, src=0, async_op=async_op)
                    self._wait(work, async_op)
                    self.assertEqual(tensor, self._expected_tensor(4, dtype, rank=0))
            for op in (dist.ReduceOp.SUM, dist.ReduceOp.AVG):
                for async_op in ASYNC_OPS:
                    for collective in (dist.all_reduce, dist.reduce):
                        with self.subTest(
                            collective=collective.__name__,
                            dtype=dtype,
                            op=op,
                            async_op=async_op,
                        ):
                            tensor = self._tensor(4, dtype)
                            kwargs = {"dst": 0} if collective is dist.reduce else {}
                            work = collective(
                                tensor, op=op, async_op=async_op, **kwargs
                            )
                            self._wait(work, async_op)
                            if collective is dist.all_reduce or self.rank == 0:
                                self.assertEqual(
                                    tensor, self._expected_reduce(4, dtype, op)
                                )

    def test_all_to_all_single_invalid_split_sizes(self):
        valid_splits = [2] * self.world_size
        invalid_splits = (
            [-1, 1],
            [-1, 5],
            [4],
            [1] * self.world_size,
            [3] * self.world_size,
        )
        for invalid_side in ("input", "output"):
            for splits in invalid_splits:
                with self.subTest(invalid_side=invalid_side, splits=splits):
                    input_splits = splits if invalid_side == "input" else valid_splits
                    output_splits = splits if invalid_side == "output" else valid_splits
                    with self.assertRaisesRegex(RuntimeError, "[Ss]plit"):
                        dist.all_to_all_single(
                            torch.empty(4, device=self.device),
                            torch.ones(4, device=self.device),
                            output_split_sizes=output_splits,
                            input_split_sizes=input_splits,
                        )

        for invalid_side in ("input", "output"):
            with self.subTest(invalid_side=invalid_side, splits=[]):
                input_size = 3 if invalid_side == "input" else 4
                output_size = 3 if invalid_side == "output" else 4
                input_splits = [] if invalid_side == "input" else valid_splits
                output_splits = [] if invalid_side == "output" else valid_splits
                with self.assertRaisesRegex(
                    RuntimeError, "does not divide equally across group size"
                ):
                    dist.all_to_all_single(
                        torch.empty(output_size, device=self.device),
                        torch.ones(input_size, device=self.device),
                        output_split_sizes=output_splits,
                        input_split_sizes=input_splits,
                    )

        with self.assertRaisesRegex(
            RuntimeError, "does not divide equally across group size"
        ):
            dist.all_to_all_single(
                torch.empty(3, 2, device=self.device),
                torch.ones(3, 2, device=self.device),
                output_split_sizes=[],
                input_split_sizes=[],
            )

    def test_sync_barrier_blocks_host_on_stream(self):
        # A synchronous barrier must host-block the CPU thread until prior work
        # on the current stream has completed, not merely stream-order after it.
        # Downstream code relies on this (e.g. the flashinfer trtllm one-shot
        # Lamport all_reduce clears its IPC buffers on the stream, then issues a
        # barrier before the first all_reduce; a stream-order-only barrier lets
        # the all_reduce race the clear and both ranks spin forever). Enqueue a
        # long-running kernel, confirm the stream is still busy, then run a
        # synchronous barrier: once it returns the stream must have drained
        # (stream.query()). Only the synchronous path (async_op=False) host-
        # blocks; an async barrier stays stream-ordered and is not tested here.
        if self.device_type != "cuda":
            self.skipTest(f"{self.backend_name} host-block test requires CUDA")
        stream = torch.cuda.current_stream()
        torch.cuda._sleep(1_000_000_000)
        self.assertFalse(
            stream.query(), "precondition: enqueued work should leave stream busy"
        )
        self.assertIsNone(dist.barrier(async_op=False))
        self.assertTrue(
            stream.query(),
            "synchronous barrier must host-block until prior stream work completes",
        )

    def test_split_group(self):
        default_pg = dist.distributed_c10d._get_default_group()
        parent_backend = default_pg._get_backend(self.device)
        if not parent_backend.supports_splitting:
            self.skipTest(f"{self.backend_name} does not support split_group")
        if self.device_type != "cpu":
            default_pg.bound_device_id = self.device
        # Initialize the parent communicator before splitting.
        dist.all_reduce(torch.ones(4, device=self.device))

        # First half are members; the rest are non-members and must get the
        # NON_GROUP_MEMBER sentinel (they still issue the no-color split so the
        # collective stays in lockstep across the parent group).
        members = list(range(self.world_size // 2))
        subgroup = dist.split_group(split_ranks=[members])

        if self.rank in members:
            self.assertIsNot(subgroup, dist.GroupMember.NON_GROUP_MEMBER)
            self.assertEqual(dist.get_process_group_ranks(subgroup), members)
            tensor = torch.full((4,), float(self.rank + 1), device=self.device)
            dist.all_reduce(tensor, group=subgroup)
            expected = float(sum(r + 1 for r in members))
            self.assertEqual(tensor, torch.full_like(tensor, expected))
        else:
            self.assertIs(subgroup, dist.GroupMember.NON_GROUP_MEMBER)

        dist.barrier()

    def test_split_group_full_partition(self):
        if self.world_size % 2 != 0:
            self.skipTest("full-partition split requires an even world size")
        default_pg = dist.distributed_c10d._get_default_group()
        parent_backend = default_pg._get_backend(self.device)
        if not parent_backend.supports_splitting:
            self.skipTest(f"{self.backend_name} does not support split_group")
        if self.device_type != "cpu":
            default_pg.bound_device_id = self.device
        dist.all_reduce(torch.ones(4, device=self.device))

        half = self.world_size // 2
        first = list(range(half))
        second = list(range(half, self.world_size))
        subgroup = dist.split_group(split_ranks=[first, second])
        my_group = first if self.rank in first else second

        self.assertIsNot(subgroup, dist.GroupMember.NON_GROUP_MEMBER)
        self.assertEqual(dist.get_process_group_ranks(subgroup), my_group)
        tensor = torch.full((4,), float(self.rank + 1), device=self.device)
        dist.all_reduce(tensor, group=subgroup)
        expected = float(sum(r + 1 for r in my_group))
        self.assertEqual(tensor, torch.full_like(tensor, expected))
        dist.barrier()

    def test_noncontiguous_all_to_all_error(self):
        input = torch.ones(self.world_size, self.world_size, device=self.device).t()
        output = torch.empty_like(input.contiguous())
        with self.assertRaisesRegex(ValueError, "Tensors must be contiguous"):
            dist.all_to_all_single(output, input)

    def test_mismatched_dtypes(self):
        input = torch.ones(4, device=self.device)
        output = torch.empty(
            self.world_size * input.numel(),
            dtype=torch.float64,
            device=self.device,
        )
        with self.assertRaises((RuntimeError, TypeError, ValueError)):
            dist.all_gather_single(output, input)
        with self.assertRaises((RuntimeError, TypeError, ValueError)):
            dist.reduce_scatter_single(input, output)
        with self.assertRaises((RuntimeError, TypeError, ValueError)):
            dist.all_to_all_single(output, input)

    def _profiled_events(self, fn):
        with torch.autograd.profiler.profile(record_shapes=True) as prof:
            fn()
            torch.cuda.synchronize(self.device)
        return [e for e in prof.function_events if e.is_async]

    def test_profiler_op_events(self):
        # Each op emits one async "<prefix><op>" event recording its inputs,
        # matching stock ProcessGroupNCCL.
        if self.profiler_prefix is None:
            self.skipTest(f"{self.backend_name} has no profiler event names")
        ws, rank = self.world_size, self.rank
        legacy = self.profiler_legacy_events
        t = torch.ones(4, device=self.device)
        t2 = torch.ones(4, device=self.device)
        gathered = [torch.empty_like(t) for _ in range(ws)]
        flat = torch.empty(ws * 4, device=self.device)
        a2a_in = torch.ones(ws, 2, device=self.device)
        a2a_out = torch.empty_like(a2a_in)
        # Uneven splits: every rank sends dst + 1 rows to rank dst.
        in_splits = [dst + 1 for dst in range(ws)]
        out_splits = [rank + 1] * ws
        a2a_uneven_in = torch.ones(sum(in_splits), 2, device=self.device)
        a2a_uneven_out = torch.empty(sum(out_splits), 2, device=self.device)
        shape = list(t.shape)

        def coalesced_all_reduce():
            with dist._coalescing_manager(device=self.device):
                dist.all_reduce(t)
                dist.all_reduce(t2)

        def batch_p2p():
            ops = [
                dist.P2POp(dist.isend, t, (rank + 1) % ws),
                dist.P2POp(dist.irecv, t2, (rank - 1) % ws),
            ]
            for work in dist.batch_isend_irecv(ops):
                work.wait()

        def send_recv():
            if rank == 0:
                dist.send(t, 1)
            elif rank == 1:
                dist.recv(t, 0)

        # Stock names send/recv with the peer.
        p2p_name = {0: "send", 1: "recv"}.get(rank)
        if legacy and p2p_name:
            p2p_name += {0: " 0->1", 1: " 1<-0"}[rank]
        cases = [
            ("all_reduce", lambda: dist.all_reduce(t), [shape]),
            ("all_reduce", lambda: dist.all_reduce(t, async_op=True).wait(), [shape]),
            ("broadcast", lambda: dist.broadcast(t, src=0), [shape]),
            ("reduce", lambda: dist.reduce(t, dst=0), [shape]),
            ("all_reduce_barrier", dist.barrier, [[1]]),
            ("all_gather", lambda: dist.all_gather(gathered, t), [shape]),
            ("_all_gather_base", lambda: dist.all_gather_into_tensor(flat, t), [shape]),
            (
                "_reduce_scatter_base",
                lambda: dist.reduce_scatter_tensor(t, flat),
                [list(flat.shape)],
            ),
            (
                "all_to_all",
                lambda: dist.all_to_all_single(a2a_out, a2a_in),
                [list(a2a_in.shape)],
            ),
            (
                "all_to_all",
                lambda: dist.all_to_all_single(
                    a2a_uneven_out, a2a_uneven_in, out_splits, in_splits
                ),
                [list(a2a_uneven_in.shape)],
            ),
            (
                "scatter",
                lambda: dist.scatter(t, gathered if rank == 0 else None, src=0),
                [shape],
            ),
            (
                "gather",
                lambda: dist.gather(t, gathered if rank == 0 else None, dst=0),
                [shape],
            ),
            # One event for the whole batch. Stock adds an input-less one from
            # endCoalescing.
            ("allreduce_coalesced", coalesced_all_reduce, [shape, shape]),
            # batch_isend_irecv: one event, send tensors only. Stock records none.
            ("coalesced", batch_p2p, [] if legacy else [shape]),
            (p2p_name, send_recv, [shape]),
        ]
        dist.all_reduce(t)
        torch.cuda.synchronize(self.device)
        for i, (name, fn, inputs) in enumerate(cases):
            with self.subTest(case=i, op=name):
                expected = [] if name is None else [(name, inputs)]
                if legacy and name == "allreduce_coalesced":
                    expected.append(("coalesced", []))
                events = self._profiled_events(fn)
                self.assertEqual(
                    [
                        (e.name, e.input_shapes)
                        for e in events
                        if e.name.startswith(self.profiler_prefix)
                    ],
                    [(self.profiler_prefix + n, s) for n, s in expected],
                )

        # dist.record_comm renames the single event.
        with self.subTest(op="record_comm"):

            def record_comm():
                with dist.record_comm("custom_comm"):
                    dist.all_reduce(t)

            events = self._profiled_events(record_comm)
            self.assertEqual(
                [e.input_shapes for e in events if e.name == "custom_comm"],
                # Stock's WorkNCCL copy also opens an input-less renamed event.
                [[shape], []] if legacy else [[shape]],
            )
            self.assertEqual(
                [e.name for e in events if e.name.startswith(self.profiler_prefix)],
                [],
            )


class AbstractIsolatedCollectivesTest(CollectivesTestMixin, C10dBackendTest):
    """Tests that use another rank's device or call a collective on only some
    ranks, so they keep fresh rank processes."""

    def test_collective_preserves_current_device(self):
        if self.device_type != "cuda":
            self.skipTest(f"{self.backend_name} does not use CUDA")
        self._init_pg()
        tensor = torch.ones(4, device=self.device)
        other_device = torch.device("cuda", (self.rank + 1) % self.world_size)
        torch.cuda.set_device(other_device)

        dist.all_reduce(tensor)

        self.assertEqual(torch.cuda.current_device(), other_device.index)
        self.assertEqual(tensor, torch.full_like(tensor, self.world_size))

    def test_all_gather_mixed_devices(self):
        self._test_all_gather_mixed_devices([4] * self.world_size)

    def test_all_gather_uneven_mixed_devices(self):
        # Uneven sizes take the per-rank path, which stages outputs on other
        # devices.
        if self.device_type != "cuda":
            self.skipTest(f"{self.backend_name} does not use CUDA")
        if not self.supports_uneven_all_gather_mixed_devices:
            # nccl-legacy hangs creating a communicator for the other device.
            self.skipTest(
                f"{self.backend_name} does not support uneven all-gather to "
                "other devices"
            )
        self._test_all_gather_mixed_devices(
            [rank + 1 for rank in range(self.world_size)]
        )

    def _test_all_gather_mixed_devices(self, sizes):
        if self.device_type != "cuda":
            self.skipTest(f"{self.backend_name} does not use CUDA")
        self._init_pg()
        tensor = self._tensor(sizes[self.rank], torch.float32)
        other_device = torch.device("cuda", (self.rank + 1) % self.world_size)
        outputs = [torch.empty(size, device=self.device) for size in sizes]
        outputs[-1] = torch.empty(sizes[-1], device=other_device)
        torch.cuda.set_device(other_device)

        work = dist.all_gather(outputs, tensor, async_op=True)

        self.assertEqual(torch.cuda.current_device(), other_device.index)
        work.wait()
        self.assertEqual(torch.cuda.current_device(), other_device.index)
        torch.cuda.synchronize(self.device)
        torch.cuda.synchronize(other_device)
        for rank, output in enumerate(outputs):
            self.assertEqual(
                output,
                torch.full_like(output, self._value(rank, output.dtype)),
            )

    def test_gather_single(self):
        self._init_pg()
        if not self.supports_gather_single:
            with self.assertRaisesRegex(RuntimeError, "does not support gather_single"):
                self._test_gather_single(4, torch.float32, False)
            return
        self._test_transport_matrix(self._test_gather_single)

        # Only the destination rank validates the output size, so only it calls
        # into the (failing) collective; the size check happens before any
        # communication is issued.
        if self.rank == 0:
            input = torch.ones(4, device=self.device)
            output = torch.empty(input.numel(), device=self.device)
            with self.assertRaises((RuntimeError, ValueError)):
                dist.gather_single(input, output, dst=0)

    def test_all_to_all_rejects_mixed_devices(self):
        if self.device_type != "cuda":
            self.skipTest(f"{self.backend_name} does not use CUDA")
        self._init_pg()
        other_device = torch.device("cuda", (self.rank + 1) % self.world_size)
        inputs = [
            torch.full((1,), float(self.rank), device=self.device)
            for _ in range(self.world_size)
        ]
        inputs[-1] = torch.full((1,), float(self.rank), device=other_device)
        outputs = [torch.empty(1, device=self.device) for _ in range(self.world_size)]

        with self.assertRaisesRegex(
            (RuntimeError, ValueError), "same device|Expected tensor on"
        ):
            dist.all_to_all(outputs, inputs)


instantiate_backend_tests(
    globals(),
    "Collectives",
    AbstractCollectivesTest,
    C10D_BACKENDS,
    harness=MultiProcContinuousTest,
)
instantiate_backend_tests(
    globals(), "IsolatedCollectives", AbstractIsolatedCollectivesTest, C10D_BACKENDS
)


if __name__ == "__main__":
    run_tests()
