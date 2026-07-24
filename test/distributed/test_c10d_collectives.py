# Owner(s): ["oncall: distributed"]

import sys

import torch
import torch.distributed as dist


if not dist.is_available():
    print("distributed package not available, skipping tests", file=sys.stderr)
    sys.exit(0)

from c10d_backend_common import (
    C10D_BACKENDS,
    C10dBackendTest,
    instantiate_backend_tests,
)

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


class AbstractCollectivesTest(C10dBackendTest):
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

    def test_broadcast(self):
        self._init_pg()
        self._test_transport_matrix(self._test_broadcast)

    def test_all_gather(self):
        self._init_pg()
        self._test_transport_matrix(self._test_all_gather)

    def test_all_gather_single(self):
        self._init_pg()
        self._test_transport_matrix(self._test_all_gather_single)

    def test_gather(self):
        self._init_pg()
        self._test_transport_matrix(self._test_gather)

    def test_scatter(self):
        self._init_pg()
        self._test_transport_matrix(self._test_scatter)

    def test_all_to_all(self):
        self._init_pg()
        self._test_transport_matrix(self._test_all_to_all)

    def test_all_to_all_single(self):
        self._init_pg()
        self._test_transport_matrix(self._test_all_to_all_single)

    def test_all_to_all_single_split_sizes(self):
        self._init_pg()
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

    def test_all_reduce(self):
        self._init_pg()
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
        self._init_pg()
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
        self._init_pg()
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

    def test_reduce_scatter_single(self):
        self._init_pg()
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

    def test_barrier(self):
        self._init_pg()
        for async_op in ASYNC_OPS:
            work = dist.barrier(async_op=async_op)
            self._wait(work, async_op)

    def test_all_reduce_coalesced(self):
        self._init_pg()
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
        self._init_pg()

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

    def test_float8_transport(self):
        if not self.float8_dtypes:
            self.skipTest(f"{self.backend_name} does not support Float8")
        if torch.cuda.get_device_capability(self.device) < (9, 0):
            self.skipTest("Float8 collectives require sm90 or newer")
        self._init_pg()
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
        self._init_pg()
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

    def test_noncontiguous_all_to_all_error(self):
        self._init_pg()
        input = torch.ones(self.world_size, self.world_size, device=self.device).t()
        output = torch.empty_like(input.contiguous())
        with self.assertRaisesRegex(ValueError, "Tensors must be contiguous"):
            dist.all_to_all_single(output, input)

    def test_mismatched_dtypes(self):
        self._init_pg()
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


instantiate_backend_tests(
    globals(), "Collectives", AbstractCollectivesTest, C10D_BACKENDS
)


if __name__ == "__main__":
    run_tests()
