# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""CPU regression tests for c10d Gloo native reduce-scatter (#205855)."""

import os
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _expected(op, world):
    values = range(1, world + 1)
    if op == dist.ReduceOp.SUM:
        return sum(values)
    if op == dist.ReduceOp.AVG:
        return sum(values) / world
    if op == dist.ReduceOp.PRODUCT:
        product = 1
        for value in values:
            product *= value
        return product
    if op == dist.ReduceOp.MIN:
        return 1
    if op == dist.ReduceOp.MAX:
        return world
    raise AssertionError(op)


def _worker(rank, world, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method="file://" + rendezvous, rank=rank, world_size=world
    )
    try:
        for dtype in (torch.float32, torch.float64):
            for op in (
                dist.ReduceOp.SUM,
                dist.ReduceOp.AVG,
                dist.ReduceOp.PRODUCT,
                dist.ReduceOp.MIN,
                dist.ReduceOp.MAX,
            ):
                data = torch.full((world * 6, 5), rank + 1, dtype=dtype)
                original = data.clone()
                output = torch.empty((6, 5), dtype=dtype)
                work = dist.reduce_scatter_tensor(output, data, op=op, async_op=True)
                work.wait()
                torch.testing.assert_close(
                    output, torch.full_like(output, _expected(op, world))
                )
                torch.testing.assert_close(data, original)

                # The native path copies contiguous storage, retaining layout semantics.
                strided_input = torch.full(
                    (5, world * 6), rank + 1, dtype=dtype
                ).t()
                strided_output = torch.empty((5, 6), dtype=dtype).t()
                assert not strided_input.is_contiguous()
                assert not strided_output.is_contiguous()
                dist.reduce_scatter_tensor(
                    strided_output, strided_input, op=op, async_op=True
                ).wait()
                torch.testing.assert_close(
                    strided_output,
                    torch.full_like(strided_output, _expected(op, world)),
                )

                chunks = [
                    torch.full((6, 5), rank + 1, dtype=dtype)
                    for _ in range(world)
                ]
                list_output = torch.empty((6, 5), dtype=dtype)
                dist.reduce_scatter(
                    list_output, chunks, op=op, async_op=True
                ).wait()
                torch.testing.assert_close(
                    list_output,
                    torch.full_like(list_output, _expected(op, world)),
                )

        # Integer bitwise ops must continue through the existing fallback.
        bits = torch.full((world * 6,), rank + 1, dtype=torch.int32)
        bits_output = torch.empty(6, dtype=torch.int32)
        dist.reduce_scatter_tensor(bits_output, bits, op=dist.ReduceOp.BOR)
        expected_bits = 0
        for value in range(1, world + 1):
            expected_bits |= value
        torch.testing.assert_close(
            bits_output, torch.full_like(bits_output, expected_bits)
        )

        # Slot reservation and progress of two simultaneous asynchronous operations.
        first = torch.full((world * 4,), rank + 1, dtype=torch.float64)
        second = torch.full((world * 4,), 2 * (rank + 1), dtype=torch.float64)
        first_out = torch.empty(4, dtype=torch.float64)
        second_out = torch.empty(4, dtype=torch.float64)
        w1 = dist.reduce_scatter_tensor(first_out, first, async_op=True)
        w2 = dist.reduce_scatter_tensor(second_out, second, async_op=True)
        w2.wait()
        w1.wait()
        total = sum(range(1, world + 1))
        torch.testing.assert_close(
            first_out, torch.full_like(first_out, total)
        )
        torch.testing.assert_close(
            second_out, torch.full_like(second_out, 2 * total)
        )
        dist.barrier()
    finally:
        dist.destroy_process_group()


class TestGlooNativeReduceScatter(unittest.TestCase):
    def _run_world(self, world):
        with tempfile.TemporaryDirectory(prefix="gloo-rs-") as directory:
            mp.spawn(
                _worker,
                args=(world, os.path.join(directory, "rendezvous")),
                nprocs=world,
                join=True,
            )

    def test_two_ranks(self):
        self._run_world(2)

    def test_three_ranks(self):
        self._run_world(3)

    def test_four_ranks(self):
        self._run_world(4)


if __name__ == "__main__":
    unittest.main()
