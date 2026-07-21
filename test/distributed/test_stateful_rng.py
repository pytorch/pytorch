# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import unittest
from functools import partial
from typing import Any, cast

import torch
from torch._library.utils import fill_defaults
from torch.distributed._rng_layout import (
    _derive_checkpointable_rng_layout,
    _rng_target_for_layout,
    _scatter_rng_result_,
)
from torch.distributed.checkpoint import CheckpointableTensor
from torch.distributed.tensor.placement_types import _StridedShard
from torch.testing._internal.common_utils import run_tests, TEST_CUDA, TestCase
from torch.utils._python_dispatch import TorchDispatchMode


aten = torch.ops.aten


class _StatefulRNGMode(TorchDispatchMode):
    def __torch_dispatch__(
        self,
        func: torch._ops.OpOverload,
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        if kwargs is None:
            kwargs = {}
        if func is not aten.normal_.default:
            return func(*args, **kwargs)

        filled_args, filled_kwargs = fill_defaults(func._schema, args, kwargs)
        tensor_arg = filled_args[0]
        if not isinstance(tensor_arg, torch.Tensor):
            return func(*args, **kwargs)
        if (
            tensor_arg.is_meta
            or tensor_arg.device.type != "cuda"
            or not tensor_arg.is_contiguous()
        ):
            return func(*args, **kwargs)
        if not isinstance(tensor_arg, CheckpointableTensor):
            return func(*args, **kwargs)
        rng_metadata = cast(CheckpointableTensor, tensor_arg)
        rng_layout = _derive_checkpointable_rng_layout(rng_metadata)
        rng_target = _rng_target_for_layout(tensor_arg, rng_layout)

        _, mean, std = filled_args
        start_indices: list[int | torch.SymInt] = []
        block_sizes: list[int | torch.SymInt] = []
        block_strides: list[int | torch.SymInt] = []
        block_counts: list[int | torch.SymInt] = []
        for start_index, block_size, block_stride, num_blocks in rng_layout.index_blocks:
            start_indices.append(start_index)
            block_sizes.append(block_size)
            block_strides.append(block_stride)
            block_counts.append(num_blocks)
        aten._philox_normal_flat_slice_.default(
            rng_target,
            rng_layout.logical_numel,
            start_indices,
            block_sizes,
            block_strides,
            block_counts,
            mean,
            std,
            generator=filled_kwargs["generator"],
        )
        _scatter_rng_result_(tensor_arg, rng_target, rng_layout)
        return tensor_arg


class TestCheckpointableTensorRNG(TestCase):
    @staticmethod
    def _set_shard_metadata(
        tensor: torch.Tensor,
        global_shape: tuple[int, ...],
        global_offsets: tuple[tuple[int, ...], ...],
        local_offsets: tuple[tuple[int, ...], ...],
        local_sizes: tuple[tuple[int, ...], ...],
    ) -> None:
        setattr(tensor, "global_shape", global_shape)  # noqa: B010
        setattr(tensor, "global_offsets", global_offsets)  # noqa: B010
        setattr(tensor, "local_offsets", local_offsets)  # noqa: B010
        setattr(tensor, "local_sizes", local_sizes)  # noqa: B010

    def test_plain_tensor_metadata_satisfies_protocol(self):
        tensor = torch.empty(1)
        self.assertNotIsInstance(tensor, CheckpointableTensor)

        self._set_shard_metadata(tensor, (1,), ((0,),), ((0,),), ((1,),))
        self.assertIsInstance(tensor, CheckpointableTensor)

    def test_shard_metadata_lowers_to_flat_indices(self):
        cases = (
            ((3, 4), (0, 2), (3, 2), ((2, 2, 4, 3),)),
            ((4, 5, 6), (1, 1, 2), (2, 3, 3), ((38, 3, 6, 3), (68, 3, 6, 3))),
        )
        for global_shape, global_offset, local_shape, expected_blocks in cases:
            with self.subTest(global_shape=global_shape):
                tensor = torch.empty(local_shape)
                zero_offset = (0,) * len(local_shape)
                self._set_shard_metadata(
                    tensor,
                    global_shape,
                    (global_offset,),
                    (zero_offset,),
                    (local_shape,),
                )

                layout = _derive_checkpointable_rng_layout(
                    cast(CheckpointableTensor, tensor)
                )

                self.assertEqual(layout.logical_numel, torch.Size(global_shape).numel())
                self.assertEqual(layout.index_blocks, expected_blocks)
                self.assertTrue(layout.is_direct)

    def test_strided_shard_lowers_to_multiple_index_blocks(self):
        # _StridedShard(dim=1, split_factor=2) on two ranks gives rank 0
        # columns [0:2, 4:6], concatenated in that order in the local tensor.
        tensor = torch.empty((3, 4))
        self._set_shard_metadata(
            tensor,
            (3, 8),
            ((0, 0), (0, 4)),
            ((0, 0), (0, 2)),
            ((3, 2), (3, 2)),
        )

        layout = _derive_checkpointable_rng_layout(
            cast(CheckpointableTensor, tensor)
        )

        self.assertEqual(layout.index_blocks, ((0, 2, 8, 3), (4, 2, 8, 3)))
        self.assertFalse(layout.is_direct)

    @unittest.skipIf(not TEST_CUDA, "CUDA is required")
    def test_initializers_match_dense(self):
        device = torch.device("cuda")
        global_shape = (5, 7)
        normal = partial(torch.nn.init.normal_, mean=0.1, std=0.02)
        trunc_normal = partial(torch.nn.init.trunc_normal_, mean=0.0, std=0.02)
        cases = (
            (
                "normal_contiguous",
                normal,
                (slice(3, 5), slice(None)),
                ((3, 0),),
            ),
            (
                "normal_strided",
                normal,
                (slice(None), slice(4, 7)),
                ((0, 4),),
            ),
            (
                "trunc_normal_strided",
                trunc_normal,
                (slice(None), slice(4, 7)),
                ((0, 4),),
            ),
        )

        for case_name, init_fn, global_slice, global_offsets in cases:
            with self.subTest(case=case_name):
                torch.manual_seed(123)
                expected = torch.empty(global_shape, device=device)
                init_fn(expected)
                expected_state = torch.cuda.get_rng_state(device)

                torch.manual_seed(123)
                actual = torch.empty(expected[global_slice].shape, device=device)
                local_size = tuple(actual.shape)
                self._set_shard_metadata(
                    actual,
                    global_shape,
                    global_offsets,
                    ((0, 0),),
                    (local_size,),
                )
                with _StatefulRNGMode():
                    init_fn(actual)

                self.assertEqual(
                    actual, expected[global_slice].contiguous(), rtol=0, atol=0
                )
                self.assertEqual(torch.cuda.get_rng_state(device), expected_state)

    @unittest.skipIf(not TEST_CUDA, "CUDA is required")
    def test_strided_shard_matches_dense(self):
        device = torch.device("cuda")
        global_shape = (3, 8)
        placement = _StridedShard(dim=1, split_factor=2)

        torch.manual_seed(123)
        dense = torch.empty(global_shape, device=device).normal_(0.1, 0.02)
        expected_shards, _ = placement._split_tensor(dense, 2, with_padding=False)
        expected_state = torch.cuda.get_rng_state(device)

        global_offsets_by_rank = (
            ((0, 0), (0, 4)),
            ((0, 2), (0, 6)),
        )
        for rank, global_offsets in enumerate(global_offsets_by_rank):
            with self.subTest(rank=rank):
                torch.manual_seed(123)
                actual = torch.empty((3, 4), device=device)
                self._set_shard_metadata(
                    actual,
                    global_shape,
                    global_offsets,
                    ((0, 0), (0, 2)),
                    ((3, 2), (3, 2)),
                )
                with _StatefulRNGMode():
                    actual.normal_(0.1, 0.02)

                self.assertEqual(actual, expected_shards[rank], rtol=0, atol=0)
                self.assertEqual(torch.cuda.get_rng_state(device), expected_state)

    @unittest.skipIf(not TEST_CUDA, "CUDA is required")
    def test_multiblock_replay_from_nonzero_generator_offset(self):
        device = torch.device("cuda")
        properties = torch.cuda.get_device_properties(device)
        block_size = 256
        total_stride = (
            block_size
            * properties.multi_processor_count
            * (properties.max_threads_per_multi_processor // block_size)
        )
        global_numel = 5 * total_stride + 8
        global_indices = torch.tensor(
            [
                2,
                3,
                total_stride + 2,
                total_stride + 3,
                2 * total_stride + 2,
                2 * total_stride + 3,
                4 * total_stride + 5,
                4 * total_stride + 6,
                4 * total_stride + 7,
            ],
            device=device,
        )

        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                expected_generator = torch.Generator(device=device).manual_seed(123)
                torch.rand(11, device=device, generator=expected_generator)
                expected = []
                for _ in range(2):
                    dense = torch.empty(global_numel, dtype=dtype, device=device)
                    dense.normal_(0.1, 0.02, generator=expected_generator)
                    expected.append(dense[global_indices])
                expected_state = expected_generator.get_state()

                actual_generator = torch.Generator(device=device).manual_seed(123)
                torch.rand(11, device=device, generator=actual_generator)
                actual = torch.empty(global_indices.numel(), dtype=dtype, device=device)
                # Metadata order is deliberately unrelated to local storage order.
                self._set_shard_metadata(
                    actual,
                    (global_numel,),
                    (
                        (4 * total_stride + 5,),
                        (2,),
                        (total_stride + 2,),
                        (2 * total_stride + 2,),
                    ),
                    ((6,), (0,), (2,), (4,)),
                    ((3,), (2,), (2,), (2,)),
                )
                results = []
                with _StatefulRNGMode():
                    for _ in range(2):
                        actual.normal_(0.1, 0.02, generator=actual_generator)
                        results.append(actual.clone())

                self.assertEqual(results, expected, rtol=0, atol=0)
                self.assertEqual(actual_generator.get_state(), expected_state)

    @unittest.skipIf(not TEST_CUDA, "CUDA is required")
    def test_empty_local_tensor_matches_dense_increment(self):
        device = torch.device("cuda")
        for global_numel in (0, 7):
            with self.subTest(global_numel=global_numel):
                torch.manual_seed(123)
                torch.empty(global_numel, device=device).normal_()
                expected_state = torch.cuda.get_rng_state(device)

                torch.manual_seed(123)
                actual = torch.empty(0, device=device)
                self._set_shard_metadata(actual, (global_numel,), (), (), ())
                with _StatefulRNGMode():
                    actual.normal_()

                self.assertEqual(torch.cuda.get_rng_state(device), expected_state)

    @unittest.skipIf(not TEST_CUDA, "CUDA is required")
    def test_invalid_metadata_does_not_advance_generator(self):
        device = torch.device("cuda")
        generator = torch.Generator(device=device).manual_seed(123)
        actual = torch.empty(3, device=device)
        self._set_shard_metadata(
            actual,
            (7,),
            ((2,),),
            ((0,),),
            ((2,),),
        )
        state = generator.get_state().clone()

        with self.assertRaisesRegex(ValueError, "cover the entire tensor"):
            with _StatefulRNGMode():
                actual.normal_(generator=generator)

        self.assertEqual(generator.get_state(), state)


class TestPhiloxFlatSliceOps(TestCase):
    @unittest.skipIf(not TEST_CUDA, "CUDA is required")
    def test_invalid_calls_do_not_advance_generator(self):
        device = torch.device("cuda")
        generator = torch.Generator(device=device).manual_seed(123)

        def assert_invalid_without_advancing(regex, fn):
            state = generator.get_state().clone()
            with self.assertRaisesRegex(RuntimeError, regex):
                fn()
            self.assertEqual(generator.get_state(), state)

        assert_invalid_without_advancing(
            "block_stride 1 must be at least block_size 2",
            lambda: torch.ops.aten._philox_normal_flat_slice_(
                torch.empty(2, device=device),
                4,
                [0],
                [2],
                [1],
                [1],
                generator=generator,
            ),
        )
        assert_invalid_without_advancing(
            "normal expects std >= 0.0",
            lambda: torch.ops.aten._philox_normal_flat_slice_(
                torch.empty(1, device=device),
                1,
                [0],
                [1],
                [1],
                [1],
                0,
                -1,
                generator=generator,
            ),
        )


if __name__ == "__main__":
    run_tests()
