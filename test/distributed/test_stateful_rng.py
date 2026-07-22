# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import unittest
from typing import Any, cast

import torch
from torch._library.utils import fill_defaults
from torch.distributed.checkpoint import CheckpointableTensor
from torch.testing._internal.common_utils import run_tests, TEST_CUDA, TestCase
from torch.utils._python_dispatch import TorchDispatchMode


aten = torch.ops.aten

# Must match PhiloxDistributionKind in PhiloxDistribution.cu.
_PHILOX_DISTRIBUTION_NORMAL = 0
_PHILOX_DISTRIBUTION_UNIFORM = 1


def _flatten_chunks(chunks: tuple[tuple[int, ...], ...]) -> list[int]:
    return [value for chunk in chunks for value in chunk]


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
        if tensor_arg.is_meta or tensor_arg.device.type != "cuda":
            return func(*args, **kwargs)
        if not isinstance(tensor_arg, CheckpointableTensor):
            return func(*args, **kwargs)
        rng_metadata = cast(CheckpointableTensor, tensor_arg)

        _, mean, std = filled_args
        aten._philox_distribution_shards_.default(
            tensor_arg,
            rng_metadata.global_shape,
            _flatten_chunks(rng_metadata.global_offsets),
            _flatten_chunks(rng_metadata.local_offsets),
            _flatten_chunks(rng_metadata.local_sizes),
            len(rng_metadata.global_offsets),
            _PHILOX_DISTRIBUTION_NORMAL,
            (mean, std),
            generator=filled_kwargs["generator"],
        )
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

    @staticmethod
    def _layout_cases():
        return (
            (
                "shard",
                (5, 7),
                (2, 7),
                ((3, 0),),
                ((0, 0),),
                ((2, 7),),
            ),
            (
                "strided_shard",
                (3, 8),
                (3, 4),
                ((0, 0), (0, 4)),
                ((0, 0), (0, 2)),
                ((3, 2), (3, 2)),
            ),
            (
                "ragged_shard",
                (4, 4),
                (3, 4),
                ((1, 0),),
                ((0, 0),),
                ((3, 4),),
            ),
            (
                "grouped_ragged_shard",
                (4, 2),
                (2, 2),
                ((2, 0),),
                ((0, 0),),
                ((2, 2),),
            ),
            (
                "grouped_owned",
                (4, 2, 3),
                (2, 2, 3),
                ((2, 0, 0),),
                ((0, 0, 0),),
                ((2, 2, 3),),
            ),
        )

    def test_plain_tensor_metadata_satisfies_protocol(self):
        tensor = torch.empty(1)
        self.assertNotIsInstance(tensor, CheckpointableTensor)

        self._set_shard_metadata(tensor, (1,), ((0,),), ((0,),), ((1,),))
        self.assertIsInstance(tensor, CheckpointableTensor)

    @unittest.skipIf(not TEST_CUDA, "CUDA is required")
    def test_layouts_match_dense(self):
        device = torch.device("cuda")
        for (
            name,
            global_shape,
            local_shape,
            global_offsets,
            local_offsets,
            local_sizes,
        ) in self._layout_cases():
            with self.subTest(layout=name):
                expected_generator = torch.Generator(device=device).manual_seed(123)
                dense = torch.empty(global_shape, device=device).normal_(
                    0.1, 0.02, generator=expected_generator
                )
                expected_state = expected_generator.get_state()
                expected = torch.empty(local_shape, device=device)
                for global_offset, local_offset, local_size in zip(
                    global_offsets,
                    local_offsets,
                    local_sizes,
                    strict=True,
                ):
                    global_slices = tuple(
                        slice(offset, offset + size)
                        for offset, size in zip(global_offset, local_size, strict=True)
                    )
                    local_slices = tuple(
                        slice(offset, offset + size)
                        for offset, size in zip(local_offset, local_size, strict=True)
                    )
                    expected[local_slices].copy_(dense[global_slices])

                actual_generator = torch.Generator(device=device).manual_seed(123)
                actual = torch.empty(local_shape, device=device)
                self._set_shard_metadata(
                    actual,
                    global_shape,
                    global_offsets,
                    local_offsets,
                    local_sizes,
                )
                with _StatefulRNGMode():
                    actual.normal_(0.1, 0.02, generator=actual_generator)

                self.assertEqual(actual, expected, rtol=0, atol=0)
                self.assertEqual(actual_generator.get_state(), expected_state)

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

        with self.assertRaisesRegex(RuntimeError, "cover the entire tensor"):
            with _StatefulRNGMode():
                actual.normal_(generator=generator)

        self.assertEqual(generator.get_state(), state)


class TestPhiloxDistributionShardsOp(TestCase):
    @unittest.skipIf(not TEST_CUDA, "CUDA is required")
    def test_uniform_matches_dense(self):
        device = torch.device("cuda")
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                generator = torch.Generator(device=device).manual_seed(123)
                expected = torch.empty(17, dtype=dtype, device=device).uniform_(
                    -0.2, 0.3, generator=generator
                )
                expected_state = generator.get_state()

                generator.manual_seed(123)
                actual = torch.empty_like(expected)
                torch.ops.aten._philox_distribution_shards_(
                    actual,
                    [17],
                    [0],
                    [0],
                    [17],
                    1,
                    _PHILOX_DISTRIBUTION_UNIFORM,
                    [-0.2, 0.3],
                    generator=generator,
                )

                self.assertEqual(actual, expected, rtol=0, atol=0)
                self.assertEqual(generator.get_state(), expected_state)

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
            "local shards 0 and 1 must not overlap",
            lambda: torch.ops.aten._philox_distribution_shards_(
                torch.empty(2, device=device),
                [4],
                [0, 2],
                [0, 0],
                [1, 1],
                2,
                _PHILOX_DISTRIBUTION_NORMAL,
                [0.0, 1.0],
                generator=generator,
            ),
        )
        assert_invalid_without_advancing(
            "normal expects std >= 0.0",
            lambda: torch.ops.aten._philox_distribution_shards_(
                torch.empty(1, device=device),
                [1],
                [0],
                [0],
                [1],
                1,
                _PHILOX_DISTRIBUTION_NORMAL,
                [0.0, -1.0],
                generator=generator,
            ),
        )
        assert_invalid_without_advancing(
            "found from=1.*> to=0",
            lambda: torch.ops.aten._philox_distribution_shards_(
                torch.empty(1, device=device),
                [1],
                [0],
                [0],
                [1],
                1,
                _PHILOX_DISTRIBUTION_UNIFORM,
                [1.0, 0.0],
                generator=generator,
            ),
        )


if __name__ == "__main__":
    run_tests()
