# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

from collections.abc import Callable
from typing import Any, cast

import torch
from torch._library.utils import fill_defaults
from torch.distributed.checkpoint import CheckpointableTensor
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase
from torch.utils._python_dispatch import TorchDispatchMode


# Must match PhiloxDistributionKind::Normal.
_NORMAL_DISTRIBUTION = 0


def _flatten_chunks(chunks: tuple[tuple[int, ...], ...]) -> list[int]:
    return [value for chunk in chunks for value in chunk]


def _normal_shards_(
    tensor: torch.Tensor,
    metadata: CheckpointableTensor,
    mean: float,
    std: float,
    generator: torch.Generator | None,
) -> torch.Tensor:
    return torch.ops.aten._philox_distribution_shards_.default(
        tensor,
        metadata.global_shape,
        _flatten_chunks(metadata.global_offsets),
        _flatten_chunks(metadata.local_offsets),
        _flatten_chunks(metadata.local_sizes),
        len(metadata.global_offsets),
        _NORMAL_DISTRIBUTION,
        (mean, std),
        generator=generator,
    )


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
        if func is not torch.ops.aten.normal_.default:
            return func(*args, **kwargs)

        filled_args, filled_kwargs = fill_defaults(func._schema, args, kwargs)
        tensor_arg = filled_args[0]
        if not isinstance(tensor_arg, torch.Tensor):
            return func(*args, **kwargs)
        if tensor_arg.is_meta or tensor_arg.device.type != "cuda":
            return func(*args, **kwargs)
        if not isinstance(tensor_arg, CheckpointableTensor):
            return func(*args, **kwargs)
        if tensor_arg.dtype not in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        ):
            return func(*args, **kwargs)
        rng_metadata = cast(CheckpointableTensor, tensor_arg)

        _, mean, std = filled_args
        _normal_shards_(
            tensor_arg,
            rng_metadata,
            mean,
            std,
            filled_kwargs["generator"],
        )
        return tensor_arg


class TestCheckpointableTensorRNG(TestCase):
    def _set_shard_metadata(
        self,
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

    def _layout_case(self, layout: str, total_stride: int):
        cases = {
            "row_slice": (
                (5, 7),
                (2, 7),
                ((3, 0),),
                ((0, 0),),
                ((2, 7),),
            ),
            # Reorder chunks that span CUDA grid-stride iterations.
            "reordered_column_chunks": (
                (5, total_stride + 8),
                (5, 4),
                ((0, total_stride + 4), (0, 0)),
                ((0, 2), (0, 0)),
                ((5, 2), (5, 2)),
            ),
            "large_row_slice": (
                (10, total_stride + 8),
                (5, total_stride + 8),
                ((5, 0),),
                ((0, 0),),
                ((5, total_stride + 8),),
            ),
            "three_dimensional_slice": (
                (4, 2, 3),
                (2, 2, 3),
                ((2, 0, 0),),
                ((0, 0, 0),),
                ((2, 2, 3),),
            ),
            "scalar": ((), (), ((),), ((),), ((),)),
            "empty_shard": (
                (7,),
                (0,),
                (),
                (),
                (),
            ),
            "empty_global": (
                (0,),
                (0,),
                (),
                (),
                (),
            ),
        }
        return cases[layout]

    @parametrize(
        "layout",
        [
            "row_slice",
            "reordered_column_chunks",
            "large_row_slice",
            "three_dimensional_slice",
            "scalar",
            "empty_shard",
            "empty_global",
        ],
    )
    @dtypes(torch.float16, torch.bfloat16, torch.float32, torch.float64)
    def test_layouts_match_dense(self, device, dtype, layout):
        properties = torch.cuda.get_device_properties(device)
        block_size = 256
        total_stride = (
            block_size
            * properties.multi_processor_count
            * (properties.max_threads_per_multi_processor // block_size)
        )
        (
            global_shape,
            local_shape,
            global_offsets,
            local_offsets,
            local_sizes,
        ) = self._layout_case(layout, total_stride)
        expected_generator = torch.Generator(device=device).manual_seed(123)
        torch.rand(11, device=device, generator=expected_generator)
        dense = torch.empty(global_shape, dtype=dtype, device=device).normal_(
            0.1, 0.02, generator=expected_generator
        )
        expected_state = expected_generator.get_state()
        expected = torch.empty(local_shape, dtype=dtype, device=device)
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
        torch.rand(11, device=device, generator=actual_generator)
        actual = torch.empty(local_shape, dtype=dtype, device=device)
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

    @dtypes(torch.float32)
    def test_noncontiguous_local_matches_dense(self, device, dtype):
        global_shape = (3, 4)
        global_offsets = ((0, 1),)
        local_offsets = ((0, 0),)
        local_sizes = ((3, 2),)

        expected_generator = torch.Generator(device=device).manual_seed(123)
        dense = torch.empty(global_shape, dtype=dtype, device=device).normal_(
            0.1, 0.02, generator=expected_generator
        )
        expected_state = expected_generator.get_state()

        actual_generator = torch.Generator(device=device).manual_seed(123)
        actual = torch.empty((2, 3), dtype=dtype, device=device).transpose(0, 1)
        self.assertFalse(actual.is_contiguous())
        self._set_shard_metadata(
            actual,
            global_shape,
            global_offsets,
            local_offsets,
            local_sizes,
        )
        with _StatefulRNGMode():
            actual.normal_(0.1, 0.02, generator=actual_generator)

        self.assertEqual(actual, dense[:, 1:3], rtol=0, atol=0)
        self.assertEqual(actual_generator.get_state(), expected_state)

    @dtypes(torch.float32)
    def test_default_generator_matches_dense(self, device, dtype):
        seed_state = torch.Generator(device=device).manual_seed(123).get_state()
        original_state = torch.cuda.get_rng_state(device)
        try:
            torch.cuda.set_rng_state(seed_state, device)
            dense = torch.empty((3, 3), dtype=dtype, device=device).normal_(0.1, 0.02)
            expected_state = torch.cuda.get_rng_state(device)

            torch.cuda.set_rng_state(seed_state, device)
            actual = torch.empty((2, 3), dtype=dtype, device=device)
            self._set_shard_metadata(
                actual,
                (3, 3),
                ((1, 0),),
                ((0, 0),),
                ((2, 3),),
            )
            with _StatefulRNGMode():
                actual.normal_(0.1, 0.02)

            self.assertEqual(actual, dense[1:], rtol=0, atol=0)
            self.assertEqual(torch.cuda.get_rng_state(device), expected_state)
        finally:
            torch.cuda.set_rng_state(original_state, device)

    @dtypes(torch.float32)
    def test_padding_is_untouched(self, device, dtype):
        expected_generator = torch.Generator(device=device).manual_seed(123)
        dense = torch.empty((2, 3), dtype=dtype, device=device).normal_(
            0.1, 0.02, generator=expected_generator
        )
        expected_state = expected_generator.get_state()
        expected = torch.full((4, 5), -1.0, dtype=dtype, device=device)
        expected[1:3, 1:4].copy_(dense)

        actual_generator = torch.Generator(device=device).manual_seed(123)
        actual = torch.full((4, 5), -1.0, dtype=dtype, device=device)
        self._set_shard_metadata(
            actual,
            (2, 3),
            ((0, 0),),
            ((1, 1),),
            ((2, 3),),
        )
        with _StatefulRNGMode():
            actual.normal_(0.1, 0.02, generator=actual_generator)

        self.assertEqual(actual, expected, rtol=0, atol=0)
        self.assertEqual(actual_generator.get_state(), expected_state)


class TestPhiloxDistributionShardsOp(TestCase):
    def test_invalid_calls_do_not_advance_generator(self, device):
        generator = torch.Generator(device=device).manual_seed(123)

        def assert_invalid_without_advancing(
            regex: str, fn: Callable[[], torch.Tensor]
        ) -> None:
            state = generator.get_state().clone()
            with self.assertRaisesRegex(RuntimeError, regex):
                fn()
            self.assertEqual(generator.get_state(), state)

        def call_distribution(
            distribution: int, params: list[complex | float]
        ) -> torch.Tensor:
            return torch.ops.aten._philox_distribution_shards_.default(
                torch.empty(1, device=device),
                [1],
                [0],
                [0],
                [1],
                1,
                distribution,
                params,
                generator=generator,
            )

        assert_invalid_without_advancing(
            "local shards 0 and 1 must not overlap",
            lambda: torch.ops.aten._philox_distribution_shards_.default(
                torch.empty(2, device=device),
                [4],
                [0, 2],
                [0, 0],
                [1, 1],
                2,
                _NORMAL_DISTRIBUTION,
                [0.0, 1.0],
                generator=generator,
            ),
        )
        assert_invalid_without_advancing(
            "normal expects std >= 0.0",
            lambda: call_distribution(_NORMAL_DISTRIBUTION, [0.0, -1.0]),
        )
        assert_invalid_without_advancing(
            "logical global tensor requires 64-bit indexing",
            lambda: torch.ops.aten._philox_distribution_shards_.default(
                torch.empty(0, dtype=torch.float64, device=device),
                [268435457],
                [],
                [],
                [],
                0,
                _NORMAL_DISTRIBUTION,
                [0.0, 1.0],
                generator=generator,
            ),
        )
        assert_invalid_without_advancing(
            "unsupported distribution kind 1",
            lambda: call_distribution(1, [0.0, 1.0]),
        )
        assert_invalid_without_advancing(
            "expects 2 parameters, got 1",
            lambda: call_distribution(_NORMAL_DISTRIBUTION, [0.0]),
        )
        assert_invalid_without_advancing(
            "parameters must be real",
            lambda: call_distribution(_NORMAL_DISTRIBUTION, [0.0j, 1.0]),
        )


instantiate_device_type_tests(
    TestCheckpointableTensorRNG, globals(), only_for=("cuda",)
)
instantiate_device_type_tests(
    TestPhiloxDistributionShardsOp, globals(), only_for=("cuda",)
)


if __name__ == "__main__":
    run_tests()
