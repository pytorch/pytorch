# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

from __future__ import annotations

import unittest
from typing import Any, cast

import torch
from torch._library.utils import fill_defaults
from torch.distributed.checkpoint import CheckpointableTensor
from torch.testing._internal.common_utils import run_tests, TEST_CUDA, TestCase
from torch.utils._python_dispatch import TorchDispatchMode


aten = torch.ops.aten

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
        if func not in (aten.normal_.default, aten.uniform_.default):
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

        aten._philox_distribution_shards_.default(
            tensor_arg,
            rng_metadata.global_shape,
            _flatten_chunks(rng_metadata.global_offsets),
            _flatten_chunks(rng_metadata.local_offsets),
            _flatten_chunks(rng_metadata.local_sizes),
            len(rng_metadata.global_offsets),
            (
                _PHILOX_DISTRIBUTION_NORMAL
                if func is aten.normal_.default
                else _PHILOX_DISTRIBUTION_UNIFORM
            ),
            tuple(filled_args[1:]),
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
    def test_noncontiguous_local_tensor_matches_dense(self):
        device = torch.device("cuda")
        generator = torch.Generator(device=device).manual_seed(123)
        expected = torch.empty((3, 2), device=device).normal_(
            0.1, 0.02, generator=generator
        )
        expected_state = generator.get_state()

        generator.manual_seed(123)
        actual = torch.empty((2, 3), device=device).t()
        self._set_shard_metadata(
            actual,
            (3, 2),
            ((0, 0),),
            ((0, 0),),
            ((3, 2),),
        )
        with _StatefulRNGMode():
            actual.normal_(0.1, 0.02, generator=generator)

        self.assertEqual(actual, expected, rtol=0, atol=0)
        self.assertEqual(generator.get_state(), expected_state)

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
        cases = (
            ("normal", (0.1, 0.02)),
            ("uniform", (-0.2, 0.3)),
        )

        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            for op_name, op_args in cases:
                with self.subTest(dtype=dtype, op=op_name):
                    expected_generator = torch.Generator(device=device).manual_seed(123)
                    torch.rand(11, device=device, generator=expected_generator)
                    expected = []
                    for _ in range(2):
                        dense = torch.empty(global_numel, dtype=dtype, device=device)
                        getattr(dense, f"{op_name}_")(
                            *op_args, generator=expected_generator
                        )
                        expected.append(dense[global_indices])
                    expected_state = expected_generator.get_state()

                    actual_generator = torch.Generator(device=device).manual_seed(123)
                    torch.rand(11, device=device, generator=actual_generator)
                    actual = torch.empty(
                        global_indices.numel(), dtype=dtype, device=device
                    )
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
                            getattr(actual, f"{op_name}_")(
                                *op_args, generator=actual_generator
                            )
                            results.append(actual.clone())

                    self.assertEqual(results, expected, rtol=0, atol=0)
                    self.assertEqual(actual_generator.get_state(), expected_state)

    @unittest.skipIf(not TEST_CUDA, "CUDA is required")
    def test_invalid_parameters_do_not_advance_generator(self):
        device = torch.device("cuda")
        max_float = torch.finfo(torch.float32).max
        cases = (
            ("negative_std", "std >= 0.0", "normal_", (0.0, -1.0)),
            ("reversed_uniform", "from=1.*> to=0", "uniform_", (1.0, 0.0)),
            ("wide_uniform", "to-from", "uniform_", (-max_float, max_float)),
        )

        for generator_kind in ("default", "explicit"):
            for case_name, error, op_name, op_args in cases:
                with self.subTest(generator=generator_kind, case=case_name):
                    torch.manual_seed(321)
                    generator = (
                        None
                        if generator_kind == "default"
                        else torch.Generator(device=device).manual_seed(321)
                    )
                    torch.rand(11, device=device, generator=generator)
                    before = (
                        torch.cuda.get_rng_state(device)
                        if generator is None
                        else generator.get_state()
                    )

                    actual = torch.empty(3, device=device)
                    self._set_shard_metadata(
                        actual,
                        (7,),
                        ((2,),),
                        ((0,),),
                        ((3,),),
                    )
                    with self.assertRaisesRegex(RuntimeError, error):
                        with _StatefulRNGMode():
                            getattr(actual, op_name)(*op_args, generator=generator)

                    after = (
                        torch.cuda.get_rng_state(device)
                        if generator is None
                        else generator.get_state()
                    )
                    self.assertEqual(after, before)

    @unittest.skipIf(not TEST_CUDA, "CUDA is required")
    def test_empty_local_tensor_matches_dense_increment(self):
        device = torch.device("cuda")
        for op_name in ("normal", "uniform"):
            for global_numel in (0, 7):
                with self.subTest(op=op_name, global_numel=global_numel):
                    torch.manual_seed(123)
                    getattr(torch.empty(global_numel, device=device), f"{op_name}_")()
                    expected_state = torch.cuda.get_rng_state(device)

                    torch.manual_seed(123)
                    actual = torch.empty(0, device=device)
                    self._set_shard_metadata(actual, (global_numel,), (), (), ())
                    with _StatefulRNGMode():
                        getattr(actual, f"{op_name}_")()

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
    @staticmethod
    def _run(
        tensor,
        global_shape,
        global_offsets,
        local_offsets,
        local_sizes,
        chunk_count,
        distribution=_PHILOX_DISTRIBUTION_UNIFORM,
        params=(0.0, 1.0),
        generator=None,
    ):
        return torch.ops.aten._philox_distribution_shards_(
            tensor,
            global_shape,
            global_offsets,
            local_offsets,
            local_sizes,
            chunk_count,
            distribution,
            params,
            generator=generator,
        )

    @unittest.skipIf(not TEST_CUDA, "CUDA is required")
    def test_scalar_matches_dense(self):
        device = torch.device("cuda")
        expected_generator = torch.Generator(device=device).manual_seed(123)
        expected = torch.empty((), device=device).uniform_(
            -0.2, 0.3, generator=expected_generator
        )
        expected_state = expected_generator.get_state()

        actual_generator = torch.Generator(device=device).manual_seed(123)
        actual = torch.empty((), device=device)
        self._run(
            actual,
            [],
            [],
            [],
            [],
            1,
            _PHILOX_DISTRIBUTION_UNIFORM,
            [-0.2, 0.3],
            actual_generator,
        )

        self.assertEqual(actual, expected, rtol=0, atol=0)
        self.assertEqual(actual_generator.get_state(), expected_state)

    @unittest.skipIf(not TEST_CUDA, "CUDA is required")
    def test_size_one_dimensions_do_not_count_toward_kernel_limit(self):
        device = torch.device("cuda")
        shape = (1,) * 31 + (2,)
        expected_generator = torch.Generator(device=device).manual_seed(123)
        expected = torch.empty(shape, device=device).normal_(
            generator=expected_generator
        )

        actual_generator = torch.Generator(device=device).manual_seed(123)
        actual = torch.empty(shape, device=device)
        self._run(
            actual,
            shape,
            [0] * len(shape),
            [0] * len(shape),
            list(shape),
            1,
            _PHILOX_DISTRIBUTION_NORMAL,
            [0.0, 1.0],
            actual_generator,
        )

        self.assertEqual(actual, expected, rtol=0, atol=0)
        self.assertEqual(actual_generator.get_state(), expected_generator.get_state())

    @unittest.skipIf(not TEST_CUDA, "CUDA is required")
    def test_multidimensional_chunk_matches_dense(self):
        device = torch.device("cuda")
        global_shape = (4, 5, 6)
        expected_generator = torch.Generator(device=device).manual_seed(123)
        expected = torch.empty(global_shape, device=device).normal_(
            0.1, 0.02, generator=expected_generator
        )
        expected_state = expected_generator.get_state()

        actual_generator = torch.Generator(device=device).manual_seed(123)
        actual = torch.empty((2, 3, 3), device=device)
        returned = self._run(
            actual,
            global_shape,
            [1, 1, 2],
            [0, 0, 0],
            [2, 3, 3],
            1,
            _PHILOX_DISTRIBUTION_NORMAL,
            [0.1, 0.02],
            actual_generator,
        )

        self.assertIs(returned, actual)
        self.assertEqual(actual, expected[1:3, 1:4, 2:5], rtol=0, atol=0)
        self.assertEqual(actual_generator.get_state(), expected_state)

    @unittest.skipIf(not TEST_CUDA, "CUDA is required")
    def test_reordered_chunks_write_to_local_offsets(self):
        device = torch.device("cuda")
        expected_generator = torch.Generator(device=device).manual_seed(123)
        expected = torch.empty(8, device=device).uniform_(
            -0.2, 0.3, generator=expected_generator
        )
        expected_state = expected_generator.get_state()

        actual_generator = torch.Generator(device=device).manual_seed(123)
        actual = torch.empty(6, device=device)
        returned = self._run(
            actual,
            [8],
            [7, 4, 1],
            [5, 0, 2],
            [1, 2, 3],
            3,
            _PHILOX_DISTRIBUTION_UNIFORM,
            [-0.2, 0.3],
            actual_generator,
        )

        self.assertIs(returned, actual)
        self.assertEqual(actual, expected[[4, 5, 1, 2, 3, 7]], rtol=0, atol=0)
        self.assertEqual(actual_generator.get_state(), expected_state)

    @unittest.skipIf(not TEST_CUDA, "CUDA is required")
    def test_invalid_calls_do_not_advance_generator(self):
        device = torch.device("cuda")
        generator = torch.Generator(device=device).manual_seed(123)

        def assert_invalid_without_advancing(regex, fn):
            state = generator.get_state().clone()
            with self.assertRaisesRegex(RuntimeError, regex):
                fn()
            self.assertEqual(generator.get_state(), state)

        cases = (
            (
                "metadata arrays",
                lambda: self._run(
                    torch.empty(2, device=device),
                    [4],
                    [0],
                    [0],
                    [],
                    1,
                    generator=generator,
                ),
            ),
            (
                "outside global_shape",
                lambda: self._run(
                    torch.empty(2, device=device),
                    [4],
                    [3],
                    [0],
                    [2],
                    1,
                    generator=generator,
                ),
            ),
            (
                "outside self shape",
                lambda: self._run(
                    torch.empty(2, device=device),
                    [4],
                    [0],
                    [1],
                    [2],
                    1,
                    generator=generator,
                ),
            ),
            (
                "global shards.*overlap",
                lambda: self._run(
                    torch.empty(4, device=device),
                    [8],
                    [0, 1],
                    [0, 2],
                    [2, 2],
                    2,
                    generator=generator,
                ),
            ),
            (
                "local shards.*overlap",
                lambda: self._run(
                    torch.empty(4, device=device),
                    [8],
                    [0, 4],
                    [0, 1],
                    [2, 2],
                    2,
                    generator=generator,
                ),
            ),
            (
                "cover the entire tensor",
                lambda: self._run(
                    torch.empty(3, device=device),
                    [7],
                    [2],
                    [0],
                    [2],
                    1,
                    generator=generator,
                ),
            ),
            (
                "more than one element of the written-to tensor",
                lambda: self._run(
                    torch.empty(1, device=device).expand(2),
                    [2],
                    [0],
                    [0],
                    [2],
                    1,
                    generator=generator,
                ),
            ),
            (
                "normal expects std >= 0.0",
                lambda: self._run(
                    torch.empty(1, device=device),
                    [1],
                    [0],
                    [0],
                    [1],
                    1,
                    _PHILOX_DISTRIBUTION_NORMAL,
                    [0.0, -1.0],
                    generator,
                ),
            ),
            (
                "found from=1.*> to=0",
                lambda: self._run(
                    torch.empty(1, device=device),
                    [1],
                    [0],
                    [0],
                    [1],
                    1,
                    _PHILOX_DISTRIBUTION_UNIFORM,
                    [1.0, 0.0],
                    generator,
                ),
            ),
            (
                "from is out of bounds",
                lambda: self._run(
                    torch.empty(1, dtype=torch.float16, device=device),
                    [1],
                    [0],
                    [0],
                    [1],
                    1,
                    _PHILOX_DISTRIBUTION_UNIFORM,
                    [-70000.0, 0.0],
                    generator,
                ),
            ),
            (
                "unsupported distribution kind 2",
                lambda: self._run(
                    torch.empty(1, device=device),
                    [1],
                    [0],
                    [0],
                    [1],
                    1,
                    2,
                    [0.0, 1.0],
                    generator,
                ),
            ),
            (
                "expects 2 parameters, got 1",
                lambda: self._run(
                    torch.empty(1, device=device),
                    [1],
                    [0],
                    [0],
                    [1],
                    1,
                    _PHILOX_DISTRIBUTION_NORMAL,
                    [0.0],
                    generator,
                ),
            ),
            (
                "parameters must be real",
                lambda: self._run(
                    torch.empty(1, device=device),
                    [1],
                    [0],
                    [0],
                    [1],
                    1,
                    _PHILOX_DISTRIBUTION_NORMAL,
                    [0j, 1.0],
                    generator,
                ),
            ),
        )
        for regex, fn in cases:
            with self.subTest(error=regex):
                assert_invalid_without_advancing(regex, fn)

if __name__ == "__main__":
    run_tests()
