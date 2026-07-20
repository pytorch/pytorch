# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

from __future__ import annotations

import unittest
from functools import partial
from typing import Any, cast, TYPE_CHECKING

import torch
from torch._library.utils import fill_defaults
from torch.distributed import StatefulRNGTensor
from torch.distributed._local_tensor import LocalIntNode, LocalTensor, LocalTensorMode
from torch.distributed._stateful_rng import (
    _is_supported_stateful_rng_op,
    _PHILOX_DISTRIBUTION_NORMAL,
    _PHILOX_DISTRIBUTION_UNIFORM,
    _run_stateful_rng_op,
)
from torch.testing._internal.common_utils import run_tests, TEST_CUDA, TestCase
from torch.utils._python_dispatch import TorchDispatchMode


if TYPE_CHECKING:
    from torch.distributed import RNGIndexBlock


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
        if func not in (aten.normal_.default, aten.uniform_.default):
            return func(*args, **kwargs)

        filled_args, _ = fill_defaults(func._schema, args, kwargs)
        tensor_arg = filled_args[0]
        if not isinstance(tensor_arg, torch.Tensor):
            return func(*args, **kwargs)
        if tensor_arg.is_meta or tensor_arg.device.type != "cuda":
            return func(*args, **kwargs)
        if not isinstance(tensor_arg, StatefulRNGTensor):
            return func(*args, **kwargs)
        rng_metadata = cast(StatefulRNGTensor, tensor_arg)

        return _run_stateful_rng_op(
            func,
            args,
            kwargs,
            rng_metadata.rng_global_numel,
            rng_metadata.rng_index_blocks,
        )


class TestStatefulRNGTensor(TestCase):
    @staticmethod
    def _set_rng_metadata(
        tensor: torch.Tensor,
        global_numel: int,
        index_blocks: tuple[RNGIndexBlock, ...],
    ) -> None:
        setattr(tensor, "rng_global_numel", global_numel)  # noqa: B010
        setattr(tensor, "rng_index_blocks", index_blocks)  # noqa: B010

    def test_plain_tensor_metadata_satisfies_protocol(self):
        tensor = torch.empty(1)
        self.assertNotIsInstance(tensor, StatefulRNGTensor)

        block: RNGIndexBlock = (0, 1, 1, 1)
        self._set_rng_metadata(tensor, 1, (block,))
        self.assertIsInstance(tensor, StatefulRNGTensor)

    @unittest.skipIf(not TEST_CUDA, "CUDA is required")
    def test_initializers_match_dense(self):
        device = torch.device("cuda")
        global_shape = (5, 7)
        normal = partial(torch.nn.init.normal_, mean=0.1, std=0.02)
        uniform = partial(torch.nn.init.uniform_, a=-0.2, b=0.3)
        trunc_normal = partial(
            torch.nn.init.trunc_normal_,
            mean=0.0,
            std=0.02,
            a=-0.06,
            b=0.06,
        )
        cases = (
            (
                "normal_contiguous",
                normal,
                (slice(3, 5), slice(None)),
                ((21, 14, 14, 1),),
            ),
            (
                "normal_strided",
                normal,
                (slice(None), slice(4, 7)),
                ((4, 3, 7, 5),),
            ),
            (
                "uniform_contiguous",
                uniform,
                (slice(3, 5), slice(None)),
                ((21, 14, 14, 1),),
            ),
            (
                "uniform_strided",
                uniform,
                (slice(None), slice(4, 7)),
                ((4, 3, 7, 5),),
            ),
            (
                "trunc_normal_strided",
                trunc_normal,
                (slice(None), slice(4, 7)),
                ((4, 3, 7, 5),),
            ),
        )

        for case_name, init_fn, global_slice, index_blocks in cases:
            with self.subTest(case=case_name):
                torch.manual_seed(123)
                expected = torch.empty(global_shape, device=device)
                init_fn(expected)
                expected_state = torch.cuda.get_rng_state(device)

                torch.manual_seed(123)
                actual = torch.empty(expected[global_slice].shape, device=device)
                self._set_rng_metadata(actual, expected.numel(), index_blocks)
                with _StatefulRNGMode():
                    init_fn(actual)

                self.assertEqual(
                    actual, expected[global_slice].contiguous(), rtol=0, atol=0
                )
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
        index_blocks = (
            (2, 2, total_stride, 3),
            (4 * total_stride + 5, 3, 3, 1),
        )
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
                    self._set_rng_metadata(actual, global_numel, index_blocks)
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
    def test_private_adapter_replays_explicit_generator_for_local_tensor(self):
        device = torch.device("cuda")
        expected_generator = torch.Generator(device=device).manual_seed(123)
        expected = torch.empty(7, device=device).uniform_(
            -0.2, 0.3, generator=expected_generator
        )
        expected_state = expected_generator.get_state()

        actual_generator = torch.Generator(device=device).manual_seed(123)
        local_tensor = LocalTensor(
            {
                0: torch.empty(3, device=device),
                1: torch.empty(3, device=device),
            }
        )
        with LocalTensorMode(local_tensor._ranks):
            self.assertTrue(
                _is_supported_stateful_rng_op(
                    torch.ops.aten.uniform_.default, local_tensor
                )
            )
            returned = _run_stateful_rng_op(
                torch.ops.aten.uniform_.default,
                (local_tensor, -0.2, 0.3),
                {"generator": actual_generator},
                7,
                ((2, 3, 3, 1),),
            )

        self.assertIs(returned, local_tensor)
        for local_result in local_tensor._local_tensors.values():
            self.assertEqual(local_result, expected[2:5], rtol=0, atol=0)
        self.assertEqual(actual_generator.get_state(), expected_state)

    @unittest.skipIf(not TEST_CUDA, "CUDA is required")
    def test_private_adapter_default_generator_with_rank_local_layout(self):
        device = torch.device("cuda")
        torch.manual_seed(123)
        expected = torch.empty(7, device=device).uniform_(-0.2, 0.3)
        expected_state = torch.cuda.get_rng_state(device)

        torch.manual_seed(123)
        local_tensor = LocalTensor(
            {
                0: torch.empty(3, device=device),
                1: torch.empty(3, device=device),
            }
        )
        rank_local_start = torch.SymInt(LocalIntNode({0: 0, 1: 4}))
        with LocalTensorMode(local_tensor._ranks):
            returned = _run_stateful_rng_op(
                torch.ops.aten.uniform_.default,
                (local_tensor, -0.2, 0.3),
                {},
                7,
                ((rank_local_start, 3, 3, 1),),
            )

        self.assertIs(returned, local_tensor)
        self.assertEqual(local_tensor._local_tensors[0], expected[:3], rtol=0, atol=0)
        self.assertEqual(local_tensor._local_tensors[1], expected[4:], rtol=0, atol=0)
        self.assertEqual(torch.cuda.get_rng_state(device), expected_state)

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
                    self._set_rng_metadata(actual, 7, ((2, 3, 3, 1),))
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
                    self._set_rng_metadata(actual, global_numel, ())
                    with _StatefulRNGMode():
                        getattr(actual, f"{op_name}_")()

                    self.assertEqual(torch.cuda.get_rng_state(device), expected_state)


class TestPhiloxDistributionFlatSliceOp(TestCase):
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
            lambda: torch.ops.aten._philox_distribution_flat_slice_(
                torch.empty(2, device=device),
                4,
                [0],
                [2],
                [1],
                [1],
                _PHILOX_DISTRIBUTION_UNIFORM,
                [0.0, 1.0],
                generator=generator,
            ),
        )
        assert_invalid_without_advancing(
            "normal expects std >= 0.0",
            lambda: torch.ops.aten._philox_distribution_flat_slice_(
                torch.empty(1, device=device),
                1,
                [0],
                [1],
                [1],
                [1],
                _PHILOX_DISTRIBUTION_NORMAL,
                [0.0, -1.0],
                generator=generator,
            ),
        )
        assert_invalid_without_advancing(
            "found from=1.*> to=0",
            lambda: torch.ops.aten._philox_distribution_flat_slice_(
                torch.empty(1, device=device),
                1,
                [0],
                [1],
                [1],
                [1],
                _PHILOX_DISTRIBUTION_UNIFORM,
                [1.0, 0.0],
                generator=generator,
            ),
        )
        assert_invalid_without_advancing(
            "from is out of bounds",
            lambda: torch.ops.aten._philox_distribution_flat_slice_(
                torch.empty(1, dtype=torch.float16, device=device),
                1,
                [0],
                [1],
                [1],
                [1],
                _PHILOX_DISTRIBUTION_UNIFORM,
                [-70000.0, 0.0],
                generator=generator,
            ),
        )
        assert_invalid_without_advancing(
            "unsupported distribution kind 2",
            lambda: torch.ops.aten._philox_distribution_flat_slice_(
                torch.empty(1, device=device),
                1,
                [0],
                [1],
                [1],
                [1],
                2,
                [0.0, 1.0],
                generator=generator,
            ),
        )
        assert_invalid_without_advancing(
            "expects 2 parameters, got 1",
            lambda: torch.ops.aten._philox_distribution_flat_slice_(
                torch.empty(1, device=device),
                1,
                [0],
                [1],
                [1],
                [1],
                _PHILOX_DISTRIBUTION_NORMAL,
                [0.0],
                generator=generator,
            ),
        )
        assert_invalid_without_advancing(
            "parameters must be real",
            lambda: torch.ops.aten._philox_distribution_flat_slice_(
                torch.empty(1, device=device),
                1,
                [0],
                [1],
                [1],
                [1],
                _PHILOX_DISTRIBUTION_NORMAL,
                [0j, 1.0],
                generator=generator,
            ),
        )

    @unittest.skipIf(not TEST_CUDA, "CUDA is required")
    def test_legacy_zero_overlap_and_order_semantics(self):
        device = torch.device("cuda")
        expected_generator = torch.Generator(device=device).manual_seed(123)
        expected_dense = torch.empty(8, device=device).uniform_(
            -0.2, 0.3, generator=expected_generator
        )
        expected_state = expected_generator.get_state()

        actual_generator = torch.Generator(device=device).manual_seed(123)
        actual = torch.empty(6, device=device)
        returned = torch.ops.aten._philox_distribution_flat_slice_(
            actual,
            8,
            [8, 4, 7, 1, 2],
            [0, 2, 3, 3, 1],
            [-1, 2, -1, 3, 1],
            [1, 1, 0, 1, 1],
            _PHILOX_DISTRIBUTION_UNIFORM,
            [-0.2, 0.3],
            generator=actual_generator,
        )

        self.assertIs(returned, actual)
        self.assertEqual(actual, expected_dense[[4, 5, 1, 2, 3, 2]], rtol=0, atol=0)
        self.assertEqual(actual_generator.get_state(), expected_state)

    def test_meta_validation(self):
        result = torch.empty(6, device="meta")
        returned = torch.ops.aten._philox_distribution_flat_slice_(
            result,
            17,
            [1, 10],
            [2, 2],
            [4, 3],
            [2, 1],
            _PHILOX_DISTRIBUTION_NORMAL,
            [0.0, 1.0],
        )
        self.assertIs(returned, result)

        with self.assertRaisesRegex(RuntimeError, "normal expects std >= 0.0"):
            torch.ops.aten._philox_distribution_flat_slice_(
                torch.empty(1, device="meta"),
                1,
                [0],
                [1],
                [1],
                [1],
                _PHILOX_DISTRIBUTION_NORMAL,
                [0.0, -1.0],
            )
        with self.assertRaisesRegex(RuntimeError, "found from=1.*> to=0"):
            torch.ops.aten._philox_distribution_flat_slice_(
                torch.empty(1, device="meta"),
                1,
                [0],
                [1],
                [1],
                [1],
                _PHILOX_DISTRIBUTION_UNIFORM,
                [1.0, 0.0],
            )
        with self.assertRaisesRegex(RuntimeError, "parameters must be real"):
            torch.ops.aten._philox_distribution_flat_slice_(
                torch.empty(1, device="meta"),
                1,
                [0],
                [1],
                [1],
                [1],
                _PHILOX_DISTRIBUTION_NORMAL,
                [0j, 1.0],
            )
        with self.assertRaisesRegex(RuntimeError, "unsupported distribution kind 2"):
            torch.ops.aten._philox_distribution_flat_slice_(
                torch.empty(1, device="meta"),
                1,
                [0],
                [1],
                [1],
                [1],
                2,
                [0.0, 1.0],
            )
        legacy_result = torch.empty(6, device="meta")
        returned = torch.ops.aten._philox_distribution_flat_slice_(
            legacy_result,
            8,
            [8, 4, 7, 1, 2],
            [0, 2, 3, 3, 1],
            [-1, 2, -1, 3, 1],
            [1, 1, 0, 1, 1],
            _PHILOX_DISTRIBUTION_UNIFORM,
            [0.0, 1.0],
        )
        self.assertIs(returned, legacy_result)


if __name__ == "__main__":
    run_tests()
