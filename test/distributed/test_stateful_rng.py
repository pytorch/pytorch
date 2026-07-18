# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import contextlib
import unittest
from functools import partial

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed import stateful_rng_mode, StatefulRNGTensor
from torch.distributed._local_tensor import LocalIntNode, LocalTensor, LocalTensorMode
from torch.distributed._stateful_rng import (
    _is_supported_stateful_rng_op,
    _run_stateful_rng_op,
)
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.testing._internal.common_utils import run_tests, TEST_CUDA, TestCase


class TestStatefulRNGTensor(TestCase):
    @staticmethod
    def _set_rng_metadata(
        tensor: torch.Tensor,
        global_numel: int,
        index_blocks: tuple[tuple[int, int, int, int], ...],
    ) -> None:
        setattr(tensor, "rng_global_numel", global_numel)  # noqa: B010
        setattr(tensor, "rng_index_blocks", index_blocks)  # noqa: B010

    @unittest.skipIf(not TEST_CUDA, "CUDA is required")
    def test_shard_layouts_match_dense(self):
        device = torch.device("cuda")
        global_shape = (5, 7)
        layouts = (
            ("shard_0_rank_0", (slice(0, 3), slice(None)), ((0, 21, 21, 1),)),
            ("shard_0_rank_1", (slice(3, 5), slice(None)), ((21, 14, 14, 1),)),
            ("shard_1_rank_0", (slice(None), slice(0, 4)), ((0, 4, 7, 5),)),
            ("shard_1_rank_1", (slice(None), slice(4, 7)), ((4, 3, 7, 5),)),
        )
        init_fns = {
            "normal": partial(torch.nn.init.normal_, mean=0.1, std=0.02),
            "uniform": partial(torch.nn.init.uniform_, a=-0.2, b=0.3),
            "trunc_normal": partial(
                torch.nn.init.trunc_normal_,
                mean=0.0,
                std=0.02,
                a=-0.06,
                b=0.06,
            ),
        }

        for init_name, init_fn in init_fns.items():
            torch.manual_seed(123)
            expected = torch.empty(global_shape, device=device)
            init_fn(expected)
            expected_state = torch.cuda.get_rng_state(device)
            expected_next = torch.rand(17, device=device)

            for layout_name, global_slice, index_blocks in layouts:
                with self.subTest(init=init_name, layout=layout_name):
                    expected_local = expected[global_slice].contiguous()
                    torch.manual_seed(123)
                    actual = torch.empty(expected_local.shape, device=device)
                    self._set_rng_metadata(actual, expected.numel(), index_blocks)
                    self.assertIsInstance(actual, StatefulRNGTensor)
                    with stateful_rng_mode():
                        init_fn(actual)
                    actual_state = torch.cuda.get_rng_state(device)
                    actual_next = torch.rand(17, device=device)

                    self.assertEqual(actual, expected_local, rtol=0, atol=0)
                    self.assertEqual(actual_state, expected_state)
                    self.assertEqual(actual_next, expected_next, rtol=0, atol=0)

    @unittest.skipIf(not TEST_CUDA, "CUDA is required")
    def test_multiple_index_blocks_match_dense(self):
        device = torch.device("cuda")
        global_indices = torch.tensor([2, 3, 7, 8, 12, 13, 20, 21, 22], device=device)
        index_blocks = ((2, 2, 5, 3), (20, 3, 3, 1))

        torch.manual_seed(123)
        expected = torch.empty(24, device=device).normal_(0.1, 0.02)
        expected_state = torch.cuda.get_rng_state(device)
        expected_next = torch.rand(17, device=device)

        torch.manual_seed(123)
        actual = torch.empty(global_indices.numel(), device=device)
        self._set_rng_metadata(actual, expected.numel(), index_blocks)
        with stateful_rng_mode():
            actual.normal_(0.1, 0.02)

        actual_state = torch.cuda.get_rng_state(device)
        actual_next = torch.rand(17, device=device)
        self.assertEqual(actual, expected[global_indices], rtol=0, atol=0)
        self.assertEqual(actual_state, expected_state)
        self.assertEqual(actual_next, expected_next, rtol=0, atol=0)

    @unittest.skipIf(not TEST_CUDA, "CUDA is required")
    def test_explicit_generator_matches_dense(self):
        device = torch.device("cuda")
        expected_generator = torch.Generator(device=device).manual_seed(123)
        expected = torch.empty((5, 7), device=device)
        expected.uniform_(-0.2, 0.3, generator=expected_generator)
        expected_state = expected_generator.get_state()
        expected_next = torch.rand(17, device=device, generator=expected_generator)

        actual_generator = torch.Generator(device=device).manual_seed(123)
        actual = torch.empty((5, 3), device=device)
        self._set_rng_metadata(actual, expected.numel(), ((4, 3, 7, 5),))
        with stateful_rng_mode():
            actual.uniform_(-0.2, 0.3, generator=actual_generator)

        actual_state = actual_generator.get_state()
        actual_next = torch.rand(17, device=device, generator=actual_generator)
        self.assertEqual(actual, expected[:, 4:], rtol=0, atol=0)
        self.assertEqual(actual_state, expected_state)
        self.assertEqual(actual_next, expected_next, rtol=0, atol=0)

    @unittest.skipIf(not TEST_CUDA, "CUDA is required")
    def test_repeated_calls_from_nonzero_offset_match_dense(self):
        device = torch.device("cuda")
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
                        dense = torch.empty((5, 7), dtype=dtype, device=device)
                        getattr(dense, f"{op_name}_")(
                            *op_args, generator=expected_generator
                        )
                        expected.append(dense[:, 4:].contiguous())
                    expected_state = expected_generator.get_state()
                    expected_next = torch.rand(
                        17, device=device, generator=expected_generator
                    )

                    actual_generator = torch.Generator(device=device).manual_seed(123)
                    torch.rand(11, device=device, generator=actual_generator)
                    actual = torch.empty((5, 3), dtype=dtype, device=device)
                    self._set_rng_metadata(actual, 35, ((4, 3, 7, 5),))
                    results = []
                    with stateful_rng_mode():
                        for _ in range(2):
                            getattr(actual, f"{op_name}_")(
                                *op_args, generator=actual_generator
                            )
                            results.append(actual.clone())
                    actual_state = actual_generator.get_state()
                    actual_next = torch.rand(
                        17, device=device, generator=actual_generator
                    )

                    self.assertEqual(results, expected, rtol=0, atol=0)
                    self.assertEqual(actual_state, expected_state)
                    self.assertEqual(actual_next, expected_next, rtol=0, atol=0)

    @unittest.skipIf(not TEST_CUDA, "CUDA is required")
    def test_private_adapter_replays_explicit_generator_for_local_tensor(self):
        device = torch.device("cuda")
        expected_generator = torch.Generator(device=device).manual_seed(123)
        expected = torch.empty(7, device=device).uniform_(
            -0.2, 0.3, generator=expected_generator
        )
        expected_state = expected_generator.get_state()
        expected_next = torch.rand(17, device=device, generator=expected_generator)

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
        actual_state = actual_generator.get_state()
        actual_next = torch.rand(17, device=device, generator=actual_generator)
        self.assertEqual(actual_state, expected_state)
        self.assertEqual(actual_next, expected_next, rtol=0, atol=0)

    @unittest.skipIf(not TEST_CUDA, "CUDA is required")
    def test_private_adapter_default_generator_with_rank_local_layout(self):
        device = torch.device("cuda")
        torch.manual_seed(123)
        expected = torch.empty(7, device=device).uniform_(-0.2, 0.3)
        expected_state = torch.cuda.get_rng_state(device)
        expected_next = torch.rand(17, device=device)

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
        actual_state = torch.cuda.get_rng_state(device)
        actual_next = torch.rand(17, device=device)
        self.assertEqual(actual_state, expected_state)
        self.assertEqual(actual_next, expected_next, rtol=0, atol=0)

    @unittest.skipIf(not TEST_CUDA, "CUDA is required")
    def test_invalid_parameters_do_not_advance_generator(self):
        device = torch.device("cuda")
        max_float = torch.finfo(torch.float32).max
        cases = (
            ("negative_std", "std >= 0.0", "normal_", (0.0, -1.0)),
            ("reversed_uniform", "from=1.*> to=0", "uniform_", (1.0, 0.0)),
            (
                "wide_uniform",
                "to-from",
                "uniform_",
                (-max_float, max_float),
            ),
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
                    reference = torch.Generator(device=device)
                    reference.set_state(before)

                    actual = torch.empty(3, device=device)
                    self._set_rng_metadata(actual, 7, ((2, 3, 3, 1),))
                    with self.assertRaisesRegex(RuntimeError, error):
                        with stateful_rng_mode():
                            getattr(actual, op_name)(*op_args, generator=generator)

                    after = (
                        torch.cuda.get_rng_state(device)
                        if generator is None
                        else generator.get_state()
                    )
                    self.assertEqual(after, before)
                    actual_next = torch.rand(17, device=device, generator=generator)
                    expected_next = torch.rand(17, device=device, generator=reference)
                    self.assertEqual(actual_next, expected_next, rtol=0, atol=0)

    @unittest.skipIf(not TEST_CUDA, "CUDA is required")
    def test_empty_local_tensor_reserves_dense_increment(self):
        device = torch.device("cuda")
        torch.manual_seed(123)
        torch.empty(7, device=device).uniform_()
        expected_state = torch.cuda.get_rng_state(device)

        torch.manual_seed(123)
        actual = torch.empty(0, device=device)
        self._set_rng_metadata(actual, 7, ())
        with stateful_rng_mode():
            actual.uniform_()

        self.assertEqual(torch.cuda.get_rng_state(device), expected_state)


class TestPhiloxFlatSliceOps(TestCase):
    def test_symbolic_total_numel_meta_and_fake(self):
        self.assertIn(
            "SymInt total_numel",
            str(torch.ops.aten._philox_normal_flat_slice_.default._schema),
        )
        self.assertIn(
            "SymInt total_numel",
            str(torch.ops.aten._philox_uniform_flat_slice_.default._schema),
        )

        for device in ("meta", "cuda"):
            with self.subTest(device=device):
                shape_env = ShapeEnv()
                total_numel = shape_env.create_unbacked_symint()
                context = (
                    FakeTensorMode(shape_env=shape_env)
                    if device == "cuda"
                    else contextlib.nullcontext()
                )
                with context:
                    result = torch.empty(1, device=device)
                    returned = torch.ops.aten._philox_uniform_flat_slice_(
                        result, total_numel, [0], [1], [1], [1]
                    )

                self.assertIs(returned, result)
                self.assertIsNone(total_numel.node.maybe_as_int())

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
            lambda: torch.ops.aten._philox_uniform_flat_slice_(
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
        assert_invalid_without_advancing(
            "found from=1.*> to=0",
            lambda: torch.ops.aten._philox_uniform_flat_slice_(
                torch.empty(1, device=device),
                1,
                [0],
                [1],
                [1],
                [1],
                1,
                0,
                generator=generator,
            ),
        )
        assert_invalid_without_advancing(
            "from is out of bounds",
            lambda: torch.ops.aten._philox_uniform_flat_slice_(
                torch.empty(1, dtype=torch.float16, device=device),
                1,
                [0],
                [1],
                [1],
                [1],
                -70000,
                0,
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
        returned = torch.ops.aten._philox_uniform_flat_slice_(
            actual,
            8,
            [8, 4, 7, 1, 2],
            [0, 2, 3, 3, 1],
            [-1, 2, -1, 3, 1],
            [1, 1, 0, 1, 1],
            -0.2,
            0.3,
            generator=actual_generator,
        )

        self.assertIs(returned, actual)
        self.assertEqual(actual, expected_dense[[4, 5, 1, 2, 3, 2]], rtol=0, atol=0)
        self.assertEqual(actual_generator.get_state(), expected_state)

    def test_meta_validation(self):
        result = torch.empty(6, device="meta")
        returned = torch.ops.aten._philox_normal_flat_slice_(
            result, 17, [1, 10], [2, 2], [4, 3], [2, 1]
        )
        self.assertIs(returned, result)

        with self.assertRaisesRegex(RuntimeError, "normal expects std >= 0.0"):
            torch.ops.aten._philox_normal_flat_slice_(
                torch.empty(1, device="meta"),
                1,
                [0],
                [1],
                [1],
                [1],
                0,
                -1,
            )
        with self.assertRaisesRegex(RuntimeError, "found from=1.*> to=0"):
            torch.ops.aten._philox_uniform_flat_slice_(
                torch.empty(1, device="meta"),
                1,
                [0],
                [1],
                [1],
                [1],
                1,
                0,
            )
        legacy_result = torch.empty(6, device="meta")
        returned = torch.ops.aten._philox_uniform_flat_slice_(
            legacy_result,
            8,
            [8, 4, 7, 1, 2],
            [0, 2, 3, 3, 1],
            [-1, 2, -1, 3, 1],
            [1, 1, 0, 1, 1],
        )
        self.assertIs(returned, legacy_result)

        key = torch.empty(2, dtype=torch.uint64, device="meta")
        with self.assertRaisesRegex(RuntimeError, "block_size must be positive"):
            torch.ops.aten._philox_uniform_indexed_(
                torch.empty(0, device="meta"), key, 8, [0], [0], [-1], [1]
            )
        with self.assertRaisesRegex(RuntimeError, "ordered and non-overlapping"):
            torch.ops.aten._philox_uniform_indexed_(
                torch.empty(4, device="meta"),
                key,
                8,
                [0, 1],
                [2, 2],
                [2, 2],
                [1, 1],
            )


if __name__ == "__main__":
    run_tests()
