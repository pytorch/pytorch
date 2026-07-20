# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import unittest
from functools import partial
from typing import Any, cast

import torch
from torch._library.utils import fill_defaults
from torch.distributed import StatefulRNGTensor
from torch.testing._internal.common_utils import run_tests, TEST_CUDA, TestCase
from torch.utils._python_dispatch import TorchDispatchMode


aten = torch.ops.aten


def _validate_normal_std(op_args: list[object]) -> None:
    std = cast(float, op_args[1])
    torch._check(
        std >= 0.0,
        lambda: f"normal expects std >= 0.0, but found std {std}",
    )


def _run_stateful_rng_op(
    tensor: torch.Tensor,
    global_numel: int,
    index_blocks: tuple[tuple[int, int, int, int], ...],
    flat_slice_op_call: torch._ops.OpOverload,
    generator: torch.Generator | None,
    *op_args: object,
) -> torch.Tensor:
    if not tensor.is_contiguous():
        raise RuntimeError(
            f"{flat_slice_op_call}: expected a contiguous local tensor, "
            f"got stride {tensor.stride()}"
        )
    return flat_slice_op_call(
        tensor,
        global_numel,
        [start_index for start_index, _, _, _ in index_blocks],
        [block_size for _, block_size, _, _ in index_blocks],
        [block_stride for _, _, block_stride, _ in index_blocks],
        [num_blocks for _, _, _, num_blocks in index_blocks],
        *op_args,
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
        if func is not aten.normal_.default:
            return func(*args, **kwargs)

        filled_args, filled_kwargs = fill_defaults(func._schema, args, kwargs)
        tensor_arg = filled_args[0]
        if not isinstance(tensor_arg, torch.Tensor):
            return func(*args, **kwargs)
        if tensor_arg.is_meta or tensor_arg.device.type != "cuda":
            return func(*args, **kwargs)
        if not isinstance(tensor_arg, StatefulRNGTensor):
            return func(*args, **kwargs)
        rng_metadata = cast(StatefulRNGTensor, tensor_arg)

        tensor, *op_args = filled_args
        generator = filled_kwargs.pop("generator")
        if filled_kwargs:
            raise AssertionError
        if not isinstance(tensor, torch.Tensor):
            raise AssertionError
        if not (generator is None or isinstance(generator, torch.Generator)):
            raise AssertionError

        _validate_normal_std(op_args)
        _run_stateful_rng_op(
            tensor,
            rng_metadata.rng_global_numel,
            rng_metadata.rng_index_blocks,
            aten._philox_normal_flat_slice_.default,
            generator,
            *op_args,
        )
        return tensor


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
            "trunc_normal": partial(
                torch.nn.init.trunc_normal_,
                mean=0.0,
                std=0.02,
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
                    with _StatefulRNGMode():
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

        torch.manual_seed(123)
        actual = torch.empty(global_indices.numel(), device=device)
        self._set_rng_metadata(actual, expected.numel(), index_blocks)
        with _StatefulRNGMode():
            actual.normal_(0.1, 0.02)

        self.assertEqual(actual, expected[global_indices], rtol=0, atol=0)
        self.assertEqual(torch.cuda.get_rng_state(device), expected_state)

    @unittest.skipIf(not TEST_CUDA, "CUDA is required")
    def test_explicit_generator_matches_dense(self):
        device = torch.device("cuda")
        expected_generator = torch.Generator(device=device).manual_seed(123)
        expected = torch.empty((5, 7), device=device)
        expected.normal_(0.1, 0.02, generator=expected_generator)

        actual_generator = torch.Generator(device=device).manual_seed(123)
        actual = torch.empty((5, 3), device=device)
        self._set_rng_metadata(actual, expected.numel(), ((4, 3, 7, 5),))
        with _StatefulRNGMode():
            actual.normal_(0.1, 0.02, generator=actual_generator)

        self.assertEqual(actual, expected[:, 4:], rtol=0, atol=0)
        self.assertEqual(actual_generator.get_state(), expected_generator.get_state())

    @unittest.skipIf(not TEST_CUDA, "CUDA is required")
    def test_invalid_std_does_not_advance_generator(self):
        device = torch.device("cuda")
        for generator_kind in ("default", "explicit"):
            with self.subTest(generator=generator_kind):
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
                with self.assertRaisesRegex(RuntimeError, "std >= 0.0"):
                    with _StatefulRNGMode():
                        actual.normal_(0.0, -1.0, generator=generator)

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
        torch.empty(7, device=device).normal_()
        expected_state = torch.cuda.get_rng_state(device)

        torch.manual_seed(123)
        actual = torch.empty(0, device=device)
        self._set_rng_metadata(actual, 7, ())
        with _StatefulRNGMode():
            actual.normal_()

        self.assertEqual(torch.cuda.get_rng_state(device), expected_state)


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
