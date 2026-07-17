# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import unittest
from functools import partial

import torch
from torch.distributed import stateful_rng_mode, StatefulRNGTensor
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

        torch.manual_seed(123)
        actual = torch.empty(global_indices.numel(), device=device)
        self._set_rng_metadata(actual, expected.numel(), index_blocks)
        with stateful_rng_mode():
            actual.normal_(0.1, 0.02)

        self.assertEqual(actual, expected[global_indices], rtol=0, atol=0)
        self.assertEqual(torch.cuda.get_rng_state(device), expected_state)

    @unittest.skipIf(not TEST_CUDA, "CUDA is required")
    def test_explicit_generator_matches_dense(self):
        device = torch.device("cuda")
        expected_generator = torch.Generator(device=device).manual_seed(123)
        expected = torch.empty((5, 7), device=device)
        expected.uniform_(-0.2, 0.3, generator=expected_generator)

        actual_generator = torch.Generator(device=device).manual_seed(123)
        actual = torch.empty((5, 3), device=device)
        self._set_rng_metadata(actual, expected.numel(), ((4, 3, 7, 5),))
        with stateful_rng_mode():
            actual.uniform_(-0.2, 0.3, generator=actual_generator)

        self.assertEqual(actual, expected[:, 4:], rtol=0, atol=0)
        self.assertEqual(actual_generator.get_state(), expected_generator.get_state())

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


if __name__ == "__main__":
    run_tests()
