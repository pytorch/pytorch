# Owner(s): ["module: inductor"]

import math
import unittest
from unittest import mock

import torch
from torch import fx
from torch._dynamo.utils import counters
from torch._inductor import config
from torch._inductor.decomposition import select_decomp_table
from torch._inductor.test_case import run_tests, TestCase
from torch.fx.experimental.proxy_tensor import make_fx


@unittest.skipIf(not torch.distributed.is_available(), "requires distributed support")
class TestDecomposeShardDimAllToAll(TestCase):
    def _reference_shard_dim_alltoall(
        self,
        inputs: list[torch.Tensor],
        *,
        gather_dim: int,
        shard_dim: int,
    ) -> list[torch.Tensor]:
        group_size = len(inputs)
        ndim = inputs[0].dim()
        gathered = torch.cat(inputs, dim=gather_dim % ndim)
        return list(torch.chunk(gathered, group_size, dim=shard_dim % ndim))

    def _run_decomp_locally(
        self,
        inputs: list[torch.Tensor],
        *,
        gather_dim: int,
        shard_dim: int,
    ) -> list[torch.Tensor]:
        group_size = len(inputs)
        ndim = inputs[0].dim()
        gather_dim = gather_dim % ndim
        shard_dim = shard_dim % ndim
        input_shape = list(inputs[0].shape)
        local_shard_dim = input_shape[shard_dim] // group_size

        pre_view_shape = list(input_shape)
        pre_view_shape[shard_dim] = local_shard_dim
        pre_view_shape.insert(shard_dim, group_size)

        post_view_shape = list(input_shape)
        post_view_shape[shard_dim] = local_shard_dim
        post_view_shape[gather_dim] *= group_size

        pre_alltoall = [
            inp.view(pre_view_shape)
            .movedim(shard_dim, 0)
            .clone(memory_format=torch.contiguous_format)
            for inp in inputs
        ]
        collective_outputs = [
            torch.cat([pre.narrow(0, rank, 1) for pre in pre_alltoall], dim=0)
            for rank in range(group_size)
        ]

        current_rank = 0

        def all_to_all_single(input, output_split_sizes, input_split_sizes, group_name):
            self.assertEqual(group_name, "test_pg")
            self.assertEqual(output_split_sizes, [1] * group_size)
            self.assertEqual(input_split_sizes, [1] * group_size)
            self.assertEqual(input, pre_alltoall[current_rank])
            return collective_outputs[current_rank]

        def wait_tensor(input):
            return input

        outputs = []
        get_group_size = self._group_size_mock(group_size)
        with config.patch({"decompose_shard_dim_alltoall": True}):
            with get_group_size:
                with (
                    mock.patch.object(
                        torch.ops._c10d_functional.all_to_all_single,
                        "default",
                        side_effect=all_to_all_single,
                    ),
                    mock.patch.object(
                        torch.ops._c10d_functional.wait_tensor,
                        "default",
                        side_effect=wait_tensor,
                    ),
                ):
                    for rank, inp in enumerate(inputs):
                        current_rank = rank
                        outputs.append(
                            self._decomp()(inp, gather_dim, shard_dim, "test_pg")
                        )

        return outputs

    def _decomp(self):
        return select_decomp_table()[torch.ops._dtensor.shard_dim_alltoall.default]

    def _group_size_mock(self, group_size: int):
        return mock.patch(
            "torch.distributed.distributed_c10d._get_group_size_by_name",
            return_value=group_size,
        )

    def _trace_decomp(
        self,
        *,
        shape: tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        gather_dim: int = 0,
        shard_dim: int = 1,
        group_size: int = 4,
    ) -> fx.GraphModule:
        inp = torch.empty(shape, dtype=dtype)

        def fn(x):
            return torch.ops._dtensor.shard_dim_alltoall.default(
                x, gather_dim, shard_dim, "test_pg"
            )

        get_group_size = self._group_size_mock(group_size)
        with config.patch({"decompose_shard_dim_alltoall": True}):
            with get_group_size:
                return make_fx(
                    fn,
                    decomposition_table=select_decomp_table(),
                    tracing_mode="fake",
                )(inp)

    def _call_decomp(
        self,
        *,
        shape: tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        gather_dim: int = 0,
        shard_dim: int = 1,
        group_size: int = 4,
        enabled: bool = True,
    ):
        inp = torch.empty(shape, dtype=dtype)
        get_group_size = self._group_size_mock(group_size)
        with config.patch({"decompose_shard_dim_alltoall": enabled}):
            with get_group_size:
                return self._decomp()(inp, gather_dim, shard_dim, "test_pg")

    def test_decomposition_exposes_c10d_collective(self) -> None:
        counters.clear()
        gm = self._trace_decomp(shape=(8, 5, 6), gather_dim=2, shard_dim=0)
        gm.graph.lint()

        targets = [node.target for node in gm.graph.nodes]
        self.assertNotIn(torch.ops._dtensor.shard_dim_alltoall.default, targets)
        self.assertIn(torch.ops._c10d_functional.all_to_all_single.default, targets)
        self.assertIn(torch.ops._c10d_functional.wait_tensor.default, targets)

        alltoall_node = next(
            node
            for node in gm.graph.nodes
            if node.target is torch.ops._c10d_functional.all_to_all_single.default
        )
        self.assertEqual(alltoall_node.args[1], [1, 1, 1, 1])
        self.assertEqual(alltoall_node.args[2], [1, 1, 1, 1])
        self.assertEqual(alltoall_node.args[3], "test_pg")
        output_node = next(node for node in gm.graph.nodes if node.op == "output")
        self.assertEqual(output_node.args[0].meta["val"].shape, (2, 5, 24))
        self.assertEqual(counters["inductor"]["decompose_shard_dim_alltoall"], 1)

    def test_decomposition_matches_original_api_semantics_locally(self) -> None:
        cases = [
            (4, (5, 8), 0, 1),
            (4, (8, 5, 6), 2, 0),
            (4, (3, 8, 4), -1, 1),
            (2, (6, 4, 5), 1, 0),
        ]

        for group_size, shape, gather_dim, shard_dim in cases:
            with self.subTest(
                group_size=group_size,
                shape=shape,
                gather_dim=gather_dim,
                shard_dim=shard_dim,
            ):
                inputs = torch.arange(
                    group_size * math.prod(shape),
                    dtype=torch.float32,
                ).reshape(group_size, *shape)
                per_rank_inputs = list(inputs.unbind(0))

                expected = self._reference_shard_dim_alltoall(
                    per_rank_inputs,
                    gather_dim=gather_dim,
                    shard_dim=shard_dim,
                )
                actual = self._run_decomp_locally(
                    per_rank_inputs,
                    gather_dim=gather_dim,
                    shard_dim=shard_dim,
                )

                self.assertEqual(actual, expected)
                self.assertTrue(all(out.is_contiguous() for out in actual))

    def test_backward_formula_matches_original_api_semantics_locally(self) -> None:
        group_size = 4
        input_shape = (8, 5, 6)
        gather_dim = 2
        shard_dim = 0
        inputs = torch.arange(
            group_size * math.prod(input_shape),
            dtype=torch.float32,
        ).reshape(group_size, *input_shape)
        forward_outputs = self._reference_shard_dim_alltoall(
            list(inputs.unbind(0)),
            gather_dim=gather_dim,
            shard_dim=shard_dim,
        )

        grad_outputs = [
            torch.arange(out.numel(), dtype=torch.float32).reshape(out.shape) + rank
            for rank, out in enumerate(forward_outputs)
        ]

        expected_grads = self._reference_shard_dim_alltoall(
            grad_outputs,
            gather_dim=shard_dim,
            shard_dim=gather_dim,
        )
        actual_grads = self._run_decomp_locally(
            grad_outputs,
            gather_dim=shard_dim,
            shard_dim=gather_dim,
        )
        self.assertEqual(actual_grads, expected_grads)

    def test_falls_back_for_unsupported_cases(self) -> None:
        cases = [
            {"shape": (5, 8), "enabled": False},
            {"shape": (5, 7)},
            {"shape": (5, 8), "dtype": torch.complex64},
            {"shape": (5, 8), "gather_dim": 1, "shard_dim": 1},
        ]

        for case in cases:
            with self.subTest(**case):
                counters.clear()
                result = self._call_decomp(**case)
                self.assertIs(result, NotImplemented)
                self.assertEqual(
                    counters["inductor"]["decompose_shard_dim_alltoall"], 0
                )

    def test_shard_dim_alltoall_registered_autograd_backward(self) -> None:
        import torch.distributed.tensor._collective_utils

        calls = []

        with torch.library._scoped_library("_dtensor", "IMPL") as lib:

            def impl(input, gather_dim: int, shard_dim: int, group_name):
                calls.append((input, gather_dim, shard_dim, group_name))
                if len(calls) == 1:
                    return input.detach().clone()
                return torch.full_like(input, 7.0)

            lib.impl("shard_dim_alltoall", impl, "CPU")

            x = torch.randn(2, 3, 4, requires_grad=True)
            y = torch.ops._dtensor.shard_dim_alltoall.default(x, 2, 0, "test_pg")
            self.assertTrue(y.requires_grad)
            y.sum().backward()

        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][1:], (2, 0, "test_pg"))
        self.assertEqual(calls[1][1:], (0, 2, "test_pg"))
        self.assertFalse(calls[1][0].requires_grad)
        self.assertEqual(x.grad, torch.full_like(x, 7.0))


if __name__ == "__main__":
    run_tests()
