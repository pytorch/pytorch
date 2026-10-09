# Owner(s): ["oncall: distributed"]
import copy
import itertools
from typing import cast
from unittest.mock import patch

import sympy

import torch
from torch._subclasses import FakeTensorMode
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor._collective_utils import (
    pad_tensor,
    pad_tensor_to_size,
    unpad_tensor,
)
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor._ops.utils import is_tensor_shardable
from torch.distributed.tensor.placement_types import (
    _hint_proves_even_shard,
    _StridedShard,
    _StridedShardOffsetMode,
    Partial,
    Replicate,
    Shard,
)
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import (
    free_symbols,
    free_unbacked_symbols,
    GuardOnDataDependentSymNode,
    optimization_hint,
    ShapeEnv,
)
from torch.testing._internal.common_utils import run_tests, TestCase


class _ReduceScatterMesh:
    def __init__(self, rank, num_chunks):
        self.rank = rank
        self.num_chunks = num_chunks

    def get_coordinate(self):
        return (self.rank,)

    def size(self, mesh_dim):
        assert mesh_dim == 0
        return self.num_chunks

    def _sym_get_coordinate(self, mesh_dim):
        assert mesh_dim == 0
        return self.rank


# Basic functionality test for Placement types.
class PlacementTypesTestCase(TestCase):
    @staticmethod
    def _mock_reduce_scatter(mesh_rank, num_chunks):
        def reduce_scatter(tensor, reduce_op, scatter_dim, group):
            del reduce_op, group
            chunk_size = tensor.size(scatter_dim) // num_chunks
            return tensor.narrow(
                scatter_dim, mesh_rank * chunk_size, chunk_size
            ).clone()

        return reduce_scatter

    def test_zero_pad_and_unpad_eager_identity(self):
        tensor = torch.randn(2, 8)

        self.assertIs(pad_tensor(tensor, 0, 0), tensor)
        self.assertIs(unpad_tensor(tensor, 0, 0), tensor)

    def test_zero_pad_and_unpad_are_recorded_by_make_fx(self):
        def pad_and_unpad(tensor):
            return unpad_tensor(pad_tensor(tensor, 0, 0), 0, 0)

        graph = make_fx(pad_and_unpad)(torch.randn(2, 8)).graph
        call_targets = [
            node.target for node in graph.nodes if node.op == "call_function"
        ]

        self.assertEqual(
            call_targets,
            [
                torch.ops.aten.constant_pad_nd.default,
                torch.ops.aten.slice.Tensor,
            ],
        )

    def test_zero_pad_and_unpad_match_positive_size_torch_compile_graphs(self):
        def capture_ops(fn):
            graphs = []

            def backend(graph_module, example_inputs):
                graphs.append(graph_module)
                return graph_module.forward

            torch.compile(fn, backend=backend, fullgraph=True)(torch.randn(2, 8))
            self.assertEqual(len(graphs), 1)
            return [
                (node.op, node.target)
                for node in graphs[0].graph.nodes
                if node.op not in ("placeholder", "output")
            ]

        def zero_pad(tensor):
            return pad_tensor(tensor, 0, 0)

        def positive_pad(tensor):
            return pad_tensor(tensor, 0, 1)

        def zero_unpad(tensor):
            return unpad_tensor(tensor, 0, 0)

        def positive_unpad(tensor):
            return unpad_tensor(tensor, 0, 1)

        pad_ops = capture_ops(zero_pad)
        unpad_ops = capture_ops(zero_unpad)
        self.assertEqual(pad_ops, capture_ops(positive_pad))
        self.assertEqual(unpad_ops, capture_ops(positive_unpad))
        self.assertEqual(len(pad_ops), 1)
        self.assertEqual(pad_ops[0][0], "call_function")
        self.assertEqual(pad_ops[0][1].__name__, "pad")
        self.assertEqual(unpad_ops, [("call_method", "narrow")])

    def test_type_identification(self):
        shard = Shard(3)
        strided_shard = _StridedShard(dim=3, split_factor=7)
        partial_sum = Partial("sum")
        partial_max = Partial("max")
        replicate = Replicate()

        ident_tests = (
            (shard, True, False, False),
            (strided_shard, False, False, False),
            (partial_sum, False, True, False),
            (partial_max, False, True, False),
            (replicate, False, False, True),
        )
        for do_deepcopy in (False, True):
            for placement, is_shard, is_partial, is_replicate in ident_tests:
                if do_deepcopy:
                    placement = copy.deepcopy(placement)
                self.assertEqual(placement.is_shard(), is_shard)
                self.assertEqual(placement.is_partial(), is_partial)
                self.assertEqual(placement.is_replicate(), is_replicate)

    def test_equality(self):
        equivalence_classes = (
            (Shard(3),),
            (Shard(4),),
            (_StridedShard(dim=3, split_factor=1),),
            (_StridedShard(dim=3, split_factor=2),),
            (_StridedShard(dim=4, split_factor=9),),
            (Replicate(),),
            (Partial("sum"),),
            (Partial("max"),),
        )
        for eq_class in equivalence_classes:
            # Each item in the equivalence class should be equal to every other item in
            # its class.
            for lhs, rhs in itertools.product(eq_class, eq_class):
                self.assertEqual(lhs, rhs)

            # Each item in the equivalence class should not be equal to any item in any
            # other class.
            for other_class in equivalence_classes:
                if other_class is eq_class:
                    continue
                for lhs, rhs in itertools.product(eq_class, other_class):
                    self.assertNotEqual(lhs, rhs)

    def test_strided_shard_kwonly_argument(self):
        with self.assertRaises(TypeError):
            _StridedShard(3, 4)
        _StridedShard(3, split_factor=4)

    def test_select_split_tensor_matches_split_tensor(self):
        """
        Test that _select_split_tensor produces the same result as indexing
        into _split_tensor. This validates that any alternative implementation
        (e.g., the narrow-based SymInt path) matches the canonical _split_tensor.
        """
        # Test various tensor sizes and num_chunks combinations
        test_cases = [
            # (dim_size, num_chunks) - covers even splits, uneven splits, edge cases
            (16, 4),  # even split
            (17, 4),  # uneven split, last chunk smaller
            (15, 4),  # uneven split
            (7, 4),  # fewer elements than chunks would like
            (5, 4),  # non-last rank is already short; last rank empty
            (3, 4),  # very few elements
            (1, 4),  # single element
            (8, 1),  # single chunk
            (8, 8),  # one element per chunk
            (8, 16),  # more chunks than elements
        ]

        for dim in [0, 1]:
            shard = Shard(dim)
            for dim_size, num_chunks in test_cases:
                # Create a tensor with distinct values for easy debugging
                if dim == 0:
                    tensor = torch.arange(dim_size * 4).reshape(dim_size, 4)
                else:
                    tensor = torch.arange(4 * dim_size).reshape(4, dim_size)

                # Get ground truth from _split_tensor
                shards, _ = shard._split_tensor(
                    tensor, num_chunks, with_padding=False, contiguous=False
                )

                # Compare _select_split_tensor against _split_tensor for each index
                for idx in range(num_chunks):
                    selected = shard._select_split_tensor(
                        tensor,
                        num_chunks,
                        idx,
                        with_padding=False,
                        contiguous=False,
                        clone=False,
                    )
                    self.assertEqual(
                        selected,
                        shards[idx],
                        msg=lambda msg: f"{msg}\nMismatch for dim={dim}, dim_size={dim_size}, "
                        f"num_chunks={num_chunks}, idx={idx}",
                    )

    def test_select_split_tensor_symint_with_padding_raises(self):
        """
        Test that _select_split_tensor raises GuardOnDataDependentSymNode when
        called with a SymInt index and with_padding=True.

        This is expected because with_padding=True requires indexing into a
        Python list with the SymInt, which is not supported.
        """
        from torch.fx.experimental.symbolic_shapes import (
            GuardOnDataDependentSymNode,
            ShapeEnv,
        )

        shape_env = ShapeEnv()
        symint_index = shape_env.create_unbacked_symint()

        shard = Shard(0)
        tensor = torch.arange(16).reshape(4, 4)

        with self.assertRaises(GuardOnDataDependentSymNode):
            shard._select_split_tensor(
                tensor,
                num_chunks=4,
                index=symint_index,
                with_padding=True,
            )

    def test_select_split_tensor_symint_matches_split_tensor(self):
        """The narrow/SymInt path must match _split_tensor, including empty trailing shards."""
        from torch._dynamo.source import ConstantSource
        from torch.fx.experimental.symbolic_shapes import SymNode

        test_cases = [
            (16, 4),  # even
            (17, 4),  # only last shard short
            (5, 4),  # non-last already short, last empty
            (1, 4),  # single element, several empty ranks
            (8, 16),  # more chunks than elements
        ]
        for dim in [0, 1]:
            shard = Shard(dim)
            for dim_size, num_chunks in test_cases:
                if dim == 0:
                    tensor = torch.arange(dim_size * 4).reshape(dim_size, 4)
                else:
                    tensor = torch.arange(4 * dim_size).reshape(4, dim_size)
                shards, _ = shard._split_tensor(
                    tensor, num_chunks, with_padding=False, contiguous=False
                )
                shape_env = ShapeEnv()
                for rank in range(num_chunks):
                    symbol = shape_env.create_symbol(
                        rank, source=ConstantSource(f"rank_{dim}_{dim_size}_{rank}")
                    )
                    idx = torch.SymInt(SymNode(symbol, shape_env, int, hint=rank))
                    self.assertIsInstance(idx, torch.SymInt)
                    selected = shard._select_split_tensor(
                        tensor,
                        num_chunks,
                        idx,
                        with_padding=False,
                        contiguous=False,
                        clone=False,
                    )
                    self.assertEqual(
                        selected,
                        shards[rank],
                        msg=lambda msg: f"{msg}\nMismatch for dim={dim}, dim_size={dim_size}, "
                        f"num_chunks={num_chunks}, rank={rank}",
                    )

    def test_select_split_tensor_unbacked_symint_uneven_does_not_raise(self):
        """Unbacked rank + dim_size=5 / 4 chunks used to set last_split=-1 and crash in narrow."""
        fake_mode = FakeTensorMode(allow_non_fake_inputs=True, shape_env=ShapeEnv())
        with fake_mode:
            idx = fake_mode.shape_env.create_unbacked_symint()
            tensor = torch.empty(5, 4)
            result = Shard(0)._select_split_tensor(
                tensor,
                num_chunks=4,
                index=idx,
                with_padding=False,
                clone=False,
                contiguous=False,
            )
        self.assertEqual(result.ndim, 2)
        self.assertEqual(result.size(1), 4)
        self.assertIsInstance(result.size(0), torch.SymInt)

    def test_reduce_shard_tensor_unpads_uneven_and_empty_shards(self):
        num_chunks = 4
        for dim in (0, 1):
            shard = Shard(dim)
            for dim_size in (2, 5, 17):
                shape = (dim_size, 3) if dim == 0 else (3, dim_size)
                tensor = torch.arange(torch.Size(shape).numel()).reshape(shape)
                expected_shards, _ = shard._split_tensor(
                    tensor,
                    num_chunks,
                    with_padding=False,
                    contiguous=True,
                )
                for rank, expected in enumerate(expected_shards):
                    mesh = cast(DeviceMesh, _ReduceScatterMesh(rank, num_chunks))
                    reduce_scatter = self._mock_reduce_scatter(rank, num_chunks)
                    with patch(
                        "torch.distributed.tensor.placement_types.funcol.reduce_scatter_single",
                        reduce_scatter,
                    ):
                        actual = shard._reduce_shard_tensor(
                            tensor, mesh, "sum", mesh_dim=0
                        )
                    self.assertEqual(actual, expected)

    def test_reduce_shard_tensor_unbacked_rank_with_padding(self):
        num_chunks = 4
        fake_mode = FakeTensorMode(allow_non_fake_inputs=True, shape_env=ShapeEnv())
        with fake_mode:
            shape_env = fake_mode.shape_env
            assert shape_env is not None
            symbolic_rank = shape_env.create_unbacked_symint()
            shape_env.constrain_symbol_range(symbolic_rank.node.expr, 0, num_chunks - 1)
            mesh = cast(DeviceMesh, _ReduceScatterMesh(symbolic_rank, num_chunks))
            tensor = torch.empty(2, 3)
            reduce_scatter = self._mock_reduce_scatter(0, num_chunks)
            with patch(
                "torch.distributed.tensor.placement_types.funcol.reduce_scatter_single",
                reduce_scatter,
            ):
                output = Shard(0)._reduce_shard_tensor(tensor, mesh, "sum", mesh_dim=0)

        self.assertIsInstance(output.size(0), torch.SymInt)
        self.assertEqual(
            free_unbacked_symbols(output.size(0)), {symbolic_rank.node.expr}
        )
        self.assertEqual(output.size(1), 3)

    def test_hinted_unbacked_even_shard_skips_padding(self):
        fake_mode = FakeTensorMode(allow_non_fake_inputs=True, shape_env=ShapeEnv())
        shard = Shard(0)

        with fake_mode:
            dim_size = fake_mode.shape_env.create_unbacked_symint()
            torch._dynamo.override_optimization_hint(dim_size, 8)
            local_tensor = torch.empty(dim_size // 4, 4)
            full_tensor = torch.empty(dim_size, 4)

            padded = shard._maybe_pad_tensor(local_tensor, dim_size, 4)
            unpadded = shard._maybe_unpad_tensor(full_tensor, dim_size, 4)

        self.assertEqual(padded.shape, local_tensor.shape)
        self.assertEqual(unpadded.shape, full_tensor.shape)

    def test_pad_tensor_to_size_handles_runtime_empty_input(self):
        traced = make_fx(lambda tensor: pad_tensor_to_size(tensor, 0, 1))(
            torch.randn(1, 3)
        )
        padded = traced(torch.empty(0, 3))

        self.assertEqual(padded, torch.zeros(1, 3))
        call_targets = {
            node.target for node in traced.graph.nodes if node.op == "call_function"
        }
        self.assertIn(torch.ops._dtensor._pad_tensor_to_size.default, call_targets)

    def test_view_as_input_handles_runtime_empty_input(self):
        traced = make_fx(torch.ops._dtensor._view_as_input.default)(
            torch.randn(1, 3), torch.randn(1, 3)
        )
        output = traced(torch.empty(0, 3), torch.empty(0, 3))

        self.assertEqual(output.shape, (0, 3))

    def test_hinted_unbacked_even_chunk_preserves_symbolic_shard_size(self):
        fake_mode = FakeTensorMode(allow_non_fake_inputs=True, shape_env=ShapeEnv())

        with fake_mode:
            batch = fake_mode.shape_env.create_unbacked_symint()
            torch._dynamo.override_optimization_hint(batch, 4)
            dim_size = 4096 * batch - 2 * torch.sym_min(1024 * batch, 2048 * batch)
            tensor = torch.empty(dim_size, 256)
            expected_chunk_size = dim_size // 8

            chunks = Shard._custom_chunk(tensor, 8, dim=0)

        self.assertEqual(len(chunks), 8)
        for chunk in chunks:
            self.assertEqual(chunk.size(0).node.expr, expected_chunk_size.node.expr)
            self.assertEqual(
                free_unbacked_symbols(chunk.size(0).node.expr),
                free_unbacked_symbols(batch.node.expr),
            )

    def test_strided_shard_preserves_symbolic_split_factor(self):
        fake_mode = FakeTensorMode(allow_non_fake_inputs=True, shape_env=ShapeEnv())

        with fake_mode:
            split_factor = fake_mode.shape_env.create_unbacked_symint()
            torch._dynamo.override_optimization_hint(split_factor, 4)
            placement = _StridedShard(dim=0, split_factor=split_factor)

        self.assertEqual(placement._split_factor_int(), 4)
        self.assertEqual(
            free_symbols(placement.split_factor), free_symbols(split_factor)
        )
        self.assertEqual(optimization_hint(placement.split_factor), 4)
        self.assertEqual(fake_mode.shape_env.replacements, {})
        deferred_asserts = [
            runtime_assert.expr
            for asserts in fake_mode.shape_env.deferred_runtime_asserts.values()
            for runtime_assert in asserts
        ]
        self.assertIn(sympy.Eq(split_factor.node.expr, 4), deferred_asserts)

    def test_unbacked_even_fallback_hint_requires_override(self):
        fake_mode = FakeTensorMode(allow_non_fake_inputs=True, shape_env=ShapeEnv())

        with fake_mode:
            dim_size = fake_mode.shape_env.create_unbacked_symint()
            self.assertFalse(_hint_proves_even_shard(dim_size, 4))

            torch._dynamo.override_optimization_hint(dim_size, 8)
            self.assertTrue(_hint_proves_even_shard(dim_size, 4))

    def test_hinted_unbacked_shardability_adds_runtime_check(self):
        fake_mode = FakeTensorMode(allow_non_fake_inputs=True, shape_env=ShapeEnv())
        mesh = DeviceMesh("cpu", torch.arange(4), _init_backend=False, _rank=0)
        spec = DTensorSpec(mesh, (Shard(1),))

        with fake_mode:
            dim_size = fake_mode.shape_env.create_unbacked_symint()

            with self.assertRaises(GuardOnDataDependentSymNode):
                is_tensor_shardable((2, dim_size), spec)

            torch._dynamo.override_optimization_hint(dim_size, 16)
            self.assertTrue(is_tensor_shardable((2, dim_size), spec))

    def test_strided_shard_unbacked_local_size_avoids_chunk_guard(self):
        fake_mode = FakeTensorMode(allow_non_fake_inputs=True, shape_env=ShapeEnv())

        with fake_mode:
            dim_size = fake_mode.shape_env.create_unbacked_symint()

        local_size, offsets = _StridedShard(
            dim=1, split_factor=9
        ).local_shard_size_and_offset(
            dim_size,
            num_chunks=2,
            rank=0,
            offset_mode=_StridedShardOffsetMode.ALL,
        )

        self.assertTrue(free_unbacked_symbols(local_size.node.expr))
        self.assertEqual(offsets, [])

    def test_strided_shard_split_tensor_uses_unbacked_safe_chunking(self):
        fake_mode = FakeTensorMode(allow_non_fake_inputs=True, shape_env=ShapeEnv())

        with fake_mode:
            dim_size = fake_mode.shape_env.create_unbacked_symint()
            tensor = torch.empty(1, dim_size)
            shards, pad_sizes = _StridedShard(dim=1, split_factor=9)._split_tensor(
                tensor,
                num_chunks=2,
                with_padding=False,
                contiguous=False,
            )

        self.assertEqual(len(shards), 2)
        self.assertEqual(pad_sizes, [])
        self.assertTrue(free_unbacked_symbols(shards[0].size(1).node.expr))


if __name__ == "__main__":
    run_tests()
