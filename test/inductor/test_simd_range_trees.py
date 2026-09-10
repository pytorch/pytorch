# Owner(s): ["module: inductor"]

import sys
import typing
import unittest
from types import SimpleNamespace

import sympy

from torch._dynamo.source import ConstantSource
from torch._inductor.codegen.simd import DerivedIterationRangesRoot, IterationRangesRoot
from torch._inductor.codegen.simd_kernel_features import SIMDKernelFeatures
from torch._inductor.codegen.triton import IndexingOptions, TritonKernel, TritonSymbols
from torch._inductor.virtualized import V
from torch.fx.experimental.symbolic_shapes import DimDynamic
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.inductor_utils import MockGraphHandler
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.functions import CeilDiv, FloorDiv


try:
    import triton  # noqa: F401
except ImportError:
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("requires triton")  # noqa: B904


class TestSIMDRangeTrees(TestCase):
    def _make_graph(self):
        return MockGraphHandler()

    def _make_kernel(self, *, persistent: bool):
        features = SIMDKernelFeatures([], sympy.Integer(4), sympy.Integer(512))
        return TritonKernel(
            {"x": sympy.Integer(4), "r0_": sympy.Integer(512)},
            features=features,
            override_persistent_reduction=persistent,
            override_cooperative_reduction=False,
        )

    def _make_parent_reduction_root(self):
        return IterationRangesRoot(
            "r0_index",
            sympy.Integer(512),
            "r0_",
            0,
            typing.cast(typing.Any, object()),
            is_loop=True,
            tensor_dim=1,
            grid_dim=None,
            has_zdim=False,
        )

    def _make_derived_root(self, r_tree, *, group_size=sympy.Integer(128)):
        reduced_block = FloorDiv(r_tree.block_size(), group_size)
        reduced_numel = FloorDiv(r_tree.numel, group_size)
        return DerivedIterationRangesRoot(
            r_tree,
            numel=reduced_numel,
            block_size=reduced_block,
            block_offset=FloorDiv(r_tree.block_offset(), group_size),
            named_constants=(
                (sympy.Symbol("nested_R0_GROUP_SIZE"), group_size, True),
                (sympy.Symbol("nested_R0_REDUCED_BLOCK"), reduced_block, True),
                (sympy.Symbol("nested_R0_REDUCED_NUMEL"), reduced_numel, False),
            ),
        )

    def test_derived_root_geometry(self):
        r_tree = self._make_parent_reduction_root()
        derived = self._make_derived_root(r_tree)

        self.assertEqual(derived.is_loop, r_tree.is_loop)
        self.assertEqual(derived.prefix, r_tree.prefix)
        self.assertEqual(derived.numel, FloorDiv(r_tree.numel, 128))
        self.assertEqual(derived.block_size(), FloorDiv(r_tree.block_size(), 128))
        self.assertEqual(derived.block_offset(), FloorDiv(r_tree.block_offset(), 128))
        self.assertTrue(derived.owns_mask(derived.mask_name()))
        self.assertFalse(derived.owns_mask(r_tree.mask_name()))
        self.assertNotEqual(derived.mask_name(), r_tree.mask_name())
        self.assertFalse(derived.supports_constant_mask())

    def test_derived_root_uses_active_range_tree_geometry(self):
        graph = self._make_graph()
        with V.set_graph_handler(graph):
            kernel = self._make_kernel(persistent=False)
            x_tree, r_tree = kernel.range_trees
            derived = self._make_derived_root(r_tree)

            self.assertFalse(kernel._has_constant_mask(derived))

            with V.set_kernel_handler(kernel):
                x_entry = x_tree.full_range()
                derived_entry = derived.full_range()
                index = x_entry.symbol() * derived.numel + derived_entry.symbol()
                self.assertTrue(kernel.is_broadcasted(index))

                indexing = IndexingOptions(
                    index_str="",
                    mask_vars=OrderedSet([derived.mask_name()]),
                    expand_str=None,
                    _has_rindex=True,
                    index=index,
                    expand_shape=None,
                )
                self.assertFalse(indexing.has_rmask())

                with kernel.use_range_trees([x_tree, derived]):
                    self.assertFalse(kernel.is_broadcasted(index))
                    self.assertEqual(
                        TritonSymbols.get_block_shape(index),
                        ("XBLOCK", "(R0_BLOCK//128)"),
                    )
                    self.assertTrue(indexing.has_rmask())

                self.assertEqual(kernel.range_trees, [x_tree, r_tree])
                self.assertFalse(indexing.has_rmask())

    def test_derived_reduction_extent_reuses_r_numel(self):
        graph = self._make_graph()
        with V.set_graph_handler(graph):
            reduction_extent = graph.sizevars.shape_env.create_symbol(
                512,
                source=ConstantSource("__test_reduction_numel"),
                dynamic_dim=DimDynamic.DYNAMIC,
                constraint_dim=None,
            )
            features = SIMDKernelFeatures([], sympy.Integer(4), reduction_extent)
            kernel = TritonKernel(
                {"x": sympy.Integer(4), "r0_": reduction_extent},
                features=features,
                override_persistent_reduction=False,
                override_cooperative_reduction=False,
            )
            x_tree, r_tree = kernel.range_trees
            derived = self._make_derived_root(r_tree, group_size=sympy.Integer(2))
            r_numel_symbol = sympy.Symbol("r0_numel", integer=True, nonnegative=True)

            index_before = reduction_extent
            expected_index = r_numel_symbol
            derived_index_before = derived.numel
            expected_derived_index = FloorDiv(r_numel_symbol, 2)

            kernel.finalize_indexing([index_before])
            self.assertEqual(
                kernel._r_numel_reuse_replacements,
                {r_numel_symbol: reduction_extent},
            )
            self.assertEqual(
                kernel._r_numel_reuse_eliminated_symbols,
                OrderedSet([reduction_extent]),
            )
            with kernel.use_range_trees([x_tree, derived]):
                self.assertEqual(
                    kernel._replace_reduction_numel_in_index(index_before),
                    expected_index,
                )
                self.assertEqual(
                    kernel._replace_reduction_numel_in_index(derived_index_before),
                    expected_derived_index,
                )

    def test_reduction_numel_reuse_supports_joint_elimination(self):
        graph = self._make_graph()
        with V.set_graph_handler(graph):
            source = graph.sizevars.shape_env.create_symbol(
                109,
                source=ConstantSource("__test_joint_source"),
                dynamic_dim=DimDynamic.DYNAMIC,
                constraint_dim=None,
            )
            r0_extent = CeilDiv(source, 2)
            r1_extent = CeilDiv(source, 3)
            kernel = TritonKernel(
                {
                    "x": sympy.Integer(1),
                    "r0_": r0_extent,
                    "r1_": r1_extent,
                },
                features=SIMDKernelFeatures(
                    [], sympy.Integer(1), r0_extent * r1_extent
                ),
                override_persistent_reduction=False,
                override_cooperative_reduction=False,
            )

            index_before = r0_extent + r1_extent
            expected_index = sympy.Symbol(
                "r0_numel", integer=True, nonnegative=True
            ) + sympy.Symbol("r1_numel", integer=True, nonnegative=True)

            kernel.finalize_indexing([r0_extent, r1_extent])
            self.assertEqual(
                kernel._r_numel_reuse_eliminated_symbols,
                OrderedSet([source]),
            )
            self.assertEqual(
                kernel._replace_reduction_numel_in_index(index_before),
                expected_index,
            )

    def test_reduction_numel_reuse_requires_dead_size_argument(self):
        graph = self._make_graph()
        with V.set_graph_handler(graph):
            dynamic_size = graph.sizevars.shape_env.create_symbol(
                1023,
                source=ConstantSource("__test_dynamic_size"),
                dynamic_dim=DimDynamic.DYNAMIC,
                constraint_dim=None,
            )
            reduction_extent = CeilDiv(dynamic_size, 2)
            features = SIMDKernelFeatures([], sympy.Integer(4), reduction_extent)
            kernel = TritonKernel(
                {"x": sympy.Integer(4), "r0_": reduction_extent},
                features=features,
                override_persistent_reduction=False,
                override_cooperative_reduction=False,
            )
            r_numel_symbol = sympy.Symbol("r0_numel", integer=True, nonnegative=True)
            index_before = reduction_extent
            expected_index = r_numel_symbol

            kernel.finalize_indexing([index_before])
            self.assertEqual(
                kernel._replace_reduction_numel_in_index(index_before),
                expected_index,
            )

            kernel.finalize_indexing([index_before, dynamic_size])
            self.assertEqual(
                kernel._replace_reduction_numel_in_index(index_before),
                index_before,
            )

            kernel.range_tree_nodes[sympy.Symbol("__test_range")] = SimpleNamespace(
                expr=dynamic_size
            )
            kernel.finalize_indexing([index_before])
            self.assertEqual(
                kernel._replace_reduction_numel_in_index(index_before),
                index_before,
            )

    def test_reduction_numel_reuse_skips_staged_indexing_schedule(self):
        graph = self._make_graph()
        with V.set_graph_handler(graph):
            dynamic_size = graph.sizevars.shape_env.create_symbol(
                1023,
                source=ConstantSource("__test_staged_dynamic_size"),
                dynamic_dim=DimDynamic.DYNAMIC,
                constraint_dim=None,
            )
            reduction_extent = CeilDiv(dynamic_size, 2)
            features = SIMDKernelFeatures(
                [],
                sympy.Integer(4),
                reduction_extent,
                indexing_node_schedule=[],
            )
            kernel = TritonKernel(
                {"x": sympy.Integer(4), "r0_": reduction_extent},
                features=features,
                override_persistent_reduction=False,
                override_cooperative_reduction=False,
            )

            index_before = reduction_extent

            kernel.finalize_indexing([index_before])
            self.assertEqual(
                kernel._replace_reduction_numel_in_index(index_before),
                index_before,
            )

    def test_use_range_trees_clears_simplify_indexing_cache(self):
        graph = self._make_graph()
        with V.set_graph_handler(graph):
            kernel = self._make_kernel(persistent=False)
            x_tree, r_tree = kernel.range_trees
            derived = self._make_derived_root(r_tree)

            with V.set_kernel_handler(kernel):
                expr = x_tree.full_range().symbol() + r_tree.full_range().symbol()
                kernel.simplify_indexing(expr)
                self.assertGreater(kernel.simplify_indexing.cache_info().currsize, 0)

                with kernel.use_range_trees([x_tree, derived]):
                    self.assertEqual(kernel.simplify_indexing.cache_info().currsize, 0)
                    kernel.simplify_indexing(expr)
                    self.assertGreater(
                        kernel.simplify_indexing.cache_info().currsize, 0
                    )

                self.assertEqual(kernel.simplify_indexing.cache_info().currsize, 0)
                self.assertEqual(kernel.range_trees, [x_tree, r_tree])


if __name__ == "__main__":
    run_tests()
