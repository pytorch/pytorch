# Owner(s): ["module: inductor"]

import sys
import unittest

import sympy

import torch
from torch._inductor.codegen.simd import DerivedIterationRangesRoot
from torch._inductor.codegen.simd_kernel_features import SIMDKernelFeatures
from torch._inductor.codegen.triton import IndexingOptions, TritonKernel, TritonSymbols
from torch._inductor.graph import GraphLowering
from torch._inductor.utils import IndentedBuffer
from torch._inductor.virtualized import V
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.functions import FloorDiv


try:
    import triton  # noqa: F401
except ImportError:
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("requires triton")  # noqa: B904


class TestSIMDRangeTrees(TestCase):
    def _make_graph(self):
        class DummyModule(torch.nn.Module):
            def forward(self, x):
                return x * 2

        return GraphLowering(torch.fx.symbolic_trace(DummyModule()))

    def _make_kernel(self, *, persistent: bool):
        features = SIMDKernelFeatures([], sympy.Integer(4), sympy.Integer(512))
        return TritonKernel(
            {"x": sympy.Integer(4), "r0_": sympy.Integer(512)},
            features=features,
            override_persistent_reduction=persistent,
            override_cooperative_reduction=False,
        )

    def _make_derived_root(self, r_tree, *, group_size=sympy.Integer(128)):
        reduced_block = FloorDiv(r_tree.block_size(), group_size)
        reduced_numel = r_tree.numel / group_size
        return DerivedIterationRangesRoot(
            r_tree,
            numel=FloorDiv(r_tree.numel, group_size),
            block_size=reduced_block,
            block_offset=FloorDiv(r_tree.block_offset(), group_size),
            named_constants=(
                (sympy.Symbol("nested_R0_GROUP_SIZE"), group_size, True),
                (sympy.Symbol("nested_R0_REDUCED_BLOCK"), reduced_block, True),
                (sympy.Symbol("nested_R0_REDUCED_NUMEL"), reduced_numel, False),
            ),
        )

    def test_derived_root_uses_active_range_tree_geometry(self):
        graph = self._make_graph()
        with V.set_graph_handler(graph):
            kernel = self._make_kernel(persistent=False)
            x_tree, r_tree = kernel.range_trees
            derived = self._make_derived_root(r_tree)

            self.assertEqual(derived.is_loop, r_tree.is_loop)
            self.assertEqual(derived.prefix, r_tree.prefix)
            self.assertNotEqual(derived.mask_name(), r_tree.mask_name())
            self.assertFalse(kernel._has_constant_mask(derived))

            with V.set_kernel_handler(kernel):
                header = IndentedBuffer()
                kernel.iteration_ranges_codegen_header(derived, header)
                header_code = header.getvalue()
                self.assertIn("nested_R0_GROUP_SIZE: tl.constexpr = 128", header_code)
                self.assertIn(
                    "nested_R0_REDUCED_BLOCK: tl.constexpr = R0_BLOCK // 128",
                    header_code,
                )
                self.assertIn("nested_R0_REDUCED_NUMEL = 4", header_code)
                self.assertIn(
                    "reduced_r0_index = r0_offset // 128 + "
                    "tl.arange(0, R0_BLOCK // 128)[None, :]",
                    header_code,
                )
                self.assertIn(
                    "reduced_r0_index_mask = reduced_r0_index < 4",
                    header_code,
                )

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
