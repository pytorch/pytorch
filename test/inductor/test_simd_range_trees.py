# Owner(s): ["module: inductor"]

import sys
import typing
import unittest

import sympy

import torch
from torch._inductor import config
from torch._inductor.codegen.simd import (
    DerivedIterationRangesRoot,
    IterationRangesRoot,
    SIMDScheduling,
)
from torch._inductor.codegen.simd_kernel_features import SIMDKernelFeatures
from torch._inductor.codegen.triton import (
    BlockParameters,
    BlockPtrOptions,
    IndexingOptions,
    TMACompatibilityChecker,
    TritonKernel,
    TritonSymbols,
)
from torch._inductor.virtualized import V
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)
from torch.testing._internal.inductor_utils import dummy_graph, MockGraphHandler
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.functions import FloorDiv, ModularIndexing
from torch.utils._sympy.symbol import SymT


try:
    import triton  # noqa: F401
except ImportError:
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("requires triton")  # noqa: B904


@instantiate_parametrized_tests
class TestSIMDRangeTrees(TestCase):
    hw_classification = HardwareClassification.GENERIC

    @parametrize(
        "pointwise,reduction",
        [
            ((), ()),
            (("x",), ("r0_",)),
            (("y", "x"), ("r0_", "r1_")),
            (("z0_", "y", "x"), ("r0_", "r1_", "r2_")),
            (("z1_", "z0_", "y", "x"), ("r0_", "r1_", "r2_", "r3_")),
            (("z2_", "z1_", "z0_", "y", "x"), ("r0_", "r1_", "r2_", "r3_", "r4_")),
        ],
    )
    @config.patch("triton.max_tiles", 1)
    def test_create_tiling_uses_input_rank(self, pointwise, reduction):
        self.assertEqual(
            SIMDScheduling.create_tiling(
                [sympy.Integer(2)] * len(pointwise),
                [sympy.Integer(3)] * len(reduction),
            ),
            {
                **dict.fromkeys(pointwise, sympy.Integer(2)),
                **dict.fromkeys(reduction, sympy.Integer(3)),
            },
        )

    def test_numbered_prefix_grid_assignment(self):
        graph = self._make_graph()
        with V.set_graph_handler(graph):
            kernel = TritonKernel(
                {
                    "z10_": sympy.Integer(2),
                    "y": sympy.Integer(3),
                    "x": sympy.Integer(4),
                },
                features=SIMDKernelFeatures([], sympy.Integer(24), sympy.S.One),
                override_persistent_reduction=False,
                override_cooperative_reduction=False,
                optimize_mask=False,
            )
            self.assertEqual(
                [
                    (tree.prefix, tree.tensor_dim, tree.grid_dim)
                    for tree in kernel.range_trees
                ],
                [("z10_", 0, 2), ("y", 1, 1), ("x", 2, 0)],
            )

    def _make_graph(self):
        return MockGraphHandler()

    def _make_kernel(self, *, persistent: bool, optimize_mask: bool = True):
        features = SIMDKernelFeatures([], sympy.Integer(4), sympy.Integer(512))
        return TritonKernel(
            {"x": sympy.Integer(4), "r0_": sympy.Integer(512)},
            features=features,
            override_persistent_reduction=persistent,
            override_cooperative_reduction=False,
            optimize_mask=optimize_mask,
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

    @parametrize("derived", [False, True])
    def test_reduction_result_masks(self, derived):
        graph = dummy_graph()
        graph.current_device = torch.device("cuda")
        with V.set_graph_handler(graph):
            kernel = self._make_kernel(persistent=False, optimize_mask=False)
            x_tree, r_tree = kernel.range_trees
            if derived:
                r_tree = self._make_derived_root(r_tree)
            with V.set_kernel_handler(kernel), kernel.use_range_trees([x_tree, r_tree]):
                value = kernel.cse.newvar(
                    dtype=torch.float32, shape=tuple(kernel.dense_size_list())
                )
                result = kernel.reduction(torch.int64, torch.float32, "argmax", value)
                self.assertEqual(result.mask_vars, OrderedSet([x_tree.mask_name()]))

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

    @parametrize("persistent", [False, True])
    @parametrize("factor", [2, 128])
    def test_derived_descriptor_config_symbol(self, persistent, factor):
        graph = dummy_graph()
        graph.current_device = torch.device("cuda")
        with V.set_graph_handler(graph):
            kernel = self._make_kernel(persistent=persistent, optimize_mask=False)
            x_tree, r_tree = kernel.range_trees
            derived = self._make_derived_root(r_tree, group_size=sympy.Integer(factor))
            params = BlockParameters(
                shape=[derived.numel],
                strides=[sympy.S.One],
                offsets=[derived.block_offset()],
                block_shape=[derived.block_size()],
            )
            with (
                V.set_kernel_handler(kernel),
                kernel.use_range_trees([x_tree, derived]),
            ):
                checker = TMACompatibilityChecker(
                    kernel, torch.float32, for_store=False, force=False
                )
                self.assertTrue(checker.are_block_parameters_compatible(params))
                expected = {} if persistent else {"R0_BLOCK": 4 * factor}
                self.assertEqual(kernel.tma_min_block_sizes, expected)
                params.block_shape = [
                    sympy.Symbol("nested_R0_REDUCED_BLOCK", integer=True, positive=True)
                ]
                self.assertFalse(checker.are_block_parameters_compatible(params))
                self.assertEqual(kernel.tma_min_block_sizes, expected)

    @parametrize("block_name", ["XBLOCK", "YBLOCK"])
    def test_descriptor_ownership_with_remapped_geometry(self, block_name):
        graph = dummy_graph()
        graph.current_device = torch.device("cuda")
        with V.set_graph_handler(graph):
            kernel = self._make_kernel(persistent=True, optimize_mask=False)
            x_tree, r_tree = kernel.range_trees
            yblock = sympy.Symbol("YBLOCK", integer=True, positive=True)
            remapped_x = DerivedIterationRangesRoot(
                x_tree,
                numel=x_tree.numel,
                block_size=yblock,
                block_offset=sympy.S.Zero,
            )
            remapped_r = DerivedIterationRangesRoot(
                r_tree,
                numel=r_tree.numel,
                block_size=x_tree.block_size(),
                block_offset=x_tree.block_offset(),
            )
            params = BlockParameters(
                shape=[sympy.Integer(512)],
                strides=[sympy.S.One],
                offsets=[sympy.S.Zero],
                block_shape=[yblock if block_name == "YBLOCK" else x_tree.block_size()],
            )
            with (
                V.set_kernel_handler(kernel),
                kernel.use_range_trees([remapped_x, remapped_r]),
            ):
                checker = TMACompatibilityChecker(
                    kernel, torch.float32, for_store=False, force=False
                )
                self.assertTrue(checker.are_block_parameters_compatible(params))
                self.assertEqual(kernel.tma_min_block_sizes, {block_name: 4})

    @parametrize("prefix", ["x", "y", "z0_", "z10_", "r0_", "r10_"])
    def test_trace_symbol_prefix(self, prefix):
        self.assertEqual(
            TritonKernel._trace_symbol_prefix(sympy.Symbol(f"{prefix}index")), prefix
        )

    @parametrize("prefix", ["z10_", "r10_"])
    def test_mask_name_for_numbered_symbol(self, prefix):
        graph = self._make_graph()
        with V.set_graph_handler(graph):
            kernel = self._make_kernel(persistent=False)
            self.assertEqual(
                TritonSymbols.mask_name_for_symbol(
                    kernel, sympy.Symbol(f"{prefix}index")
                ),
                f"{prefix}mask",
            )

    def test_pointer_advancements_rejects_enum_keys(self):
        graph = self._make_graph()
        with V.set_graph_handler(graph):
            kernel = self._make_kernel(persistent=False)
            with self.assertRaisesRegex(KeyError, "XBLOCK"):
                kernel.pointer_advancements[SymT.XBLOCK]

    def test_derived_block_pointer_folded_offset(self):
        graph = self._make_graph()
        with V.set_graph_handler(graph):
            kernel = self._make_kernel(persistent=False)
            x_tree, r_tree = kernel.range_trees
            derived = self._make_derived_root(r_tree, group_size=sympy.Integer(2))
            offset = FloorDiv(derived.block_offset(), 4)
            self.assertEqual(offset, FloorDiv(r_tree.block_offset(), 8))
            params = BlockParameters(
                shape=[sympy.Integer(64), sympy.Integer(8)],
                strides=[sympy.Integer(8), sympy.S.One],
                offsets=[offset, ModularIndexing(derived.block_offset(), 4, 8)],
                block_shape=[FloorDiv(r_tree.block_size(), 8), sympy.Integer(8)],
            )
            with (
                V.set_kernel_handler(kernel),
                kernel.use_range_trees([x_tree, derived]),
            ):
                options = BlockPtrOptions(
                    params=params,
                    constant_offset=sympy.S.Zero,
                    order=[1, 0],
                    mask_vars=OrderedSet(),
                    broadcast_shape=params.block_shape,
                    broadcasting_dims=[False, False],
                    final_shape=params.block_shape,
                    stride_sorter=BlockParameters.StrideSorter([8, 1], [0, 1]),
                )
                self.assertEqual(options.remove_roffsets(offset), 0)
                self.assertEqual(
                    options.advance_roffset(derived),
                    [
                        FloorDiv(r_tree.block_size(), 8),
                        ModularIndexing(r_tree.block_size(), 8, 8),
                    ],
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
