# Owner(s): ["module: inductor"]

import itertools
import math
import sys
import types
import typing
import unittest
import warnings

import sympy

import torch
from torch._inductor import config
from torch._inductor.codegen.mps import MetalScheduling
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
    TritonScheduling,
    TritonSymbols,
)
from torch._inductor.runtime.tiling import get_z_blocks
from torch._inductor.runtime.triton_heuristics import Grid3D, GridExpr, GridND
from torch._inductor.tiling_utils import (
    CoalesceVarAnalysis,
    FusedNormalizedReadsWrites,
    VarTiling,
)
from torch._inductor.virtualized import V
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    subtest,
    TestCase,
)
from torch.testing._internal.inductor_utils import dummy_graph, MockGraphHandler
from torch.testing._internal.logging_utils import logs_to_string
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

    @parametrize(
        "rank, max_tiles, expect_hint",
        [
            (1, 5, False),
            (2, 5, False),
            (3, 3, False),
            (3, 4, True),
            (3, 5, True),
            (4, 4, True),
            (5, 5, True),
        ],
    )
    @torch._inductor.config.patch(
        {"triton.coalesce_tiling_analysis": True, "triton.prefer_nd_tiling": False}
    )
    def test_nd_coalescing_tiling(self, rank, max_tiles, expect_hint):
        index_vars = OrderedSet(sympy.symbols(f"p0:{rank}", integer=True))
        ranges = dict(zip(index_vars, (8 * 2**i for i in range(rank)), strict=True))
        analysis = CoalesceVarAnalysis(
            coalesced_by_var=dict.fromkeys(index_vars, 100),
            uncoalesced_addrs={},
            norm_read_writes=FusedNormalizedReadsWrites(
                index_vars=index_vars,
                reduce_vars=OrderedSet(),
                reads={},
                writes={},
                var_ranges=ranges,
            ),
        )
        log_stream, log_context = logs_to_string(
            "torch._inductor.codegen.simd", "perf_hints"
        )
        with (
            torch._inductor.config.patch({"triton.max_tiles": max_tiles}),
            V.set_graph_handler(self._make_graph()),
            log_context(),
            warnings.catch_warnings(),
        ):
            warnings.filterwarnings(
                "error",
                message="Coalescing tiling analysis enumerates variable combinations",
            )
            selection = TritonScheduling.select_tiling_with_memory(
                [], sympy.Integer(math.prod(ranges.values())), sympy.S.One, analysis
            )
        output = log_stream.getvalue()
        self.assertEqual(
            output.count("Coalescing tiling analysis enumerates variable combinations"),
            int(expect_hint),
        )
        if expect_hint:
            self.assertIn("triton.prefer_nd_tiling=True", output)
        expected = {
            **dict(
                zip(
                    (*(f"z{i}_" for i in reversed(range(rank - 2))), "y", "x")[-rank:],
                    ranges.values(),
                    strict=True,
                )
            ),
            "r0_": 1,
        }
        self.assertEqual(selection.tiling, expected)

    @parametrize(
        "backend,max_tiles",
        [
            subtest((TritonScheduling, None), name="triton_default"),
            subtest((TritonScheduling, 3), name="triton_3"),
            subtest((TritonScheduling, 4), name="triton_4"),
            subtest((MetalScheduling, None), name="metal_default"),
            subtest((MetalScheduling, 3), name="metal_3"),
        ],
    )
    @config.patch(
        {
            "triton.coalesce_tiling_analysis": True,
            "triton.prefer_nd_tiling": False,
            "triton.tile_reductions": False,
        }
    )
    def test_coalescing_reduction_combined_tile_budget(self, backend, max_tiles):
        p0, p1, r0 = sympy.symbols("p0 p1 r0", integer=True)
        analysis = CoalesceVarAnalysis(
            coalesced_by_var={p0: 100, p1: 100, r0: 100},
            uncoalesced_addrs={},
            norm_read_writes=FusedNormalizedReadsWrites(
                index_vars=OrderedSet((p0, p1)),
                reduce_vars=OrderedSet((r0,)),
                reads={},
                writes={},
                var_ranges={
                    p0: sympy.Integer(64),
                    p1: sympy.Integer(32),
                    r0: sympy.Integer(8),
                },
            ),
            suggested_split=VarTiling(p0, 8, 10000),
        )
        expected = (
            {"z0_": 8, "y": 8, "x": 32, "r0_": 8}
            if max_tiles == 4
            else {"y": 64, "x": 32, "r0_": 8}
        )
        with (
            config.patch({"triton.max_tiles": max_tiles}),
            V.set_graph_handler(self._make_graph()),
        ):
            selection = backend.select_tiling_with_memory(
                [], sympy.Integer(64 * 32), sympy.Integer(8), analysis
            )
            self.assertEqual(selection.tiling, expected)

    @parametrize("max_tiles", [None, 0, 1, 2, 3, 4, 5])
    def test_mps_max_tiles(self, max_tiles):
        with config.patch({"triton.max_tiles": max_tiles}):
            if max_tiles is not None and max_tiles > 3:
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"MPS requires torch\._inductor\.config\.triton\.max_tiles <= 3",
                ):
                    MetalScheduling(None)
            else:
                MetalScheduling(None)

    @parametrize("rank", [4, 5])
    def test_nd_lazy_cpp_grid(self, rank):
        grid = GridExpr.from_meta_lazy({"grid_type": "GridND", "rank": rank}, "kernel")
        self.assertEqual(
            grid.x_grid,
            "((xnumel + (kernel_result.xblocks[0] - 1)) / (kernel_result.xblocks[0]))",
        )
        self.assertEqual(
            grid.y_grid,
            "((ynumel + (kernel_result.yblocks[0] - 1)) / (kernel_result.yblocks[0]))",
        )
        expected_z = " * ".join(
            f"((z{i}_numel + (kernel_result.zblocks[{i}] - 1)) / (kernel_result.zblocks[{i}]))"
            for i in range(rank - 2)
        )
        self.assertEqual(grid.z_grid, expected_z)
        self.assertEqual(
            get_z_blocks(
                {"Z2_BLOCK": 8, "R0_BLOCK": 64, "Z0_BLOCK": 32, "Z1_BLOCK": 16}
            ),
            [32, 16, 8],
        )

    def _make_graph(self):
        return MockGraphHandler()

    @parametrize("ndim", [3, 4, 5])
    def test_nd_grid_decode_with_padding(self, ndim):
        numels = {"z0_": 3, "y": 7, "x": 5}
        blocks = {"Z0_BLOCK": 2, "YBLOCK": 2, "XBLOCK": 2}
        if ndim >= 4:
            numels = {"z1_": 3, **numels}
            blocks["Z1_BLOCK"] = 1
        if ndim == 5:
            numels = {"z2_": 5, **numels}
            blocks["Z2_BLOCK"] = 1
        grid_type = Grid3D if ndim == 3 else GridND
        grid = grid_type(inductor_meta={"rank": ndim})
        grid.generate(blocks)
        launch_grid = grid.eval_slow({f"{p}numel": n for p, n in numels.items()})
        self.assertEqual(launch_grid, (3, 4, {3: 2, 4: 6, 5: 30}[ndim]))

        with V.set_graph_handler(self._make_graph()):
            features = SIMDKernelFeatures(
                [], sympy.Integer(math.prod(numels.values())), sympy.Integer(1)
            )
            kernel = TritonKernel(
                {p: sympy.Integer(n) for p, n in numels.items()}
                | {"r0_": sympy.Integer(1)},
                features=features,
            )
            self.assertIs(kernel._get_grid_type(), grid_type)
            expressions = [
                kernel.iteration_ranges_get_pid(tree)
                for tree in kernel.range_trees
                if not tree.is_reduction
            ]
        extents = tuple(
            (n + blocks[f"{p.upper()}BLOCK"] - 1) // blocks[f"{p.upper()}BLOCK"]
            for p, n in numels.items()
        )
        seen = []
        for program in itertools.product(*(range(n) for n in launch_grid)):
            tl = types.SimpleNamespace(
                program_id=lambda axis: program[axis],
                cdiv=lambda n, block: (n + block - 1) // block,
                int64=None,
                int32=None,
            )
            scope = {"tl": tl, **blocks, **{f"{p}numel": n for p, n in numels.items()}}
            coordinates = tuple(
                eval(
                    expr.replace(".to(tl.int64)", "").replace(".to(tl.int32)", ""),
                    scope,
                )
                for expr in expressions
            )
            self.assertEqual(coordinates[-2:], (program[1], program[0]))
            seen.append(coordinates)
        self.assertCountEqual(
            seen, list(itertools.product(*(range(n) for n in extents)))
        )

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
