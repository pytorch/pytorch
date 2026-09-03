# Owner(s): ["module: inductor"]
import ast
import contextlib
import dataclasses
import plistlib
import unittest
from collections import namedtuple
from enum import Enum, Flag, IntEnum, IntFlag
from types import SimpleNamespace
from unittest.mock import patch

import sympy

import torch
import torch._inductor.config as inductor_config
from torch._dynamo.exc import BackendCompilerFailed
from torch._inductor import ir
from torch._inductor.choices import InductorChoices
from torch._inductor.codegen import triton_utils
from torch._inductor.codegen.common import ArgName, CSEVariable, SizeArg, TensorArg
from torch._inductor.codegen.cpp_wrapper_cpu import CppWrapperCpu
from torch._inductor.codegen.simd import IterationRangesRoot
from torch._inductor.codegen.simd_kernel_features import SIMDKernelFeatures
from torch._inductor.codegen.triton import (
    _materialize_trunc_to_float_expr,
    FixedTritonConfig,
    get_triton_reduction_function,
    IndexingOptions,
    TritonCSEVariable,
    TritonKernel,
    TritonKernelOverrides,
    TritonSymbols,
)
from torch._inductor.codegen.wrapper import _escape_triton_kernel_source_for_wrapper
from torch._inductor.dtype_propagation import DtypePropagationOpsHandler, promote_types
from torch._inductor.graph import GraphLowering
from torch._inductor.runtime.hints import DeviceProperties
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import (
    get_importable_constexpr_types,
    is_triton_fp8_dtype_supported,
    run_and_get_code,
    run_and_get_kernels,
)
from torch._inductor.virtualized import V
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    subtest,
)
from torch.testing._internal.inductor_utils import (
    GPU_TYPE,
    HAS_CPU,
    HAS_GPU,
    HAS_GPU_AND_TRITON,
)
from torch.utils._sympy.functions import FloorDiv, TruncToFloat, TruncToInt
from torch.utils._sympy.symbol import make_symbol, SymT
from torch.utils._sympy.value_ranges import ValueRanges
from torch.utils._triton import has_triton_package


try:
    from triton_constexpr_configs import (
        runner as LauncherScopeShadowConfig,
        tl as TritonLanguageShadowConfig,
        UserDefinedAttrsLikeConfig,
        UserDefinedAttrsPrivateFieldConfig,
        UserDefinedBareNestedReprConfig,
        UserDefinedPydanticLikeConfig,
        UserDefinedPydanticLikeNoEqConfig,
        UserDefinedTritonKernelCoercingConfig,
        UserDefinedTritonKernelConfigMode,
        UserDefinedTritonKernelConfigNamespace,
        UserDefinedTritonKernelCountingConfig,
        UserDefinedTritonKernelDefaultArgConfig,
        UserDefinedTritonKernelEnumConfig,
        UserDefinedTritonKernelHiddenConfig,
        UserDefinedTritonKernelHiddenDefaultConfig,
        UserDefinedTritonKernelNestedConfig,
        UserDefinedTritonKernelNonInitConfig,
        UserDefinedTritonKernelSelfReferentialConfig,
    )
except ImportError:
    from test.inductor.triton_constexpr_configs import (
        runner as LauncherScopeShadowConfig,
        tl as TritonLanguageShadowConfig,
        UserDefinedAttrsLikeConfig,
        UserDefinedAttrsPrivateFieldConfig,
        UserDefinedBareNestedReprConfig,
        UserDefinedPydanticLikeConfig,
        UserDefinedPydanticLikeNoEqConfig,
        UserDefinedTritonKernelCoercingConfig,
        UserDefinedTritonKernelConfigMode,
        UserDefinedTritonKernelConfigNamespace,
        UserDefinedTritonKernelCountingConfig,
        UserDefinedTritonKernelDefaultArgConfig,
        UserDefinedTritonKernelEnumConfig,
        UserDefinedTritonKernelHiddenConfig,
        UserDefinedTritonKernelHiddenDefaultConfig,
        UserDefinedTritonKernelNestedConfig,
        UserDefinedTritonKernelNonInitConfig,
        UserDefinedTritonKernelSelfReferentialConfig,
    )


def _constexpr_source(value):
    # Render one constant through the production entry point used by
    # define_user_defined_triton_kernel; None when rendering declines.
    from torch._inductor.codegen.wrapper import _render_constexpr_mappings

    try:
        (rendered,), imports = _render_constexpr_mappings([{"VALUE": value}])
    except RuntimeError:
        return None
    return repr(rendered["VALUE"]), imports


@instantiate_parametrized_tests
class TestCodegenTriton(InductorTestCase):
    def setUp(self):
        super().setUp()

        class DummyModule(torch.nn.Module):
            def forward(self, x):
                return x * 2

        self._gm = torch.fx.symbolic_trace(DummyModule())
        self._graph = GraphLowering(self._gm)

        self._stack = contextlib.ExitStack()
        self._stack.enter_context(V.set_graph_handler(self._graph))

    def tearDown(self):
        self._stack.close()
        super().tearDown()

    def test_strict_signed_zero_reduction_function(self):
        for reduction_type in ("min", "max"):
            self.assertEqual(
                get_triton_reduction_function(reduction_type),
                f"triton_helpers.{reduction_type}2",
            )
            with inductor_config.patch(strict_signed_zero=True):
                self.assertEqual(
                    get_triton_reduction_function(reduction_type),
                    f"triton_helpers.{reduction_type}2_strict",
                )

    @unittest.skipUnless(HAS_GPU_AND_TRITON, "requires GPU and Triton")
    def test_strict_signed_zero_unrolled_reduction(self):
        def fn(x):
            return torch.amin(x, dim=1), torch.amax(x, dim=1)

        x = torch.tensor(
            [
                [0.0, -0.0, -0.0, -0.0],
                [-0.0, 0.0, 0.0, 0.0],
                [1.0, float("nan"), -1.0, 0.0],
            ],
            device=GPU_TYPE,
        )
        expected_min, expected_max = fn(x)
        with inductor_config.patch(strict_signed_zero=True):
            (actual_min, actual_max), code = run_and_get_code(
                torch.compile(fn, fullgraph=True), x
            )

        actual_min_bits = actual_min[:2].view(torch.int32)
        actual_max_bits = actual_max[:2].view(torch.int32)
        expected_min_bits = expected_min[:2].view(torch.int32)
        expected_max_bits = expected_max[:2].view(torch.int32)
        self.assertEqual(actual_min_bits, expected_min_bits)
        self.assertEqual(actual_max_bits, expected_max_bits)
        self.assertTrue(torch.isnan(actual_min[2]).item())
        self.assertTrue(torch.isnan(actual_max[2]).item())
        self.assertIn("tl.where", " ".join(code))

    def test_range_tree_entry_ownership_uses_root_identity(self):
        class AlternateR0Root(IterationRangesRoot):
            def block_size(self):
                return sympy.Symbol("ALT_R0_BLOCK", integer=True, positive=True)

        kernel = TritonKernel(
            {"x": sympy.Integer(4), "r0_": sympy.Integer(512)},
            features=SIMDKernelFeatures([], sympy.Integer(4), sympy.Integer(512)),
            override_persistent_reduction=False,
            override_cooperative_reduction=False,
        )
        x_tree, r_tree = kernel.range_trees
        alt_r_tree = AlternateR0Root(
            "alt_r0_index",
            sympy.Integer(4),
            "r0_",
            r_tree.index,
            kernel,
            pid_cache=r_tree.pid_cache,
            is_loop=r_tree.is_loop,
            tensor_dim=r_tree.tensor_dim,
            grid_dim=r_tree.grid_dim,
            has_zdim=r_tree.has_zdim,
        )

        with V.set_kernel_handler(kernel):
            parent_entry = r_tree.full_range()
            alt_entry = alt_r_tree.full_range()

            self.assertEqual(
                alt_r_tree.vars_and_sizes(parent_entry.symbol() + alt_entry.symbol()),
                ([alt_entry.symbol()], [alt_r_tree.numel]),
            )
            saved_range_trees = kernel.range_trees
            kernel.range_trees = [x_tree, alt_r_tree]
            try:
                self.assertEqual(
                    TritonSymbols.get_block_shape(parent_entry.symbol()),
                    (1, "R0_BLOCK"),
                )
            finally:
                kernel.range_trees = saved_range_trees

    def test_importable_constexpr_types_nested_values(self):
        type_specs = get_importable_constexpr_types(
            [
                {
                    "cfg": UserDefinedTritonKernelNestedConfig(
                        nested=UserDefinedTritonKernelConfigNamespace.Nested(offset=2)
                    )
                }
            ]
        )
        self.assertEqual(
            type_specs,
            [
                (
                    UserDefinedTritonKernelConfigNamespace.__module__,
                    "UserDefinedTritonKernelConfigNamespace.Nested",
                    "UserDefinedTritonKernelConfigNamespace",
                ),
                (
                    UserDefinedTritonKernelNestedConfig.__module__,
                    "UserDefinedTritonKernelNestedConfig",
                    "UserDefinedTritonKernelNestedConfig",
                ),
            ],
        )

    def test_importable_constexpr_types_sibling_nested_classes(self):
        namespace = UserDefinedTritonKernelConfigNamespace
        type_specs = get_importable_constexpr_types(
            [namespace.Nested(offset=1), namespace.Sibling(offset=2)]
        )
        self.assertEqual(len(type_specs), 1)
        self.assertEqual(type_specs[0].module, namespace.__module__)
        self.assertEqual(type_specs[0].root_name, namespace.__name__)

    def test_importable_constexpr_types_ignore_repr_spelling(self):
        # Defaults are never repr'd into the generated module (the user's own
        # def-time expression is spliced), so a repr that spells a nested type
        # by its bare name is irrelevant; only the root import matters.
        nested_type = UserDefinedTritonKernelConfigNamespace.BareNested
        type_specs = get_importable_constexpr_types([nested_type(offset=2)])
        self.assertEqual(
            [spec.root_name for spec in type_specs],
            [UserDefinedTritonKernelConfigNamespace.__name__],
        )

    def test_importable_constexpr_types_skip_hidden_dataclass_field(self):
        @dataclasses.dataclass
        class LocalHiddenValue:
            offset: int

        type_specs = get_importable_constexpr_types(
            [UserDefinedTritonKernelHiddenConfig(2, LocalHiddenValue(3))]
        )
        self.assertEqual(len(type_specs), 1)
        self.assertEqual(
            type_specs[0].qualname,
            UserDefinedTritonKernelHiddenConfig.__qualname__,
        )

    def test_importable_constexpr_types_ignore_constructor_shape(self):
        # Same reason as above: an init=False repr field only matters for
        # rendered constants (which decline in _render_constexpr_mappings), not
        # for a default the user constructed themselves.
        value = UserDefinedTritonKernelNonInitConfig(offset=2)
        type_specs = get_importable_constexpr_types([value])
        self.assertEqual(
            [spec.qualname for spec in type_specs],
            [UserDefinedTritonKernelNonInitConfig.__qualname__],
        )

    def test_importable_constexpr_types_skip_builtin_repr(self):
        with patch("builtins.repr") as repr_mock:
            self.assertEqual(get_importable_constexpr_types([{"values": [1, 2]}]), [])
        repr_mock.assert_not_called()

    def test_importable_constexpr_types_reserved_name_error(self):
        with self.assertRaisesRegex(ImportError, "import name tl.*reserved"):
            get_importable_constexpr_types([TritonLanguageShadowConfig(offset=2)])

    def test_importable_constexpr_types_set(self):
        namespace = UserDefinedTritonKernelConfigNamespace
        type_specs = get_importable_constexpr_types(
            [
                {
                    UserDefinedTritonKernelNestedConfig(namespace.Nested(offset=2)),
                    UserDefinedTritonKernelHiddenConfig(3, "hidden"),
                }
            ]
        )
        root_names = [type_spec.root_name for type_spec in type_specs]
        self.assertEqual(root_names, sorted(root_names))
        self.assertEqual(
            root_names,
            [
                "UserDefinedTritonKernelConfigNamespace",
                "UserDefinedTritonKernelHiddenConfig",
                "UserDefinedTritonKernelNestedConfig",
            ],
        )

    def test_importable_constexpr_types_repr_protocols(self):
        nested_type = UserDefinedTritonKernelConfigNamespace.Nested

        @dataclasses.dataclass
        class LocalHiddenValue:
            offset: int

        for config_type in (UserDefinedAttrsLikeConfig, UserDefinedPydanticLikeConfig):
            type_specs = get_importable_constexpr_types(
                [config_type(nested_type(offset=2), LocalHiddenValue(3))]
            )
            self.assertEqual(
                [type_spec.qualname for type_spec in type_specs],
                [config_type.__qualname__, nested_type.__qualname__],
            )

    def test_importable_constexpr_types_local_class_error(self):
        @dataclasses.dataclass(frozen=True)
        class LocalConfig:
            offset: int

        with self.assertRaisesRegex(ImportError, "not importable"):
            get_importable_constexpr_types([LocalConfig(offset=2)])

    def test_escape_triton_kernel_source_for_wrapper(self):
        source = """\
@triton.jit
def helper(x):
    s = "slash \\\\"
    t = '''quoted'''
    \"\"\"doc\"\"\"
    return x
"""

        for cpp_wrapper in (False, True):
            with inductor_config.patch("cpp_wrapper", cpp_wrapper):
                escaped = _escape_triton_kernel_source_for_wrapper(source)

            wrapper_src = f"async_compile.triton('helper', '''{escaped}''')"
            compile(wrapper_src, "<test-wrapper-source>", "exec")

            if cpp_wrapper:
                nested_src = f'wrapper_src = r"""{wrapper_src}"""'
                compile(nested_src, "<test-nested-wrapper-source>", "exec")
                wrapper_src = ast.literal_eval(ast.parse(nested_src).body[0].value)

            call = ast.parse(wrapper_src).body[0].value
            self.assertEqual(ast.literal_eval(call.args[1]), source)

    def test_persistent_reduction_choice_two_arg_override(self):
        seen_scores = []

        class CustomChoices(InductorChoices):
            @staticmethod
            def should_use_persistent_reduction(features, cooperative_reduction):
                seen_scores.append(features.tiling_scores)
                return False

        tiling_scores = {"x": sympy.Integer(1), "r0_": sympy.Integer(32)}
        with V.set_choices_handler(CustomChoices()):
            kernel = TritonKernel(
                {"x": sympy.Integer(4), "r0_": sympy.Integer(512)},
                features=SIMDKernelFeatures([], sympy.Integer(4), sympy.Integer(512)),
                tiling_scores=tiling_scores,
                override_cooperative_reduction=False,
            )

        self.assertFalse(kernel.persistent_reduction)
        self.assertEqual(seen_scores, [tiling_scores])

    def test_reduction_invariant_load_indexing(self):
        self._stack.enter_context(self._graph.set_current_device(torch.device("cuda")))
        xnumel = sympy.Integer(65)
        rnumel = sympy.Integer(65)
        kernel = TritonKernel(
            {"x": xnumel, "r0_": rnumel},
            features=SIMDKernelFeatures([], xnumel, rnumel),
            override_persistent_reduction=False,
            override_cooperative_reduction=False,
        )

        with V.set_kernel_handler(kernel):
            x_tree, r_tree = kernel.range_trees
            invariant_index = sympy.Symbol("s0", integer=True, positive=True)
            x_index = x_tree.full_range().symbol()
            r_index = r_tree.full_range().symbol()
            scalar_mask = TritonCSEVariable(
                "tmp0", ValueRanges.unknown(), torch.bool, shape=("1", "1")
            )

            def indexing(index):
                options = kernel.indexing(
                    index,
                    allow_reduction_invariant_indexing=True,
                )
                self.assertIsInstance(options, IndexingOptions)
                return options

            with patch.object(kernel, "_load_mask", scalar_mask):
                invariant_options = indexing(invariant_index)
                x_options = indexing(x_index)
                reduction_options = indexing(r_index)

            self.assertEqual(
                tuple(map(str, invariant_options.expand_shape or ())), ("1", "1")
            )
            self.assertEqual(
                tuple(map(str, x_options.expand_shape or ())), ("XBLOCK", "1")
            )
            self.assertTrue(invariant_options.reduction_axes_omitted)
            self.assertTrue(x_options.reduction_axes_omitted)
            self.assertFalse(reduction_options.reduction_axes_omitted)

            x_mask = TritonCSEVariable(
                "tmp1", ValueRanges.unknown(), torch.bool, shape=("XBLOCK", "1")
            )
            with patch.object(kernel, "_load_mask", x_mask):
                predicate_options = indexing(invariant_index)
            self.assertEqual(
                tuple(map(str, predicate_options.expand_shape or ())),
                ("XBLOCK", "1"),
            )
            self.assertTrue(predicate_options.reduction_axes_omitted)

    def test_reduction_invariant_load_indexing_extents(self):
        self._stack.enter_context(self._graph.set_current_device(torch.device("cuda")))
        xnumel = sympy.Integer(65)
        rnumel = sympy.Integer(65)
        kernel = TritonKernel(
            {"x": xnumel, "r0_": rnumel},
            features=SIMDKernelFeatures([], xnumel, rnumel),
            override_persistent_reduction=False,
            override_cooperative_reduction=False,
        )

        with V.set_kernel_handler(kernel):
            x_tree, r_tree = kernel.range_trees
            self.assertTrue(r_tree.is_loop)
            invariant_index = sympy.Symbol("s0", integer=True, positive=True)

            def indexing(mask):
                with patch.object(kernel, "_load_mask", mask):
                    options = kernel.indexing(
                        invariant_index,
                        allow_reduction_invariant_indexing=True,
                    )
                self.assertIsInstance(options, IndexingOptions)
                return options

            scalar_mask = TritonCSEVariable(
                "tmp0", ValueRanges.unknown(), torch.bool, shape=("1", "1")
            )
            for reduction_numel in (
                sympy.Symbol("u1", integer=True, nonnegative=True),
                sympy.S.Zero,
            ):
                with (
                    self.subTest(reduction_numel=reduction_numel),
                    patch.object(r_tree, "numel", reduction_numel),
                ):
                    self.assertTrue(indexing(scalar_mask).reduction_axes_omitted)

        persistent_kernel = TritonKernel(
            {"x": xnumel, "r0_": rnumel},
            features=SIMDKernelFeatures([], xnumel, rnumel),
            optimize_mask=False,
            override_persistent_reduction=True,
            override_cooperative_reduction=False,
        )
        with V.set_kernel_handler(persistent_kernel):
            r_tree = persistent_kernel.range_trees[1]
            self.assertFalse(r_tree.is_loop)
            scalar_mask = TritonCSEVariable(
                "tmp1", ValueRanges.unknown(), torch.bool, shape=("1", "1")
            )

            def persistent_indexing():
                with patch.object(persistent_kernel, "_load_mask", scalar_mask):
                    options = persistent_kernel.indexing(
                        sympy.Symbol("s1", integer=True, positive=True),
                        allow_reduction_invariant_indexing=True,
                    )
                self.assertIsInstance(options, IndexingOptions)
                return options

            self.assertTrue(persistent_indexing().reduction_axes_omitted)
            for reduction_numel in (
                sympy.Symbol("u2", integer=True, nonnegative=True),
                sympy.S.Zero,
            ):
                with (
                    self.subTest(persistent_reduction_numel=reduction_numel),
                    patch.object(r_tree, "numel", reduction_numel),
                ):
                    self.assertFalse(persistent_indexing().reduction_axes_omitted)

        no_x_kernel = TritonKernel(
            {"x": xnumel, "r0_": rnumel},
            features=SIMDKernelFeatures([], xnumel, rnumel),
            fixed_config=FixedTritonConfig({"XBLOCK": 1, "R0_BLOCK": 128}),
            is_combo_kernel=True,
            optimize_mask=False,
            per_subkernel_blocks=True,
            override_persistent_reduction=True,
            override_cooperative_reduction=False,
        )
        with V.set_kernel_handler(no_x_kernel):
            x_tree, r_tree = no_x_kernel.range_trees
            self.assertTrue(no_x_kernel.no_x_dim)
            self.assertIsNone(x_tree.tensor_dim)
            self.assertEqual(r_tree.tensor_dim, 0)
            scalar_mask = TritonCSEVariable(
                "tmp2", ValueRanges.unknown(), torch.bool, shape=("1",)
            )

            def no_x_indexing():
                with patch.object(no_x_kernel, "_load_mask", scalar_mask):
                    options = no_x_kernel.indexing(
                        sympy.Symbol("s1", integer=True, positive=True),
                        allow_reduction_invariant_indexing=True,
                    )
                self.assertIsInstance(options, IndexingOptions)
                return options

            self.assertEqual(
                tuple(map(str, no_x_indexing().expand_shape or ())),
                ("1",),
            )
            for pointwise_numel in (
                sympy.Symbol("u0", integer=True, nonnegative=True),
                sympy.S.Zero,
            ):
                with (
                    self.subTest(pointwise_numel=pointwise_numel),
                    patch.object(x_tree, "numel", pointwise_numel),
                ):
                    self.assertTrue(no_x_indexing().reduction_axes_omitted)

    def test_reduction_invariant_load_indexing_device_gate(self):
        xnumel = sympy.Integer(65)
        rnumel = sympy.Integer(65)
        kernel = TritonKernel(
            {"x": xnumel, "r0_": rnumel},
            features=SIMDKernelFeatures([], xnumel, rnumel),
            override_persistent_reduction=False,
            override_cooperative_reduction=False,
        )

        with V.set_kernel_handler(kernel):
            x_index = kernel.range_trees[0].full_range().symbol()
            x_mask = TritonCSEVariable(
                "tmp0", ValueRanges.unknown(), torch.bool, shape=("XBLOCK", "1")
            )
            self.assertIsNone(self._graph.current_device)
            for device, expected_narrowing in (
                (None, False),
                (torch.device("cpu"), False),
                (torch.device("cuda"), True),
            ):
                device_context = (
                    contextlib.nullcontext()
                    if device is None
                    else self._graph.set_current_device(device)
                )
                with (
                    self.subTest(device=device),
                    device_context,
                    patch.object(kernel, "_load_mask", x_mask),
                ):
                    options = kernel.indexing(
                        x_index,
                        allow_reduction_invariant_indexing=True,
                    )
                self.assertIsInstance(options, IndexingOptions)
                self.assertEqual(expected_narrowing, options.reduction_axes_omitted)

    def test_reduction_invariant_load_indexing_unknown_mask(self):
        self._stack.enter_context(self._graph.set_current_device(torch.device("cuda")))
        xnumel = sympy.Integer(65)
        rnumel = sympy.Integer(65)
        kernel = TritonKernel(
            {"x": xnumel, "r0_": rnumel},
            features=SIMDKernelFeatures([], xnumel, rnumel),
            override_persistent_reduction=False,
            override_cooperative_reduction=False,
        )

        with V.set_kernel_handler(kernel):
            scalar_mask = TritonCSEVariable(
                "tmp0", ValueRanges.unknown(), torch.bool, shape=("1", "1")
            )
            indirect = kernel.cse.namedvar("tmp1", dtype=torch.int64, shape=("1", "1"))
            self.assertIsInstance(indirect, TritonCSEVariable)
            indirect_index = sympy.Symbol(indirect.name, integer=True)

            with patch.object(kernel, "_load_mask", scalar_mask):
                resolved_options = kernel.indexing(
                    indirect_index,
                    allow_reduction_invariant_indexing=True,
                )
                indirect.mask_vars.add("unknown_mask")
                options = kernel.indexing(
                    indirect_index,
                    allow_reduction_invariant_indexing=True,
                )

            self.assertIsInstance(resolved_options, IndexingOptions)
            self.assertTrue(resolved_options.reduction_axes_omitted)
            self.assertIsInstance(options, IndexingOptions)
            self.assertFalse(options.reduction_axes_omitted)
            self.assertEqual(
                tuple(map(str, options.expand_shape or ())),
                ("XBLOCK", "R0_BLOCK"),
            )

    def test_reduction_invariant_load_indexing_schedule_guards(self):
        self._stack.enter_context(self._graph.set_current_device(torch.device("cuda")))
        xnumel = sympy.Integer(65)
        rnumel = sympy.Integer(65)
        kernel = TritonKernel(
            {"x": xnumel, "r0_": rnumel},
            features=SIMDKernelFeatures([], xnumel, rnumel),
            override_persistent_reduction=False,
            override_cooperative_reduction=False,
        )

        with V.set_kernel_handler(kernel):
            x_tree = kernel.range_trees[0]
            x_index = x_tree.full_range().symbol()
            scalar_mask = TritonCSEVariable(
                "tmp0", ValueRanges.unknown(), torch.bool, shape=("1", "1")
            )
            x_mask = TritonCSEVariable(
                "tmp1", ValueRanges.unknown(), torch.bool, shape=("XBLOCK", "1")
            )
            invariant_index = sympy.Symbol("s0", integer=True, positive=True)
            for guarded_mode in (
                "cooperative_reduction",
                "mix_order_reduction",
            ):
                with (
                    self.subTest(guarded_mode=guarded_mode),
                    patch.object(kernel, guarded_mode, True),
                ):
                    with patch.object(kernel, "_load_mask", scalar_mask):
                        options = kernel.indexing(
                            invariant_index,
                            allow_reduction_invariant_indexing=True,
                        )
                        self.assertIsInstance(options, IndexingOptions)
                        self.assertFalse(options.reduction_axes_omitted)
                    with patch.object(kernel, "_load_mask", x_mask):
                        options = kernel.indexing(
                            x_index,
                            allow_reduction_invariant_indexing=True,
                        )
                        self.assertIsInstance(options, IndexingOptions)
                        self.assertFalse(options.reduction_axes_omitted)

    def test_reduction_invariant_load_indexing_copy_shape(self):
        xnumel = sympy.Integer(65)
        rnumel = sympy.Integer(65)
        kernel = TritonKernel(
            {"x": xnumel, "r0_": rnumel},
            features=SIMDKernelFeatures([], xnumel, rnumel),
            override_persistent_reduction=False,
            override_cooperative_reduction=False,
        )

        with V.set_kernel_handler(kernel):
            x_index = kernel.range_trees[0].full_range().symbol()
            x_mask = TritonCSEVariable(
                "tmp0", ValueRanges.unknown(), torch.bool, shape=("XBLOCK", "1")
            )
            with patch.object(kernel, "_load_mask", x_mask):
                options = kernel.indexing(
                    x_index,
                    copy_shape=("XBLOCK", "R0_BLOCK"),
                    allow_reduction_invariant_indexing=True,
                )

        self.assertIsInstance(options, IndexingOptions)
        self.assertFalse(options.reduction_axes_omitted)
        self.assertEqual(options.expand_shape, ("XBLOCK", "R0_BLOCK"))

    def test_reduction_invariant_load_indexing_override_mask(self):
        xnumel = sympy.Integer(65)
        rnumel = sympy.Integer(65)
        kernel = TritonKernel(
            {"x": xnumel, "r0_": rnumel},
            features=SIMDKernelFeatures([], xnumel, rnumel),
            override_persistent_reduction=False,
            override_cooperative_reduction=False,
        )

        with V.set_kernel_handler(kernel):
            x_tree, r_tree = kernel.range_trees
            x_index = x_tree.full_range().symbol()
            override_mask = r_tree.mask_name()
            x_mask = TritonCSEVariable(
                "tmp0", ValueRanges.unknown(), torch.bool, shape=("XBLOCK", "1")
            )
            with patch.object(kernel, "_load_mask", x_mask):
                options = kernel.indexing(
                    x_index,
                    override_mask=override_mask,
                    allow_reduction_invariant_indexing=True,
                )

        self.assertIsInstance(options, IndexingOptions)
        self.assertFalse(options.reduction_axes_omitted)
        self.assertEqual(options.expand_shape, ("XBLOCK", "R0_BLOCK"))
        self.assertIn(override_mask, options.mask_vars)

    @inductor_config.patch("triton.divisible_by_16", True)
    def test_config_of_sizearg(self):
        from torch._inductor.utils import (
            get_triton_attrs_descriptor_version,
            TritonAttrsDescriptorVersion,
        )

        two = sympy.Integer(2)
        eight = sympy.Integer(8)
        sixteen = sympy.Integer(16)
        s0 = sympy.Symbol("s0", positive=True, integer=True)
        s1 = sympy.Symbol("s1", positive=True, integer=True)

        def _check_divisibility(expected_divisible_indices, config):
            if get_triton_attrs_descriptor_version() in {
                TritonAttrsDescriptorVersion.V1_COMPILER,
                TritonAttrsDescriptorVersion.V0_NO_TRITON,
            }:
                self.assertEqual(expected_divisible_indices, config.divisible_by_16)
            elif get_triton_attrs_descriptor_version() in {
                TritonAttrsDescriptorVersion.V2_BACKENDS,
                TritonAttrsDescriptorVersion.V3_BACKENDS_TUPLE,
            }:
                self.assertEqual(expected_divisible_indices, config.divisibility_16)
            else:
                if (
                    get_triton_attrs_descriptor_version()
                    != TritonAttrsDescriptorVersion.V4_DICT
                ):
                    raise AssertionError
                self.assertIsInstance(config, dict)

                for idx in expected_divisible_indices:
                    # config is in the form
                    # {(idx,): [["tt.divisibility", 16]]}
                    # where (idx,) is a tuple in order to support tuple inputs to triton kernels.
                    self.assertTrue((idx,) in config)
                    self.assertTrue(["tt.divisibility", 16] in config[(idx,)])

        _check_divisibility(
            (2,),
            triton_utils.config_of(
                [
                    SizeArg("A", two),  # no
                    SizeArg("B", eight),  # no
                    SizeArg("C", sixteen),  # yes
                    SizeArg("D", s0),  # no
                    SizeArg("E", s1),  # no
                ]
            ),
        )

        _check_divisibility(
            (0, 2, 4, 5, 6),
            triton_utils.config_of(
                [
                    SizeArg("A", two * eight),  # 0: yes
                    SizeArg("B", eight * s0),  # 1: no
                    SizeArg("C", two * eight * s0),  # 2: yes
                    SizeArg("D", s0 * s1),  # 3: no
                    SizeArg("E", sixteen * s0),  # 4: yes
                    SizeArg("F", sixteen * eight * s0 * s1),  # 5: yes
                    SizeArg("G", two * eight * s0 * s1),  # 6: yes
                ]
            ),
        )

    def test_config_of_sizearg_with_check_constraint(self):
        from torch.utils._sympy.functions import Mod

        s2 = sympy.Symbol("s2", positive=True, integer=True)

        self.assertFalse(
            V.graph.sizevars.statically_known_multiple_of(s2, 16),
        )

        shape_env = V.graph.sizevars.shape_env
        shape_env.axioms[sympy.Eq(Mod(s2, 16), 0)] = sympy.true

        self.assertTrue(
            V.graph.sizevars.statically_known_multiple_of(s2, 16),
        )

    @inductor_config.patch("triton.divisible_by_16", True)
    def test_config_of_skips_graph_input_tensor_divisibility_for_cpp_wrapper_jit(self):
        from torch._inductor.utils import (
            get_triton_attrs_descriptor_version,
            TritonAttrsDescriptorVersion,
        )

        def _has_divisibility_16(config, idx):
            if get_triton_attrs_descriptor_version() in {
                TritonAttrsDescriptorVersion.V1_COMPILER,
                TritonAttrsDescriptorVersion.V0_NO_TRITON,
            }:
                return idx in config.divisible_by_16
            if get_triton_attrs_descriptor_version() in {
                TritonAttrsDescriptorVersion.V2_BACKENDS,
                TritonAttrsDescriptorVersion.V3_BACKENDS_TUPLE,
            }:
                return idx in config.divisibility_16
            self.assertIsInstance(config, dict)
            return (idx,) in config and ["tt.divisibility", 16] in config[(idx,)]

        input_arg = TensorArg(name="in_ptr0", buffer="arg0_1", dtype=torch.float32)
        buffer_arg = TensorArg(name="out_ptr0", buffer="buf0", dtype=torch.float32)
        V.graph.graph_inputs[input_arg.buffer] = object()

        original_cpp_wrapper = V.graph.cpp_wrapper
        original_aot_mode = V.graph.aot_mode
        original_graph_input_names = list(V.graph.graph_input_names)
        original_inputs_to_check = V.graph.inputs_to_check
        try:
            V.graph.graph_input_names = [input_arg.buffer]
            with patch.object(triton_utils, "is_unaligned_buffer", return_value=False):

                def _check(input_divisible, buffer_divisible, *, skip=False):
                    triton_config = triton_utils.config_of(
                        [input_arg, buffer_arg],
                        skip_cpp_wrapper_input_tensor_alignment=skip,
                    )
                    self.assertEqual(
                        input_divisible, _has_divisibility_16(triton_config, 0)
                    )
                    self.assertEqual(
                        buffer_divisible, _has_divisibility_16(triton_config, 1)
                    )

                V.graph.cpp_wrapper = False
                _check(True, True)

                V.graph.cpp_wrapper = True
                V.graph.aot_mode = False
                _check(True, True)

                V.graph.inputs_to_check = ()
                _check(True, True, skip=True)

                V.graph.inputs_to_check = (0,)
                _check(False, True, skip=True)

                V.graph.aot_mode = True
                _check(True, True, skip=True)
        finally:
            V.graph.cpp_wrapper = original_cpp_wrapper
            V.graph.aot_mode = original_aot_mode
            V.graph.graph_input_names = original_graph_input_names
            V.graph.inputs_to_check = original_inputs_to_check
            V.graph.graph_inputs.pop(input_arg.buffer, None)

    def test_cpp_wrapper_python_fallback_return_slots_skip_mutation_outputs(self):
        original_cpp_wrapper = V.graph.cpp_wrapper
        try:
            V.graph.cpp_wrapper = True
            with torch.library._scoped_library(
                "cpp_wrapper_fallback_test", "FRAGMENT"
            ) as m:
                m.define(
                    "mutate_and_return_(Tensor(a!) out, Tensor x, str tag) -> Tensor"
                )

                op_overload = (
                    torch.ops.cpp_wrapper_fallback_test.mutate_and_return_.default
                )
                wrapper = CppWrapperCpu()
                mutation_output = object.__new__(ir.MutationOutput)
                raw_args = [
                    SimpleNamespace(codegen_reference=lambda: "out_handle"),
                    SimpleNamespace(codegen_reference=lambda: "x_handle"),
                    "tag",
                ]

                wrapper.generate_fallback_kernel_with_runtime_lookup_python(
                    "buf",
                    "test_kernel",
                    op_overload,
                    raw_args=raw_args,
                    output_args=["mutated", "actual"],
                    raw_outputs=[mutation_output, object()],
                )

                code = "\n".join(str(line) for line in wrapper.lines)
                self.assertIn(
                    "actual = reinterpret_cast<AtenTensorHandle>("
                    "PyCapsule_GetPointer(py_buf.get(), NULL));",
                    code,
                )
                self.assertNotIn("PyList_GET_ITEM(py_buf.get(), 1)", code)
                self.assertNotIn("RAIIAtenTensorHandle mutated;", code)
        finally:
            V.graph.cpp_wrapper = original_cpp_wrapper

    def test_pow_uses_active_override_constant_lowering(self):
        exponent = CSEVariable("ks0", ValueRanges.unknown(), torch.int64)

        class TestTritonKernelOverrides(TritonKernelOverrides):
            @classmethod
            def constant(cls, value, dtype):
                return f"custom_constant({value}, {dtype})"

        self.assertEqual(
            TestTritonKernelOverrides.pow(2, exponent),
            "libdevice.pow(custom_constant(2, torch.float64), (ks0).to(tl.float64))",
        )

    def test_pow_preserves_integer_dtype_for_unsigned_scalar_exponents(self):
        exponent = CSEVariable("ks0", ValueRanges.unknown(), torch.uint32)

        self.assertEqual(
            DtypePropagationOpsHandler().pow(2, exponent),
            promote_types([2, exponent]),
        )

    def test_pow_uses_integer_helper_for_unsigned_scalar_exponents(self):
        exponent = CSEVariable("ks0", ValueRanges.unknown(), torch.uint32)

        class TestTritonKernelOverrides(TritonKernelOverrides):
            @classmethod
            def constant(cls, value, dtype):
                return f"custom_constant({value}, {dtype})"

        self.assertEqual(
            TestTritonKernelOverrides.pow(3, exponent),
            "triton_helpers.pow_integer(custom_constant(3, torch.uint32), ks0)",
        )

    def test_materialize_trunc_to_float_expr_preserves_integer_subexpressions(self):
        s0 = sympy.Symbol("s0")

        trunc_expr = TruncToInt(s0)
        self.assertEqual(
            _materialize_trunc_to_float_expr(trunc_expr, torch.float64),
            TruncToFloat(s0),
        )

        integer_expr = FloorDiv(trunc_expr, sympy.Integer(5))
        self.assertEqual(
            _materialize_trunc_to_float_expr(integer_expr, torch.float64),
            integer_expr,
        )

        predicate_expr = sympy.Eq(trunc_expr, sympy.Integer(9007199254740993))
        self.assertEqual(
            _materialize_trunc_to_float_expr(predicate_expr, torch.float64),
            predicate_expr,
        )

        float_expr = sympy.Float(0.5) + trunc_expr
        self.assertEqual(
            _materialize_trunc_to_float_expr(float_expr, torch.float64),
            sympy.Float(0.5) + TruncToFloat(s0),
        )

    @inductor_config.patch("triton.emit_pointer_range_32", True)
    def test_config_of_emit_pointer_range_32_enabled(self):
        from torch._inductor.utils import (
            get_triton_attrs_descriptor_version,
            TritonAttrsDescriptorVersion,
        )

        sixteen = sympy.Integer(16)
        s0 = sympy.Symbol("s0", positive=True, integer=True)

        config = triton_utils.config_of(
            [SizeArg("A", sixteen), SizeArg("B", s0)],
            pointer_range_override=(0,),
        )

        if get_triton_attrs_descriptor_version() in {
            TritonAttrsDescriptorVersion.V0_NO_TRITON,
            TritonAttrsDescriptorVersion.V1_COMPILER,
            TritonAttrsDescriptorVersion.V2_BACKENDS,
            TritonAttrsDescriptorVersion.V3_BACKENDS_TUPLE,
        }:
            self.assertEqual(config.pointer_range_32, (0,))
        else:
            self.assertIsInstance(config, dict)
            self.assertIn(["tt.pointer_range", 32], config[(0,)])

    @inductor_config.patch("triton.emit_pointer_range_32", False)
    def test_config_of_emit_pointer_range_32_disabled(self):
        from torch._inductor.utils import (
            get_triton_attrs_descriptor_version,
            TritonAttrsDescriptorVersion,
        )

        sixteen = sympy.Integer(16)
        s0 = sympy.Symbol("s0", positive=True, integer=True)

        config = triton_utils.config_of(
            [SizeArg("A", sixteen), SizeArg("B", s0)],
            pointer_range_override=(),
        )

        if get_triton_attrs_descriptor_version() in {
            TritonAttrsDescriptorVersion.V0_NO_TRITON,
            TritonAttrsDescriptorVersion.V1_COMPILER,
            TritonAttrsDescriptorVersion.V2_BACKENDS,
            TritonAttrsDescriptorVersion.V3_BACKENDS_TUPLE,
        }:
            self.assertEqual(config.pointer_range_32, ())
        else:
            self.assertIsInstance(config, dict)
            if (0,) in config:
                self.assertNotIn(["tt.pointer_range", 32], config[(0,)])

    @unittest.skipUnless(torch.version.hip is not None, "pointer_range_32 is HIP-only")
    @unittest.skipUnless(HAS_GPU_AND_TRITON, "requires GPU and Triton")
    def test_pointer_range_in_generated_code(self):
        """Verify tt.pointer_range=32 appears in generated Triton code on HIP."""

        def fn(x):
            return x + 1

        x = torch.randn(64, 64, device=GPU_TYPE, dtype=torch.bfloat16)
        _, code = run_and_get_code(torch.compile(fn), x)
        code_str = " ".join(code)
        self.assertIn("tt.pointer_range", code_str)

    def test_is_multiple_of_rules(self):
        """Test structural divisibility rules in _is_multiple_of."""
        from torch.utils._sympy.functions import FloorDiv, Mod

        sv = V.graph.sizevars
        shape_env = sv.shape_env

        s1 = sympy.Symbol("s1", positive=True, integer=True)
        s2 = sympy.Symbol("s2", positive=True, integer=True)
        s3 = sympy.Symbol("s3", positive=True, integer=True)

        # Product: any factor divisible → product divisible
        self.assertTrue(sv.statically_known_multiple_of(16 * s1, 16))
        self.assertTrue(sv.statically_known_multiple_of(4 * 4 * s1, 16))
        shape_env.axioms[sympy.Eq(Mod(s1, 16), 0)] = sympy.true
        self.assertTrue(sv.statically_known_multiple_of(s1 * s2, 16))
        self.assertFalse(sv.statically_known_multiple_of(s2 * s3, 16))

        # Sum: all terms divisible → sum divisible
        self.assertFalse(sv.statically_known_multiple_of(s1 + s2, 16))
        shape_env.axioms[sympy.Eq(Mod(s2, 16), 0)] = sympy.true
        self.assertTrue(sv.statically_known_multiple_of(s1 + s2, 16))
        self.assertTrue(sv.statically_known_multiple_of(s1 + 32, 16))
        self.assertFalse(sv.statically_known_multiple_of(s1 + 3, 16))

        # FloorDiv(a, b): a must be multiple of b*n
        self.assertFalse(sv.statically_known_multiple_of(FloorDiv(s1, 3), 16))
        shape_env.axioms[sympy.Eq(Mod(s3, 48), 0)] = sympy.true
        self.assertTrue(sv.statically_known_multiple_of(FloorDiv(s3, 3), 16))

        # Mod(a, b): both a and b must be multiples of n
        self.assertTrue(sv.statically_known_multiple_of(Mod(s1, 48), 16))
        s_nodiv = sympy.Symbol("s_nodiv", positive=True, integer=True)
        self.assertFalse(sv.statically_known_multiple_of(Mod(s_nodiv, 32), 16))
        self.assertFalse(sv.statically_known_multiple_of(Mod(s1, 7), 16))

        # Axiom fallback: bare symbol resolved via statically_known_true
        s4 = sympy.Symbol("s4", positive=True, integer=True)
        self.assertFalse(sv.statically_known_multiple_of(s4, 8))
        shape_env.axioms[sympy.Eq(Mod(s4, 8), 0)] = sympy.true
        self.assertTrue(sv.statically_known_multiple_of(s4, 8))

    def test_signature_of_fp8_dtypes(self):
        """fp8 dtypes should produce correct Triton pointer signatures via _type_of."""
        expected = {
            torch.float8_e4m3fn: "*fp8e4nv",
            torch.float8_e5m2: "*fp8e5",
            torch.float8_e4m3fnuz: "*fp8e4b8",
            torch.float8_e5m2fnuz: "*fp8e5b16",
        }
        for dtype, expected_sig in expected.items():
            arg = TensorArg(name="x", buffer="buf0", dtype=dtype)
            sig = triton_utils.signature_of(arg, size_dtype=None)
            self.assertEqual(
                sig, expected_sig, lambda msg: f"{msg}\nwrong signature for {dtype}"
            )

    @unittest.skipUnless(has_triton_package(), "requires Triton package")
    def test_fp8_dtype_support_matrix(self):
        self.assertFalse(
            is_triton_fp8_dtype_supported(
                torch.float8_e4m3fn, triton_backend="cuda", triton_arch=80
            )
        )
        self.assertTrue(
            is_triton_fp8_dtype_supported(
                torch.float8_e4m3fn, triton_backend="cuda", triton_arch=89
            )
        )
        self.assertTrue(
            is_triton_fp8_dtype_supported(
                torch.float8_e5m2, triton_backend="cuda", triton_arch=75
            )
        )
        self.assertFalse(
            is_triton_fp8_dtype_supported(
                torch.float8_e4m3fnuz, triton_backend="cuda", triton_arch=100
            )
        )
        self.assertFalse(
            is_triton_fp8_dtype_supported(
                torch.float8_e5m2fnuz, triton_backend="cuda", triton_arch=100
            )
        )
        self.assertTrue(
            is_triton_fp8_dtype_supported(
                torch.float8_e4m3fnuz, triton_backend="hip", triton_arch="gfx942"
            )
        )
        self.assertTrue(
            is_triton_fp8_dtype_supported(
                torch.float8_e5m2fnuz, triton_backend="hip", triton_arch="gfx942"
            )
        )

    def test_signature_of_float8_e4m3fn_uses_uint8_on_pre_sm89_cuda_inputs(self):
        class FakeGraph:
            mutated_buffers = set()

            def is_unspec_arg(self, name):
                return False

            def get_current_device_or_throw(self):
                return torch.device("cuda")

        props = DeviceProperties(
            type="cuda",
            index=0,
            multi_processor_count=1,
            cc=80,
            major=8,
        )
        arg = TensorArg(name="in_ptr0", buffer="buf0", dtype=torch.float8_e4m3fn)
        out_arg = TensorArg(name="out_ptr0", buffer="buf0", dtype=torch.float8_e4m3fn)

        with (
            patch.object(torch.version, "hip", None),
            V.set_graph_handler(FakeGraph()),
            patch.object(DeviceProperties, "create", return_value=props),
        ):
            self.assertEqual(triton_utils.signature_of(arg, size_dtype=None), "*u8")
            self.assertEqual(
                triton_utils.signature_of(out_arg, size_dtype=None), "*fp8e4nv"
            )

        with (
            patch.object(torch.version, "hip", None),
            V.set_graph_handler(FakeGraph()),
            patch.object(
                DeviceProperties, "create", return_value=props._replace(cc=89)
            ),
        ):
            self.assertEqual(
                triton_utils.signature_of(arg, size_dtype=None), "*fp8e4nv"
            )

    @inductor_config.patch("_use_fp64_for_unbacked_floats", True)
    @patch(
        "torch._inductor.codegen.triton_utils.device_supports_fp64",
        return_value=True,
    )
    def test_signature_to_meta_can_match_triton_python_float_signature(self, mock):
        class FakeGraph:
            current_device = torch.device("cuda")

        signature = [
            SizeArg("scale", 0.5),
            SizeArg("runtime_scale", make_symbol(SymT.UNBACKED_FLOAT, 0)),
        ]
        argdefs = [ArgName("scale"), ArgName("runtime_scale")]
        with V.set_graph_handler(FakeGraph()):
            self.assertEqual(
                triton_utils.signature_to_meta(
                    signature, size_dtype=None, argdefs=argdefs
                ),
                {"scale": "fp64", "runtime_scale": "fp64"},
            )
            self.assertEqual(
                triton_utils.signature_to_meta(
                    signature,
                    size_dtype=None,
                    argdefs=argdefs,
                    use_fp64_for_python_float=False,
                ),
                {"scale": "fp32", "runtime_scale": "fp32"},
            )

    @unittest.skipUnless(HAS_GPU_AND_TRITON, "requires GPU and Triton")
    @patch("torch._inductor.codegen.triton.device_supports_fp64", return_value=False)
    @patch(
        "torch._inductor.codegen.triton_utils.device_supports_fp64",
        return_value=False,
    )
    def test_no_fp64_in_kernel_when_device_unsupported(self, mock1, mock2):
        """Compile a kernel with dynamic shape division to verify fp64 is
        downgraded to fp32 in the generated Triton kernel body when the device
        does not support fp64.

        ``x / x.shape[0]`` with dynamic shapes keeps the shape as a sympy
        symbol, so the int-to-float cast goes through TritonPrinter._print_ToFloat
        which respects device_supports_fp64().
        """
        import re

        def div_by_shape(x):
            return x / x.shape[0]

        x = torch.randn(16, 16, device=GPU_TYPE)
        compiled = torch.compile(div_by_shape, dynamic=True)
        _, kernels = run_and_get_kernels(compiled, x, remove_quote=True)
        # Extract only the function body (after ``def triton_...:``) to avoid
        # matching metadata like ``'has_fp64': True`` in DeviceProperties.
        matched_kernel_body = False
        for kernel in kernels:
            m = re.search(r"def triton_\w+\([^)]*\):\n(.*)", kernel, re.DOTALL)
            if m:
                matched_kernel_body = True
                body = m.group(1)
                self.assertNotIn("tl.float64", body)
                self.assertIn("tl.float32", body)
        self.assertTrue(
            matched_kernel_body,
            "Expected at least one generated Triton kernel body to match",
        )

    @unittest.skipUnless(torch.version.hip is not None, "pointer_range_32 is HIP-only")
    @unittest.skipUnless(HAS_GPU_AND_TRITON, "requires GPU and Triton")
    def test_pointer_range_not_in_user_defined_triton_kernel(self):
        """User-defined Triton kernels should not get pointer_range_32."""
        import triton
        import triton.language as tl

        @triton.jit
        def add_kernel(in_ptr0, in_ptr1, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(axis=0)
            offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(in_ptr0 + offsets, mask=mask)
            y = tl.load(in_ptr1 + offsets, mask=mask)
            tl.store(out_ptr + offsets, x + y, mask=mask)

        def fn(x, y):
            out = torch.empty_like(x)
            n = x.numel()

            def grid(meta):
                return (triton.cdiv(n, meta["BLOCK_SIZE"]),)

            add_kernel[grid](x, y, out, n, BLOCK_SIZE=128)
            return out

        x = torch.randn(64, 64, device=GPU_TYPE, dtype=torch.bfloat16)
        y = torch.randn(64, 64, device=GPU_TYPE, dtype=torch.bfloat16)
        _, code = run_and_get_code(torch.compile(fn), x, y)
        code_str = " ".join(code)
        self.assertNotIn("tt.pointer_range", code_str)

    @unittest.skipUnless(
        HAS_GPU_AND_TRITON or (HAS_CPU and has_triton_package()),
        "requires CPU or GPU Triton",
    )
    def test_user_defined_triton_kernel_non_builtin_constexpr(self):
        import triton
        import triton.language as tl

        @triton.jit
        def add_constexpr_kernel(
            x,
            out,
            n_elements,
            cfg: tl.constexpr,
            BLOCK_SIZE: tl.constexpr,
        ):
            pid = tl.program_id(axis=0)
            offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            values = tl.load(x + offsets, mask=mask)
            tl.store(out + offsets, values + cfg.nested.offset, mask=mask)

        # Dynamo graph-breaks on a dataclass argument of a directly called
        # kernel, so route the call through a triton_op: the kernel reaches
        # inductor via the HOP side table and fullgraph=True guarantees the
        # constexpr is compiled into the kernel rather than run in eager.
        @torch.library.triton_op("test_codegen_triton::add_cfg", mutates_args={})
        def add_cfg(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            n_elements = x.numel()

            def grid(meta):
                return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

            torch.library.wrap_triton(add_constexpr_kernel)[grid](
                x,
                out,
                n_elements,
                cfg=UserDefinedTritonKernelNestedConfig(
                    nested=UserDefinedTritonKernelConfigNamespace.Nested(offset=2)
                ),
                BLOCK_SIZE=128,
            )
            return out

        def fn(x):
            return add_cfg(x)

        device = GPU_TYPE if HAS_GPU_AND_TRITON else "cpu"
        x = torch.randn(1024, device=device)
        actual, code = run_and_get_code(torch.compile(fn, fullgraph=True), x)
        self.assertEqual(actual, x + 2)
        code_str = " ".join(code)
        self.assertIn("add_constexpr_kernel", code_str)
        self.assertIn("UserDefinedTritonKernelNestedConfig(nested=", code_str)

    @unittest.skipUnless(
        HAS_GPU_AND_TRITON or (HAS_CPU and has_triton_package()),
        "requires CPU or GPU Triton",
    )
    @parametrize(
        "op_name, cfg",
        (
            subtest(
                (
                    "nested_point",
                    UserDefinedTritonKernelConfigNamespace.Point(offset=2),
                ),
                name="nested_point",
            ),
            subtest(
                ("launcher_shadow", LauncherScopeShadowConfig(offset=3)),
                name="launcher_shadow",
            ),
        ),
    )
    def test_namedtuple_constexpr_launcher_does_not_reimport_types(self, op_name, cfg):
        # The generated launcher references constants only via injected
        # _constexpr_N object bindings, never by type name, so a class-nested
        # namedtuple (bare class name in its repr) and a type named like a
        # launcher scope binding ("runner") must both compile and run.
        import triton
        import triton.language as tl

        @triton.jit
        def add_offset_kernel(
            x,
            out,
            n_elements,
            cfg: tl.constexpr,
            BLOCK_SIZE: tl.constexpr,
        ):
            pid = tl.program_id(axis=0)
            offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            values = tl.load(x + offsets, mask=mask)
            tl.store(out + offsets, values + cfg.offset, mask=mask)

        @torch.library.triton_op(f"test_codegen_triton::{op_name}", mutates_args={})
        def op(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            n_elements = x.numel()

            def grid(meta):
                return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

            torch.library.wrap_triton(add_offset_kernel)[grid](
                x, out, n_elements, cfg=cfg, BLOCK_SIZE=128
            )
            return out

        device = GPU_TYPE if HAS_GPU_AND_TRITON else "cpu"
        x = torch.randn(1024, device=device)
        compiled = torch.compile(lambda x: op(x), fullgraph=True)
        actual, code = run_and_get_code(compiled, x)
        self.assertEqual(actual, x + cfg.offset)
        self.assertIn(f"{type(cfg).__qualname__}._make(", " ".join(code))

    @unittest.skipUnless(
        HAS_GPU_AND_TRITON or (HAS_CPU and has_triton_package()),
        "requires CPU or GPU Triton",
    )
    def test_hidden_field_constexpr_value_fidelity(self):
        # A config whose repr omits a field only renders when the omitted state
        # still matches what the constructor call rebuilds; otherwise the
        # kernel would silently compute with wrong constants.
        import triton
        import triton.language as tl

        @triton.jit
        def scaled_offset_kernel(
            x, out, n_elements, cfg: tl.constexpr, BLOCK_SIZE: tl.constexpr
        ):
            pid = tl.program_id(axis=0)
            offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            values = tl.load(x + offsets, mask=mask)
            tl.store(out + offsets, values + cfg.offset * cfg.scale, mask=mask)

        def make_op(name, cfg):
            @torch.library.triton_op(f"test_codegen_triton::{name}", mutates_args={})
            def op(x: torch.Tensor) -> torch.Tensor:
                out = torch.empty_like(x)
                n_elements = x.numel()

                def grid(meta):
                    return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

                torch.library.wrap_triton(scaled_offset_kernel)[grid](
                    x, out, n_elements, cfg=cfg, BLOCK_SIZE=128
                )
                return out

            return op

        device = GPU_TYPE if HAS_GPU_AND_TRITON else "cpu"
        x = torch.randn(1024, device=device)
        faithful = UserDefinedTritonKernelHiddenDefaultConfig(offset=3)
        op = make_op("scaled_offset_faithful", faithful)
        compiled = torch.compile(lambda t, op=op: op(t), fullgraph=True)
        self.assertEqual(compiled(x), x + 3)
        unfaithful = UserDefinedTritonKernelHiddenDefaultConfig(offset=3, scale=7)
        op = make_op("scaled_offset_unfaithful", unfaithful)
        compiled = torch.compile(lambda t, op=op: op(t), fullgraph=True)
        with self.assertRaisesRegex(BackendCompilerFailed, "cannot be written into"):
            compiled(x)

    @unittest.skipUnless(
        HAS_GPU_AND_TRITON or (HAS_CPU and has_triton_package()),
        "requires CPU or GPU Triton",
    )
    def test_object_constexpr_default_in_user_defined_triton_kernel(self):
        # The def-time default expression is evaluated when the generated
        # module re-execs the spliced kernel def (even when the caller passes
        # cfg explicitly), and it references the config type by its bare root
        # name, so the module must bind it via `from module import Root as
        # Root`.
        import triton
        import triton.language as tl

        @triton.jit
        def add_default_cfg_kernel(
            x,
            out,
            n_elements,
            BLOCK_SIZE: tl.constexpr,
            cfg: tl.constexpr = UserDefinedTritonKernelDefaultArgConfig(),
        ):
            pid = tl.program_id(axis=0)
            offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            values = tl.load(x + offsets, mask=mask)
            tl.store(out + offsets, values + cfg.offset, mask=mask)

        @torch.library.triton_op(
            "test_codegen_triton::add_default_cfg", mutates_args={}
        )
        def add_default_cfg(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            n_elements = x.numel()

            def grid(meta):
                return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

            torch.library.wrap_triton(add_default_cfg_kernel)[grid](
                x,
                out,
                n_elements,
                BLOCK_SIZE=128,
                cfg=UserDefinedTritonKernelDefaultArgConfig(offset=5),
            )
            return out

        device = GPU_TYPE if HAS_GPU_AND_TRITON else "cpu"
        x = torch.randn(1024, device=device)
        compiled = torch.compile(lambda t: add_default_cfg(t), fullgraph=True)
        actual, code = run_and_get_code(compiled, x)
        self.assertEqual(actual, x + 5)
        root = "UserDefinedTritonKernelDefaultArgConfig"
        module = UserDefinedTritonKernelDefaultArgConfig.__module__
        self.assertIn(f"from {module} import {root} as {root}", " ".join(code))

    @unittest.skipUnless(
        HAS_GPU_AND_TRITON or (HAS_CPU and has_triton_package()),
        "requires CPU or GPU Triton",
    )
    def test_user_defined_triton_kernel_python_float_arg_signature_matches_triton(self):
        import triton
        import triton.language as tl
        from triton.runtime.jit import mangle_type

        @triton.jit
        def scale_kernel(in_ptr, out_ptr, n_elements, scale, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(axis=0)
            offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(in_ptr + offsets, mask=mask)
            tl.store(out_ptr + offsets, x * scale, mask=mask)

        def fn(x):
            out = torch.empty_like(x)
            n = x.numel()

            def grid(meta):
                return (triton.cdiv(n, meta["BLOCK_SIZE"]),)

            scale_kernel[grid](x, out, n, 0.5, BLOCK_SIZE=128)
            return out

        device = GPU_TYPE if HAS_GPU_AND_TRITON else "cpu"
        x = torch.randn(64, 64, device=device)
        result, code = run_and_get_code(torch.compile(fn), x)
        self.assertEqual(result, x * 0.5)
        code_str = " ".join(code)
        expected_signature = mangle_type(0.5)
        self.assertIn(f"'scale': '{expected_signature}'", code_str)
        if expected_signature != "fp64":
            self.assertNotIn("'scale': 'fp64'", code_str)

    def test_imports_for_benchmark_kernel_multiline_get_raw_stream(self):
        # Regression: a backend whose import_get_raw_stream_as returns a
        # multi-line snippet (e.g. the CPU override, which MTIA uses) must not
        # break the textwrap.dedent of the benchmark-kernel imports. Formatting
        # before dedenting used to leave the imports indented (IndentationError).
        # TritonKernel and ComboKernel carry identical copies of this helper.
        from torch._inductor.codegen.triton_combo_kernel import ComboKernel

        class FakeDeviceOps:
            def import_get_raw_stream_as(self, name):
                return f"def {name}(_):\n    return 0"

        class FakeGraph:
            device_ops = FakeDeviceOps()

        for kernel_cls in (TritonKernel, ComboKernel):
            with V.set_graph_handler(FakeGraph()):
                # imports_for_benchmark_kernel does not use self.
                imports = kernel_cls.imports_for_benchmark_kernel(None)
            # Compiles without IndentationError and the top-level imports stay at
            # column 0 (they would be indented if dedent ran after substitution).
            compile(imports, "<benchmark_kernel_imports>", "exec")
            self.assertIn("\nfrom torch._dynamo.testing import rand_strided\n", imports)
            self.assertIn("\nimport torch\n", imports)

    def test_constexpr_source_module_enum_rendering(self):
        from torch._inductor.codegen.wrapper import _render_constexpr_mappings

        self.assertEqual(
            _constexpr_source(plistlib.FMT_XML),
            (
                "__inductor_constexpr_module_0.PlistFormat['FMT_XML']",
                ["import plistlib as __inductor_constexpr_module_0"],
            ),
        )
        (rendered,), imports = _render_constexpr_mappings(
            [{"FORMAT": plistlib.FMT_XML}]
        )
        self.assertEqual(
            repr(rendered["FORMAT"]),
            "__inductor_constexpr_module_0.PlistFormat['FMT_XML']",
        )
        self.assertEqual(imports, ["import plistlib as __inductor_constexpr_module_0"])

    def test_constexpr_source_constructor_repr_config(self):
        config = UserDefinedTritonKernelNestedConfig(
            nested=UserDefinedTritonKernelConfigNamespace.Nested(offset=2)
        )
        source, imports = _constexpr_source(config)
        alias = "__inductor_constexpr_module_0"
        self.assertEqual(
            source,
            f"{alias}.UserDefinedTritonKernelNestedConfig(nested={alias}.UserDefinedTritonKernelConfigNamespace.Nested(offset=2))",
        )
        self.assertEqual(imports, [f"import {type(config).__module__} as {alias}"])
        scope = {}
        exec("\n".join(imports), scope)
        self.assertEqual(eval(source, scope), config)

    def test_constexpr_source_constructor_repr_enum_field(self):
        config = UserDefinedTritonKernelEnumConfig(
            mode=UserDefinedTritonKernelConfigMode.FAST
        )
        source, _ = _constexpr_source(config)
        # The interchangeable IntEnum field goes through the stack's enum
        # normalization, not the field's raw repr.
        self.assertEqual(
            source,
            "__inductor_constexpr_module_0.UserDefinedTritonKernelEnumConfig(mode=1)",
        )

    @parametrize(
        "config_type",
        (
            subtest(UserDefinedAttrsLikeConfig, name="attrs"),
            subtest(UserDefinedPydanticLikeConfig, name="pydantic"),
        ),
    )
    def test_constexpr_source_constructor_repr_protocols(self, config_type):
        nested = UserDefinedTritonKernelConfigNamespace.Nested(offset=2)
        source, imports = _constexpr_source(config_type(nested=nested))
        alias = "__inductor_constexpr_module_0"
        self.assertEqual(
            source,
            f"{alias}.{config_type.__name__}(nested={alias}.UserDefinedTritonKernelConfigNamespace.Nested(offset=2))",
        )
        scope = {}
        exec("\n".join(imports), scope)
        rebuilt = eval(source, scope)
        self.assertIs(type(rebuilt), config_type)
        self.assertEqual(rebuilt.nested, nested)

    def test_constexpr_source_attrs_private_field_uses_alias(self):
        nested = UserDefinedTritonKernelConfigNamespace.Nested(offset=2)
        config = UserDefinedAttrsPrivateFieldConfig(nested=nested)
        source, imports = _constexpr_source(config)
        alias = "__inductor_constexpr_module_0"
        self.assertEqual(
            source,
            f"{alias}.UserDefinedAttrsPrivateFieldConfig(nested={alias}.UserDefinedTritonKernelConfigNamespace.Nested(offset=2))",
        )
        scope = {}
        exec("\n".join(imports), scope)
        self.assertEqual(eval(source, scope), config)

    def test_constexpr_source_constructor_repr_types(self):
        # Types with a constructor-style repr over literal arguments but no
        # field protocol (Fraction, Decimal) render through the module alias and
        # rebuild equal, as they did before the field-protocol renderer.
        from decimal import Decimal
        from fractions import Fraction

        for value in (Fraction(1, 2), Decimal("1.5")):
            source, imports = _constexpr_source(value)
            scope = {}
            exec("\n".join(imports), scope)
            rebuilt = eval(source, scope)
            self.assertIs(type(rebuilt), type(value))
            self.assertEqual(rebuilt, value)

    def test_constexpr_source_repr_args_without_eq(self):
        # A pydantic-style __repr_args__ type is verified field-wise, so it does
        # not need an __eq__ of its own.
        nested = UserDefinedTritonKernelConfigNamespace.Nested(offset=2)
        source, imports = _constexpr_source(
            UserDefinedPydanticLikeNoEqConfig(nested=nested)
        )
        scope = {}
        exec("\n".join(imports), scope)
        rebuilt = eval(source, scope)
        self.assertIs(type(rebuilt), UserDefinedPydanticLikeNoEqConfig)
        self.assertEqual(rebuilt.nested, nested)

    def test_constexpr_source_repr_args_hidden_state_declines(self):
        # __repr_args__ hides `hidden`; the type's own __eq__ sees it, so a
        # non-default hidden value must decline rather than render as the
        # default and silently miscompute.
        from torch._inductor.codegen.wrapper import _render_constexpr_mappings

        nested = UserDefinedTritonKernelConfigNamespace.Nested(offset=2)
        with self.assertRaisesRegex(RuntimeError, "not equal to the original"):
            _render_constexpr_mappings(
                [{"CFG": UserDefinedPydanticLikeConfig(nested=nested, hidden="secret")}]
            )
        # The default hidden value still renders (equal after rebuild).
        source, _ = _constexpr_source(UserDefinedPydanticLikeConfig(nested=nested))
        self.assertIn("UserDefinedPydanticLikeConfig(nested=", source)

    def test_constexpr_decline_names_evaluation_error(self):
        # A constructor repr over a name the generated module cannot bind
        # declines with the evaluation error, not a guess about hidden fields.
        from torch._inductor.codegen.wrapper import _render_constexpr_mappings

        with self.assertRaisesRegex(RuntimeError, "evaluating it raised NameError"):
            _render_constexpr_mappings([{"CFG": UserDefinedBareNestedReprConfig(1)}])

    def test_restore_degraded_kwargs_requires_agreement(self):
        # A cached raw value that several candidates serialize to is only
        # restored when they agree on the typed value; a Mode.A vs raw 1 split
        # must re-autotune instead of picking by candidate order.
        from types import SimpleNamespace

        from torch._inductor.runtime.autotune_cache import _restore_degraded_kwargs

        class Mode(Enum):
            A = 1

        for order in (
            [
                SimpleNamespace(kwargs={"MODE": 1}),
                SimpleNamespace(kwargs={"MODE": Mode.A}),
            ],
            [
                SimpleNamespace(kwargs={"MODE": Mode.A}),
                SimpleNamespace(kwargs={"MODE": 1}),
            ],
        ):
            self.assertFalse(_restore_degraded_kwargs({"MODE": 1, "BLOCK": 8}, order))
        # Agreement is structural: (1, 2) and (1.0, 2) both serialize to the
        # cached [1.0, 2] but Triton specializes int vs float, so a split inside
        # a tuple re-autotunes too, in either order.
        for order in (
            [
                SimpleNamespace(kwargs={"S": (1, 2)}),
                SimpleNamespace(kwargs={"S": (1.0, 2)}),
            ],
            [
                SimpleNamespace(kwargs={"S": (1.0, 2)}),
                SimpleNamespace(kwargs={"S": (1, 2)}),
            ],
        ):
            self.assertFalse(_restore_degraded_kwargs({"S": [1.0, 2]}, order))
        best = {"MODE": 1, "SHAPE": [2, 3], "BLOCK": 8, "num_warps": 4}
        agreeing = [
            SimpleNamespace(kwargs={"MODE": Mode.A, "SHAPE": (2, 3), "BLOCK": b})
            for b in (8, 16)
        ]
        self.assertTrue(_restore_degraded_kwargs(best, agreeing))
        self.assertIs(best["MODE"], Mode.A)
        self.assertEqual(best["SHAPE"], (2, 3))
        self.assertEqual(best["num_warps"], 4)

    def test_constexpr_decline_detail_names_cause(self):
        from torch._inductor.codegen.wrapper import _render_constexpr_mappings

        class Local(list):
            __slots__ = ()

        with self.assertRaisesRegex(RuntimeError, "subclasses list"):
            _render_constexpr_mappings([{"CFG": Local([1])}])
        with self.assertRaisesRegex(RuntimeError, "repr-visible but init=False"):
            _render_constexpr_mappings(
                [{"CFG": UserDefinedTritonKernelNonInitConfig(offset=2)}]
            )
        with self.assertRaisesRegex(RuntimeError, "not equal to the original"):
            _render_constexpr_mappings(
                [{"CFG": UserDefinedTritonKernelHiddenDefaultConfig(offset=3, scale=7)}]
            )

    def test_hashable_constexpr_key(self):
        from torch._inductor.codegen.wrapper import _hashable_constexpr_key

        key = _hashable_constexpr_key({"a": [1, {2, 3}], "b": (4, [5])})
        hash(key)
        self.assertEqual(
            key, _hashable_constexpr_key({"a": [1, {3, 2}], "b": (4, [5])})
        )
        self.assertNotEqual(
            key, _hashable_constexpr_key({"a": [1, {2, 3}], "b": (4, (5,))})
        )
        self.assertIs(_hashable_constexpr_key(7), 7)

    def test_constexpr_nested_nan_decline_names_nan(self):
        from torch._inductor.codegen.wrapper import _render_constexpr_mappings

        # A NaN nested inside a container declines the whole value; the error
        # must still name NaN as the cause rather than fall through to the
        # generic message.
        for value in ([1.0, float("nan")], {"k": (float("nan"),)}):
            with self.assertRaisesRegex(RuntimeError, "NaN constexprs are rejected"):
                _render_constexpr_mappings([{"CFG": value}])

    def test_constexpr_source_constructor_repr_declines(self):
        from torch._inductor.codegen.wrapper import _render_constexpr_mappings

        @dataclasses.dataclass(frozen=True)
        class LocalConfig:
            offset: int

        # A local (unimportable) config type declines with the loud error,
        # which names the offending scope.
        with self.assertRaisesRegex(
            RuntimeError,
            r"cannot be written into.*LocalConfig is defined inside a function",
        ):
            _render_constexpr_mappings([{"CFG": LocalConfig(offset=2)}])
        # A repr-visible init=False field cannot be passed as a constructor
        # argument; a hidden required init parameter makes the constructor
        # call fail. Both decline instead of crashing the generated module.
        self.assertIsNone(
            _constexpr_source(UserDefinedTritonKernelNonInitConfig(offset=2))
        )
        self.assertIsNone(
            _constexpr_source(UserDefinedTritonKernelHiddenConfig(2, object()))
        )

    def test_constexpr_source_declines_value_unfaithful_configs(self):
        from torch._inductor.codegen.wrapper import _render_constexpr_mappings

        # The repr omits scale, so evaluating it would silently rebuild the
        # config with the default scale=1 instead of 7; rendering must decline
        # into the loud error rather than compile with wrong constants.
        unfaithful = UserDefinedTritonKernelHiddenDefaultConfig(offset=3, scale=7)
        with self.assertRaisesRegex(RuntimeError, "cannot be written into"):
            _render_constexpr_mappings([{"CFG": unfaithful}])
        # __post_init__ coerces again on rebuild: repr shows offset=6, but
        # evaluating Cfg(offset=6) produces offset=12.
        coercing = UserDefinedTritonKernelCoercingConfig(offset=3)
        self.assertEqual(coercing.offset, 6)
        self.assertIsNone(_constexpr_source(coercing))
        # A hidden field still holding its default rebuilds faithfully.
        faithful = UserDefinedTritonKernelHiddenDefaultConfig(offset=3)
        source, imports = _constexpr_source(faithful)
        scope = {}
        exec("\n".join(imports), scope)
        self.assertEqual(eval(source, scope), faithful)

    def test_constexpr_source_declines_self_referential_config(self):
        from torch._inductor.codegen.wrapper import _render_constexpr_mappings

        cfg = UserDefinedTritonKernelSelfReferentialConfig()
        cfg.child = cfg
        with self.assertRaisesRegex(RuntimeError, "cannot be written into"):
            _render_constexpr_mappings([{"CFG": cfg}])

    def test_constexpr_source_constructs_each_object_once(self):
        inner = UserDefinedTritonKernelCountingConfig()
        cfg = UserDefinedTritonKernelCountingConfig(
            UserDefinedTritonKernelCountingConfig(inner)
        )
        UserDefinedTritonKernelCountingConfig.constructed = 0
        result = _constexpr_source(cfg)
        self.assertIsNotNone(result)
        # Rendering verifies by evaluating the source once at the top level, so
        # compile-time constructor work stays linear in the number of objects.
        self.assertEqual(UserDefinedTritonKernelCountingConfig.constructed, 3)

    @parametrize(
        "value, expected",
        (
            subtest((IntEnum("Mode", {"EVEN": 2}).EVEN, 2), name="int_enum"),
            subtest((IntFlag("Flags", {"A": 1}).A, 1), name="int_flag"),
            subtest((Enum("TextMode", {"A": "a"}, type=str).A, "a"), name="str_enum"),
            subtest(
                (Enum("FloatMode", {"A": 1.5}, type=float).A, 1.5), name="float_enum"
            ),
        ),
    )
    def test_constexpr_constant_enum_interchange(self, value, expected):
        from torch._inductor.codegen.wrapper import _constexpr_constant

        self.assertEqual(_constexpr_constant(value), expected)

    def test_constexpr_constant_namedtuple_recursion(self):
        from torch._inductor.codegen.wrapper import _constexpr_constant

        class Mode(IntEnum):
            EVEN = 2

        Pair = namedtuple("Pair", ("left", "right"))
        nested = {Mode.EVEN: [Mode.EVEN, (Mode.EVEN,), Pair(Mode.EVEN, 3)]}
        self.assertEqual(_constexpr_constant(nested), {2: [2, (2,), Pair(2, 3)]})

    def test_constexpr_constant_namedtuple_with_shifting_new(self):
        from torch._inductor.codegen.wrapper import _constexpr_constant

        class Shifted(namedtuple("ShiftedBase", ("value",))):
            __slots__ = ()

            def __new__(cls, value):
                return super().__new__(cls, value + 1)

        shifted = Shifted(1)
        self.assertEqual(_constexpr_constant(shifted).value, 2)

    def test_constexpr_source_declines_local_enum(self):
        from torch._inductor.codegen.wrapper import _render_constexpr_mappings

        class Local(Enum):
            VALUE = 1

        with self.assertRaisesRegex(
            RuntimeError, r"cannot be written into.*Local is defined inside a function"
        ):
            _render_constexpr_mappings([{"MODE": Local.VALUE}])

    def test_constexpr_source_declines_main_module_enum(self):
        from torch._inductor.codegen.wrapper import _render_constexpr_mappings

        # An enum living in __main__ (an enum defined at the top of a user
        # script) is not importable from the generated module; decline rather
        # than emit a dangling import, and tell the user where it lives.
        with (
            patch.object(plistlib.PlistFormat, "__module__", "__main__"),
            self.assertRaisesRegex(
                RuntimeError,
                r"cannot be written into.*PlistFormat is defined in __main__",
            ),
        ):
            _render_constexpr_mappings([{"FORMAT": plistlib.FMT_XML}])

    def test_constexpr_source_names_nested_unimportable_scope(self):
        from torch._inductor.codegen.wrapper import _render_constexpr_mappings

        @dataclasses.dataclass(frozen=True)
        class LocalNested:
            offset: int

        # The unimportable type may sit inside an otherwise renderable config.
        cfg = UserDefinedTritonKernelNestedConfig(nested=LocalNested(offset=2))
        with self.assertRaisesRegex(
            RuntimeError, r"cannot be written into.*LocalNested is defined inside"
        ):
            _render_constexpr_mappings([{"CFG": cfg}])

    def test_constexpr_source_ordered_set_round_trip(self):
        from torch.utils._ordered_set import OrderedSet

        # A set subclass's type and iteration order are semantic: the source
        # must reconstruct the exact type in order, not a sorted builtin set.
        ordered = OrderedSet([2, 1])
        source, imports = _constexpr_source(ordered)
        scope = {}
        exec("\n".join(imports), scope)
        reconstructed = eval(source, scope)
        self.assertIs(type(reconstructed), OrderedSet)
        self.assertEqual(list(reconstructed), [2, 1])

    def test_constexpr_source_declines_set_subclasses(self):
        from torch.utils._ordered_set import OrderedSet

        class LocalSet(set):
            pass

        self.assertIsNone(_constexpr_source(LocalSet({1})))

        # Only OrderedSet exactly reconstructs: other subclasses -- even fully
        # importable ones, which the reference-resolution path alone would
        # accept -- decline rather than emit hash-order-nondeterministic or
        # constructor-assuming source.
        import torch.utils._ordered_set as ordered_set_module

        class OrderedSubSet(OrderedSet):
            pass

        OrderedSubSet.__module__ = "torch.utils._ordered_set"
        OrderedSubSet.__qualname__ = "OrderedSubSet"
        with patch.object(
            ordered_set_module, "OrderedSubSet", OrderedSubSet, create=True
        ):
            self.assertIsNone(_constexpr_source(OrderedSubSet([1])))

    @parametrize(
        "base, sample",
        (
            subtest((dict, {"a": 1}), name="dict"),
            subtest((list, [1]), name="list"),
            subtest((tuple, (1,)), name="tuple"),
        ),
    )
    def test_constexpr_container_subclass_declines(self, base, sample):
        # Consistent with the set-subclass policy: emitting a container display
        # for a dict/list/tuple subclass would silently drop its type, so these
        # decline with the clear error instead of coercing to plain builtins.
        from torch._inductor.codegen.wrapper import (
            _constexpr_constant,
            _render_constexpr_mappings,
        )

        class Local(base):
            __slots__ = ()

        value = Local(sample)
        self.assertIs(_constexpr_constant(value), value)
        with self.assertRaisesRegex(RuntimeError, "cannot be written into"):
            _render_constexpr_mappings([{"MODE": value}])

    @parametrize(
        "value, expected",
        (
            subtest((64, ("64", [])), name="int"),
            subtest((float("inf"), ("float('inf')", [])), name="positive_inf"),
            subtest((float("-inf"), ("float('-inf')", [])), name="negative_inf"),
            subtest((float("nan"), None), name="nan"),
            subtest((-float("nan"), None), name="negative_nan"),
            subtest(
                (
                    torch.float32,
                    (
                        "__inductor_constexpr_module_0.float32",
                        ["import torch as __inductor_constexpr_module_0"],
                    ),
                ),
                name="torch_dtype",
            ),
            subtest((torch.Size((128,)), ("(128,)", [])), name="torch_size"),
            subtest(
                ((torch.Size((2, 3)), 1), ("((2, 3), 1)", [])),
                name="nested_torch_size",
            ),
            subtest((slice(1, 5, 2), ("slice(1, 5, 2)", [])), name="slice"),
            subtest((range(3), ("range(0, 3)", [])), name="range"),
            subtest(
                (frozenset({1, 2}), ("frozenset((1, 2,))", [])),
                name="frozenset",
            ),
            subtest(({2, 1}, ("{1, 2}", [])), name="set"),
            subtest((set(), ("set()", [])), name="empty_set"),
        ),
    )
    def test_constexpr_builtin_source(self, value, expected):
        self.assertEqual(_constexpr_source(value), expected)

    def test_constexpr_bytearray_source(self):
        value = bytearray(b"a\x00'\"b")
        source, imports = _constexpr_source(value)
        self.assertEqual(imports, [])
        rebuilt = eval(source)
        self.assertIs(type(rebuilt), bytearray)
        self.assertEqual(rebuilt, value)

    def test_source_literal_eq_hash(self):
        from torch._inductor.codegen.wrapper import _SourceLiteral

        left, right = _SourceLiteral("mod.X"), _SourceLiteral("mod.X")
        self.assertEqual(left, right)
        self.assertEqual(hash(left), hash(right))
        self.assertNotEqual(left, _SourceLiteral("mod.Y"))
        self.assertNotEqual(left, "mod.X")

    @unittest.skipUnless(has_triton_package(), "requires Triton")
    def test_constexpr_triton_dtype_source(self):
        import triton.language as tl

        self.assertEqual(
            _constexpr_source(tl.float32),
            (
                "__inductor_constexpr_module_0.float32",
                ["import triton.language as __inductor_constexpr_module_0"],
            ),
        )
        self.assertEqual(
            _constexpr_source((tl.float32,)),
            (
                "(__inductor_constexpr_module_0.float32,)",
                ["import triton.language as __inductor_constexpr_module_0"],
            ),
        )

    def test_constexpr_enum_imports_do_not_collide(self):
        import uuid

        from torch._inductor.codegen.wrapper import _render_constexpr_mappings

        values = (plistlib.FMT_XML, uuid.SafeUUID.safe)
        rendered, imports = _render_constexpr_mappings(
            [{"LEFT": values[0]}, {"RIGHT": values[1]}]
        )
        scope = {}
        exec("\n".join(imports), scope)
        left, right = (eval(repr(mapping), scope) for mapping in rendered)
        self.assertIs(left["LEFT"], values[0])
        self.assertIs(right["RIGHT"], values[1])

        class Bits(Flag):
            LEFT = 1
            RIGHT = 2

        self.assertIsNone(_constexpr_source(Bits.LEFT | Bits.RIGHT))

    def test_launcher_constexpr_scope_avoids_argument_names(self):
        from torch._inductor.runtime.triton_heuristics import CompileResult

        class Mode(Enum):
            VALUE = 1

        result = CompileResult(
            None,
            SimpleNamespace(kwargs={}),
            {
                "constants": {"MODE": Mode.VALUE, "LIMIT": float("inf")},
                "signature": {
                    "x": "i32",
                    "_constexpr_0": "i32",
                    "MODE": "i32",
                    "LIMIT": "fp32",
                },
            },
            {},
        )
        with patch(
            "torch._inductor.runtime.triton_heuristics.triton_version_uses_attrs_dict",
            return_value=True,
        ):
            call_args, def_args, _, constant_scope = result._get_arg_lists(
                ["x", "_constexpr_0", "MODE", "LIMIT"], {2, 3}
            )
        self.assertEqual(def_args, ["x", "_constexpr_0"])
        self.assertEqual(
            call_args, ["x", "_constexpr_0", "_constexpr_1", "_constexpr_2"]
        )
        self.assertIs(constant_scope["_constexpr_1"], Mode.VALUE)
        self.assertEqual(constant_scope["_constexpr_2"], float("inf"))

    @unittest.skipUnless(HAS_GPU_AND_TRITON, "requires GPU and Triton")
    def test_enum_constexpr_in_user_defined_triton_kernel(self):
        """Custom Triton kernel with IntEnum constexpr generates valid code."""
        import triton
        import triton.language as tl

        class Mode(IntEnum):
            ADD = 1
            MUL = 2

        @triton.jit
        def enum_kernel(
            in_ptr, out_ptr, numel, MODE: tl.constexpr, BLOCK_SIZE: tl.constexpr
        ):
            offsets = tl.arange(0, BLOCK_SIZE)
            mask = offsets < numel
            x = tl.load(in_ptr + offsets, mask=mask)
            if MODE == 1:
                output = x + 1
            else:
                output = x * 2
            tl.store(out_ptr + offsets, output, mask=mask)

        def fn(x):
            y = torch.empty_like(x)
            enum_kernel[(1,)](x, y, x.numel(), Mode.ADD, 256)
            return y

        x = torch.randn(128, device=GPU_TYPE)
        fn_c = torch.compile(fn)
        res, code = run_and_get_code(fn_c, x)
        self.assertEqual(fn(x), res)
        # Verify generated code doesn't contain invalid Enum repr like <Mode.ADD: 1>
        self.assertNotIn("<Mode.", code[0])

    @unittest.skipUnless(HAS_GPU_AND_TRITON, "requires GPU and Triton")
    def test_plain_enum_constexpr_in_cpp_wrapper(self):
        # The cpp_wrapper/AOTI path consumes the kernel metas returned by
        # define_user_defined_triton_kernel; those must carry real values (no
        # rendering placeholders) and still compile with a non-interchangeable
        # Enum constexpr.
        import triton
        import triton.language as tl

        @triton.jit
        def enum_kernel(
            in_ptr, out_ptr, numel, MODE: tl.constexpr, BLOCK_SIZE: tl.constexpr
        ):
            offsets = tl.arange(0, BLOCK_SIZE)
            mask = offsets < numel
            x = tl.load(in_ptr + offsets, mask=mask)
            if MODE.value == 1:
                output = x + 1
            else:
                output = x * 2
            tl.store(out_ptr + offsets, output, mask=mask)

        def fn(x):
            y = torch.empty_like(x)
            enum_kernel[(1,)](x, y, x.numel(), plistlib.FMT_XML, BLOCK_SIZE=256)
            return y

        x = torch.randn(128, device=GPU_TYPE)
        with inductor_config.patch(cpp_wrapper=True):
            res, code = run_and_get_code(torch.compile(fn), x)
        self.assertEqual(fn(x), res)
        for generated in code:
            self.assertNotIn("<PlistFormat.", generated)

    @unittest.skipUnless(HAS_GPU_AND_TRITON, "requires GPU and Triton")
    def test_enum_constexpr_in_autotune_config(self):
        import triton
        import triton.language as tl

        class Mode(IntEnum):
            ADD = 1
            MUL = 2

        @triton.autotune(
            configs=[
                triton.Config({"MODE": Mode.ADD, "BLOCK_SIZE": 64}, num_warps=4),
                triton.Config({"MODE": Mode.MUL, "BLOCK_SIZE": 128}, num_warps=4),
            ],
            key=[],
        )
        @triton.jit
        def enum_kernel(
            in_ptr, out_ptr, numel, MODE: tl.constexpr, BLOCK_SIZE: tl.constexpr
        ):
            offsets = tl.arange(0, BLOCK_SIZE)
            mask = offsets < numel
            x = tl.load(in_ptr + offsets, mask=mask)
            output = tl.where(MODE == 1, x + 1, x * 2)
            tl.store(out_ptr + offsets, output, mask=mask)

        def fn(x):
            output = torch.empty_like(x)

            def grid(meta):
                return (triton.cdiv(x.numel(), meta["BLOCK_SIZE"]),)

            enum_kernel[grid](x, output, x.numel())
            return output

        x = torch.randn(128, device=GPU_TYPE)
        actual, code = run_and_get_code(torch.compile(fn), x)
        self.assertEqual(actual, fn(x))
        self.assertNotIn("<Mode.", code[0])

    @unittest.skipUnless(HAS_GPU_AND_TRITON, "requires GPU and Triton")
    def test_plain_enum_constexpr_in_autotune_config(self):
        # A non-interchangeable Enum in an autotune config reaches the runtime as
        # a real Enum member; autotuning (with >1 config) then saves the winner to
        # the JSON autotune cache, which must store the underlying value.
        import triton
        import triton.language as tl

        @triton.autotune(
            configs=[
                triton.Config(
                    {"MODE": plistlib.FMT_XML, "BLOCK_SIZE": 64},
                    num_warps=4,
                ),
                triton.Config(
                    {"MODE": plistlib.FMT_BINARY, "BLOCK_SIZE": 128}, num_warps=4
                ),
            ],
            key=[],
        )
        @triton.jit
        def enum_kernel(
            in_ptr, out_ptr, numel, MODE: tl.constexpr, BLOCK_SIZE: tl.constexpr
        ):
            offsets = tl.arange(0, BLOCK_SIZE)
            mask = offsets < numel
            x = tl.load(in_ptr + offsets, mask=mask)
            output = x + 1
            tl.store(out_ptr + offsets, output, mask=mask)

        def fn(x):
            output = torch.empty_like(x)

            def grid(meta):
                return (triton.cdiv(x.numel(), meta["BLOCK_SIZE"]),)

            enum_kernel[grid](x, output, x.numel())
            return output

        x = torch.randn(128, device=GPU_TYPE)
        actual, code = run_and_get_code(torch.compile(fn), x)
        self.assertEqual(actual, fn(x))
        self.assertIn("PlistFormat['", code[0])

    @unittest.skipUnless(HAS_GPU_AND_TRITON, "requires GPU and Triton")
    def test_user_autotune_cache_not_stamped_coordesc(self):
        # Coordinate descent tuning never runs for USER_AUTOTUNE kernels, so their
        # cache entries must not claim found_by_coordesc: a warm load would skip
        # candidate matching and reconstruct a Config whose Enum kwargs degrade to
        # the raw JSON values.
        if inductor_config.force_disable_caches:
            self.skipTest("requires autotune caching enabled")
        if not inductor_config.autotune_local_cache:
            self.skipTest("requires the local autotune cache")
        import triton
        import triton.language as tl

        from torch._inductor.runtime.autotune_cache import AutotuneCache

        @triton.autotune(
            configs=[
                triton.Config(
                    {"CD_MODE": plistlib.FMT_XML, "BLOCK_SIZE": 64},
                    num_warps=4,
                ),
                triton.Config(
                    {"CD_MODE": plistlib.FMT_BINARY, "BLOCK_SIZE": 128}, num_warps=4
                ),
            ],
            key=[],
        )
        @triton.jit
        def enum_kernel(
            in_ptr, out_ptr, numel, CD_MODE: tl.constexpr, BLOCK_SIZE: tl.constexpr
        ):
            offsets = tl.arange(0, BLOCK_SIZE)
            mask = offsets < numel
            x = tl.load(in_ptr + offsets, mask=mask)
            output = x + 1
            tl.store(out_ptr + offsets, output, mask=mask)

        def fn(x):
            output = torch.empty_like(x)

            def grid(meta):
                return (triton.cdiv(x.numel(), meta["BLOCK_SIZE"]),)

            enum_kernel[grid](x, output, x.numel())
            return output

        x = torch.randn(128, device=GPU_TYPE)
        with (
            inductor_config.patch(coordinate_descent_tuning=True),
            patch.object(
                AutotuneCache, "save", autospec=True, side_effect=AutotuneCache.save
            ) as saves,
        ):
            self.assertEqual(torch.compile(fn)(x), fn(x))
        enum_saves = [c for c in saves.call_args_list if "CD_MODE" in c.args[1].kwargs]
        self.assertTrue(enum_saves)
        for call in enum_saves:
            self.assertFalse(call.kwargs.get("found_by_coordesc", False))

    @unittest.skipUnless(has_triton_package(), "requires Triton")
    def test_autotune_cache_matches_enum_config_kwargs(self):
        import json

        import triton

        from torch._inductor.runtime.autotune_cache import (
            _config_json_cacheable,
            _json_config_value,
            _load_cached_autotuning,
        )

        cfg = triton.Config(
            {"MODE": plistlib.FMT_XML, "BLOCK": 64},
            num_warps=4,
            num_stages=2,
        )
        self.assertTrue(_config_json_cacheable(cfg))
        data = {key: _json_config_value(val) for key, val in cfg.kwargs.items()}
        data.update({"num_warps": 4, "num_stages": 2, "configs_hash": "h"})
        best = json.loads(json.dumps(data))  # must be JSON-representable
        self.assertIs(_load_cached_autotuning(best, "h", [cfg], {}), cfg)

    @unittest.skipUnless(has_triton_package(), "requires Triton")
    @parametrize("via_enum", (True, False))
    def test_autotune_cache_declines_non_json_config_kwargs(self, via_enum):
        import triton

        from torch._inductor.runtime.autotune_cache import (
            _config_json_cacheable,
            AutotuneCache,
        )

        class SetMode(Enum):
            PAIR = frozenset({1, 2})

        value = SetMode.PAIR if via_enum else {1, 2}
        cfg = triton.Config({"MODE": value, "BLOCK": 64}, num_warps=4, num_stages=2)
        self.assertFalse(_config_json_cacheable(cfg))
        # save() must skip such a config entirely. A bare instance suffices: the
        # gate returns before any cache state is touched, so if the gate were
        # removed this call would fail on the missing attributes.
        cache = object.__new__(AutotuneCache)
        self.assertIsNone(cache.save(cfg, time_taken_ns=0))

    @unittest.skipUnless(has_triton_package(), "requires Triton")
    def test_autotune_cache_matches_tuple_config_kwargs(self):
        # Tuple kwargs (directly or as an Enum's value) serialize as JSON
        # lists; warm load must match the stored list back to the tuple-kwarg
        # candidate and return that exact Config object so the kernel sees a
        # real tuple, not the degraded list.
        import os
        import tempfile

        import triton

        from torch._inductor.remote_cache import LocalAutotuneCache
        from torch._inductor.runtime.autotune_cache import (
            _config_json_cacheable,
            _load_cached_autotuning,
            AutotuneCache,
        )

        class TupleMode(Enum):
            PAIR = (1, 2)

        kw = {"MODE": TupleMode.PAIR, "SHAPE": (2, 3), "NEST": [(1,)], "BLOCK": 64}
        cfg = triton.Config(dict(kw), num_warps=4, num_stages=2)
        self.assertTrue(_config_json_cacheable(cfg))

        with tempfile.TemporaryDirectory() as tmpdir:
            key = os.path.join(tmpdir, "kernel.best_config")
            local_cache = LocalAutotuneCache("local-autotune")
            cache = AutotuneCache(configs_hash="h", local_cache=(local_cache, key))
            cache.save(cfg, time_taken_ns=0)
            best = local_cache.get(key)
        self.assertIsNotNone(best)
        self.assertEqual(best["SHAPE"], [2, 3])
        loaded = _load_cached_autotuning(best, "h", [cfg], {})
        self.assertIs(loaded, cfg)
        self.assertEqual(loaded.kwargs, kw)
        self.assertIs(loaded.kwargs["MODE"], TupleMode.PAIR)
        self.assertIs(type(loaded.kwargs["SHAPE"]), tuple)

    @unittest.skipUnless(has_triton_package(), "requires Triton")
    def test_autotune_cache_coordesc_entry_restores_typed_kwargs(self):
        # A coordesc-stamped winner is reconstructed from JSON, so a tuple
        # constexpr comes back as a list; the candidates (which hold the tuple,
        # constant across configs) supply the typed value instead of forcing a
        # re-autotune on every warm start.
        import json

        import triton

        from torch._inductor.runtime.autotune_cache import _load_cached_autotuning

        cfg = triton.Config({"SHAPE": (2, 3), "BLOCK": 64}, num_warps=4, num_stages=2)
        data = {"SHAPE": [2, 3], "BLOCK": 128, "num_warps": 8, "num_stages": 2}
        data.update({"configs_hash": "h", "found_by_coordesc": True})
        best = json.loads(json.dumps(data))
        loaded = _load_cached_autotuning(
            best, "h", [cfg], {"coordinate_descent_tuning": True}
        )
        self.assertIsNotNone(loaded)
        self.assertTrue(loaded.found_by_coordesc)
        self.assertEqual(loaded.kwargs, {"SHAPE": (2, 3), "BLOCK": 128})
        self.assertIs(type(loaded.kwargs["SHAPE"]), tuple)

    @unittest.skipUnless(has_triton_package(), "requires Triton")
    def test_autotune_cache_degraded_no_match_reautotunes(self):
        # A stored winner matching no candidate normally reconstructs (it may
        # be a dynamically added config), but when some candidate kwarg
        # degrades under the JSON round-trip (tuple -> list) the miss may be a
        # serialization artifact and reconstruction would bake the degraded
        # value into the kernel; warm load must re-autotune instead.
        import json

        import triton

        from torch._inductor.runtime.autotune_cache import _load_cached_autotuning

        cfg = triton.Config({"SHAPE": (2, 3), "BLOCK": 64}, num_warps=4, num_stages=2)
        data = {"SHAPE": [4, 5], "BLOCK": 32, "num_warps": 4, "num_stages": 2}
        data["configs_hash"] = "h"
        best = json.loads(json.dumps(data))
        self.assertIsNone(_load_cached_autotuning(best, "h", [cfg], {}))

        # A plain Enum kwarg (identity __eq__) is degraded too; IntEnum and
        # str-mixin members ==-match their unwrapped values and are not.
        class Mode(Enum):
            A = 1

        cfg_enum = triton.Config({"MODE": Mode.A}, num_warps=4, num_stages=2)
        data = {"MODE": 3, "num_warps": 4, "num_stages": 2, "configs_hash": "h"}
        best = json.loads(json.dumps(data))
        self.assertIsNone(_load_cached_autotuning(best, "h", [cfg_enum], {}))

        # Without degraded candidate kwargs the same miss still reconstructs.
        plain = triton.Config({"BLOCK": 64}, num_warps=4, num_stages=2)
        data = {"BLOCK": 32, "num_warps": 4, "num_stages": 2, "configs_hash": "h"}
        best = json.loads(json.dumps(data))
        loaded = _load_cached_autotuning(best, "h", [plain], {})
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.kwargs, {"BLOCK": 32})

    @unittest.skipUnless(has_triton_package(), "requires Triton")
    def test_autotune_cache_saves_container_enum_config_kwargs(self):
        # Enum members nested inside list/dict kwargs pass _config_json_cacheable
        # (which recurses), so _json_config_value must unwrap them recursively
        # too: otherwise save() hands raw Enum members to json.dumps and the
        # resulting TypeError aborts the run. A plain Enum (not IntEnum, which
        # json serializes as its int) exercises this.
        import json
        import os
        import tempfile

        import triton

        from torch._inductor.remote_cache import LocalAutotuneCache
        from torch._inductor.runtime.autotune_cache import (
            _config_json_cacheable,
            _json_config_value,
            _load_cached_autotuning,
            AutotuneCache,
        )

        class Mode(Enum):
            A = 1
            B = 2

        kwargs = {"MODES": [Mode.A], "TABLE": {"m": Mode.A}, "BLOCK": 64}
        cfg = triton.Config(dict(kwargs), num_warps=4, num_stages=2)
        self.assertTrue(_config_json_cacheable(cfg))
        data = {key: _json_config_value(val) for key, val in cfg.kwargs.items()}
        self.assertEqual(data, {"MODES": [1], "TABLE": {"m": 1}, "BLOCK": 64})
        json.dumps(data)  # anything cacheable must serialize

        with tempfile.TemporaryDirectory() as tmpdir:
            key = os.path.join(tmpdir, "kernel.best_config")
            local_cache = LocalAutotuneCache("local-autotune")
            cache = AutotuneCache(configs_hash="h", local_cache=(local_cache, key))
            cache.save(cfg, time_taken_ns=0)
            best = local_cache.get(key)
        self.assertIsNotNone(best)
        # Warm load must match back to the original config, keeping the real
        # Enum members rather than the degraded JSON values.
        loaded = _load_cached_autotuning(best, "h", [cfg], {})
        self.assertIs(loaded, cfg)
        self.assertEqual(loaded.kwargs, kwargs)
        self.assertIs(loaded.kwargs["MODES"][0], Mode.A)

    @unittest.skipUnless(has_triton_package(), "requires Triton")
    def test_autotune_cache_ambiguous_match_reautotunes(self):
        # Enum unwrapping makes MODE=Mode.A and MODE=1 serialize identically, so
        # a cached winner matches both candidates. Reconstructing would bake the
        # raw JSON value even when the winner was the Enum member (plain Enum
        # __eq__ is identity, a silent semantic flip); re-autotune instead.
        import json

        import triton

        from torch._inductor.runtime.autotune_cache import _load_cached_autotuning

        class Mode(Enum):
            A = 1

        cfg_enum = triton.Config({"MODE": Mode.A}, num_warps=4, num_stages=2)
        cfg_raw = triton.Config({"MODE": 1}, num_warps=4, num_stages=2)
        data = {"MODE": 1, "num_warps": 4, "num_stages": 2, "configs_hash": "h"}
        best = json.loads(json.dumps(data))
        self.assertIsNone(_load_cached_autotuning(best, "h", [cfg_enum, cfg_raw], {}))

    @unittest.skipUnless(has_triton_package(), "requires Triton")
    def test_autotune_cache_multi_match_without_enums_reconstructs(self):
        # Multiple value-identical matches carry no Enum-vs-raw ambiguity
        # (duplicate identical configs, or a config whose kwargs are a subset
        # of the saved best); warm load must fall through to reconstruction
        # rather than silently re-autotuning on every run.
        import json

        import triton

        from torch._inductor.runtime.autotune_cache import _load_cached_autotuning

        dup = [triton.Config({"BLOCK": 64}, num_warps=4, num_stages=2) for _ in "ab"]
        data = {"BLOCK": 64, "num_warps": 4, "num_stages": 2, "configs_hash": "h"}
        loaded = _load_cached_autotuning(json.loads(json.dumps(data)), "h", dup, {})
        self.assertIs(loaded, dup[0])
        self.assertEqual(loaded.num_warps, 4)
        self.assertEqual(loaded.num_stages, 2)

        # Duplicate candidates with tuple kwargs (JSON stores lists) are
        # interchangeable, so warm load must return one of them rather than
        # treating the tuple/list mismatch as Enum ambiguity and re-autotuning.
        dup_tuple = [
            triton.Config({"BLOCK_SHAPE": (64, 32)}, num_warps=4, num_stages=2)
            for _ in "ab"
        ]
        data_tuple = {"BLOCK_SHAPE": [64, 32], "num_warps": 4, "num_stages": 2}
        data_tuple["configs_hash"] = "h"
        loaded = _load_cached_autotuning(
            json.loads(json.dumps(data_tuple)), "h", dup_tuple, {}
        )
        self.assertIs(loaded, dup_tuple[0])
        self.assertIs(type(loaded.kwargs["BLOCK_SHAPE"]), tuple)

        subset = [
            triton.Config({"BLOCK": 64}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK": 64, "SPLIT": 2}, num_warps=4, num_stages=2),
        ]
        data["SPLIT"] = 2
        loaded = _load_cached_autotuning(json.loads(json.dumps(data)), "h", subset, {})
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.kwargs, {"BLOCK": 64, "SPLIT": 2})

    @unittest.skipUnless(HAS_GPU_AND_TRITON, "requires GPU and Triton")
    def test_plain_enum_constexpr_in_user_defined_triton_kernel(self):
        import triton
        import triton.language as tl

        @triton.jit
        def enum_kernel(
            in_ptr, out_ptr, numel, MODE: tl.constexpr, BLOCK_SIZE: tl.constexpr
        ):
            offsets = tl.arange(0, BLOCK_SIZE)
            mask = offsets < numel
            x = tl.load(in_ptr + offsets, mask=mask)
            if MODE.value == 1:
                output = x + 1
            else:
                output = x * 2
            tl.store(out_ptr + offsets, output, mask=mask)

        def fn(x):
            output = torch.empty_like(x)
            enum_kernel[(1,)](x, output, x.numel(), plistlib.FMT_XML, BLOCK_SIZE=256)
            return output

        x = torch.randn(128, device=GPU_TYPE)
        actual, code = run_and_get_code(torch.compile(fn), x)
        self.assertEqual(actual, fn(x))
        self.assertIn("PlistFormat['FMT_XML']", code[0])

    @unittest.skipUnless(HAS_GPU_AND_TRITON, "requires GPU and Triton")
    def test_torch_size_constexpr_in_user_defined_triton_kernel(self):
        # torch.Size is a tuple subclass with no semantics of its own, so it
        # must render as a plain tuple rather than hit the generic
        # tuple-subclass decline ("cannot be written into the generated
        # kernel"), which would regress from eager.
        import triton
        import triton.language as tl

        @triton.jit
        def size_kernel(
            in_ptr, out_ptr, numel, SHAPE: tl.constexpr, BLOCK_SIZE: tl.constexpr
        ):
            offsets = tl.arange(0, BLOCK_SIZE)
            mask = offsets < numel
            x = tl.load(in_ptr + offsets, mask=mask)
            tl.store(out_ptr + offsets, x + SHAPE[0], mask=mask)

        def fn(x):
            output = torch.empty_like(x)
            size_kernel[(1,)](x, output, x.numel(), x.shape, BLOCK_SIZE=256)
            return output

        x = torch.randn(128, device=GPU_TYPE)
        actual, code = run_and_get_code(torch.compile(fn), x)
        self.assertEqual(actual, fn(x))
        self.assertIn("(128,)", code[0])

    @unittest.skipUnless(HAS_GPU_AND_TRITON, "requires GPU and Triton")
    def test_escape_bearing_constexpr_in_user_defined_triton_kernel(self):
        # The rendered constants are spliced into the '''...''' literal passed
        # to async_compile.triton and parsed twice; reprs carrying backslashes
        # or quotes (a "\n" string, bytes) must be escaped like kernel_src or
        # the generated module fails with SyntaxError on the second parse.
        import triton
        import triton.language as tl

        @triton.jit
        def esc_kernel(
            in_ptr,
            out_ptr,
            numel,
            MODE: tl.constexpr,
            TAG: tl.constexpr,
            BLOCK_SIZE: tl.constexpr,
        ):
            offsets = tl.arange(0, BLOCK_SIZE)
            mask = offsets < numel
            x = tl.load(in_ptr + offsets, mask=mask)
            if MODE == "a\nb":
                output = x + 1
            else:
                output = x * 2
            tl.store(out_ptr + offsets, output, mask=mask)

        def fn(x):
            y = torch.empty_like(x)
            esc_kernel[(1,)](x, y, x.numel(), "a\nb", b"\x00\\", 256)
            return y

        x = torch.randn(128, device=GPU_TYPE)
        res, code = run_and_get_code(torch.compile(fn), x)
        # x + 1, not x * 2: the "a\nb" constexpr survived both parses intact.
        self.assertEqual(fn(x), res)
        compile(code[0], "<test-generated-wrapper>", "exec")

    @unittest.skipUnless(HAS_GPU_AND_TRITON, "requires GPU and Triton")
    def test_module_enum_constexpr_compiles_in_subprocess_worker(self):
        # The generated "import plistlib as __inductor_constexpr_module_N" must
        # execute inside a real compile-worker subprocess, not just the parent.
        import triton
        import triton.language as tl

        from torch._inductor.async_compile import AsyncCompile, shutdown_compile_workers
        from torch._inductor.utils import fresh_cache

        @triton.jit
        def enum_kernel(
            in_ptr, out_ptr, numel, MODE: tl.constexpr, BLOCK_SIZE: tl.constexpr
        ):
            offsets = tl.arange(0, BLOCK_SIZE)
            mask = offsets < numel
            x = tl.load(in_ptr + offsets, mask=mask)
            if MODE.value == 1:
                output = x + 1
            else:
                output = x * 2
            tl.store(out_ptr + offsets, output, mask=mask)

        def fn(x):
            y = torch.empty_like(x)
            enum_kernel[(1,)](x, y, x.numel(), plistlib.FMT_XML, BLOCK_SIZE=256)
            return y

        x = torch.randn(128, device=GPU_TYPE)
        patched = {"compile_threads": 2, "worker_start_method": "subprocess"}
        with inductor_config.patch(patched):
            shutdown_compile_workers()
            try:
                AsyncCompile.warm_pool()
                AsyncCompile.wakeup()
                AsyncCompile.wait_pool_ready()
                self.assertTrue(AsyncCompile.use_process_pool())
                logger_name = "torch._inductor.async_compile"
                with (
                    fresh_cache(),
                    self.assertNoLogs(logger_name, level="WARNING"),
                ):
                    res, code = run_and_get_code(torch.compile(fn), x)
            finally:
                shutdown_compile_workers()
        self.assertEqual(fn(x), res)
        self.assertIn("PlistFormat['FMT_XML']", code[0])

    @unittest.skipUnless(HAS_GPU_AND_TRITON, "requires GPU and Triton")
    def test_constexpr_module_missing_in_worker_falls_back_in_process(self):
        # Compile workers snapshot PYTHONPATH when the pool spawns, so a module
        # added to sys.path afterwards imports fine in the parent (where codegen
        # validates it) but raises ModuleNotFoundError in the worker. The
        # compile must fall back to in-process compilation instead of failing.
        import importlib
        import os
        import sys
        import tempfile

        import triton
        import triton.language as tl

        from torch._inductor.async_compile import AsyncCompile, shutdown_compile_workers
        from torch._inductor.utils import fresh_cache

        @triton.jit
        def probe_kernel(
            in_ptr, out_ptr, numel, MODE: tl.constexpr, BLOCK_SIZE: tl.constexpr
        ):
            offsets = tl.arange(0, BLOCK_SIZE)
            mask = offsets < numel
            x = tl.load(in_ptr + offsets, mask=mask)
            if MODE.value == 1:
                output = x + 1
            else:
                output = x * 2
            tl.store(out_ptr + offsets, output, mask=mask)

        module_name = f"inductor_constexpr_probe_{os.getpid()}"
        module_src = "import enum\n\nclass Mode(enum.Enum):\n    ADD = 1\n    MUL = 2\n"
        x = torch.randn(128, device=GPU_TYPE)
        patched = {"compile_threads": 2, "worker_start_method": "subprocess"}
        with tempfile.TemporaryDirectory() as tmpdir, inductor_config.patch(patched):
            with open(os.path.join(tmpdir, f"{module_name}.py"), "w") as f:
                f.write(module_src)
            shutdown_compile_workers()
            try:
                # Spawn the worker pool before the module becomes importable so
                # the workers' PYTHONPATH snapshot cannot contain tmpdir.
                AsyncCompile.warm_pool()
                AsyncCompile.wakeup()
                AsyncCompile.wait_pool_ready()
                self.assertTrue(AsyncCompile.use_process_pool())
                sys.path.insert(0, tmpdir)
                importlib.invalidate_caches()
                mod = importlib.import_module(module_name)

                def fn(x):
                    y = torch.empty_like(x)
                    probe_kernel[(1,)](x, y, x.numel(), mod.Mode.ADD, BLOCK_SIZE=256)
                    return y

                logger_name = "torch._inductor.async_compile"
                with (
                    fresh_cache(),
                    self.assertLogs(logger_name, level="WARNING") as logs,
                ):
                    res, code = run_and_get_code(torch.compile(fn), x)
                self.assertTrue(
                    any("in-process compilation" in msg for msg in logs.output)
                )
                self.assertEqual(fn(x), res)
                self.assertIn(module_name, code[0])
            finally:
                if tmpdir in sys.path:
                    sys.path.remove(tmpdir)
                sys.modules.pop(module_name, None)
                shutdown_compile_workers()

    def test_constexpr_fallback_result_is_memoized(self):
        # LambdaFuture.result() re-runs its result_fn on every call and the
        # worker task re-raises the same SubprocException, so without
        # memoization every re-entry of the fallback path would warn again and
        # recompile the kernel in-process again.
        import os
        from unittest.mock import Mock

        from torch._inductor.async_compile import AsyncCompile
        from torch._inductor.compile_worker.subproc_pool import SubprocException

        module_name = f"inductor_fallback_probe_{os.getpid()}"
        source_code = f"import {module_name} as __inductor_constexpr_module_0\n"
        task = Mock()
        task.result.side_effect = SubprocException(
            f"ModuleNotFoundError: No module named '{module_name}'"
        )
        pool = Mock()
        pool.submit.return_value = task
        sentinel_kernel = object()
        with (
            patch.object(AsyncCompile, "use_process_pool", return_value=True),
            patch.object(AsyncCompile, "process_pool", return_value=pool),
            patch.object(
                AsyncCompile, "_compile_triton_in_process", return_value=sentinel_kernel
            ) as compile_in_process,
        ):
            future = AsyncCompile().triton("probe_kernel", source_code)
            logger_name = "torch._inductor.async_compile"
            with self.assertLogs(logger_name, level="WARNING") as logs:
                self.assertIs(future.result(), sentinel_kernel)
            fallback_warnings = [
                msg for msg in logs.output if "in-process compilation" in msg
            ]
            self.assertEqual(len(fallback_warnings), 1)
            with self.assertNoLogs(logger_name, level="WARNING"):
                self.assertIs(future.result(), sentinel_kernel)
            self.assertEqual(compile_in_process.call_count, 1)
        self.assertEqual(task.result.call_count, 1)

    def test_constexpr_module_missing_in_worker_accepts_exception(self):
        # Spawn/fork worker pools re-raise the worker's ModuleNotFoundError
        # directly instead of wrapping it in SubprocException; the helper must
        # match on the exception's .name.
        from torch._inductor.async_compile import _constexpr_module_missing_in_worker

        src = "import foo.bar as __inductor_constexpr_module_0\n"
        err = ModuleNotFoundError("No module named 'foo'", name="foo")
        self.assertEqual(_constexpr_module_missing_in_worker(src, err), "foo.bar")
        other = ModuleNotFoundError("No module named 'baz'", name="baz")
        self.assertIsNone(_constexpr_module_missing_in_worker(src, other))
        nameless = ModuleNotFoundError("No module named 'foo'")
        self.assertIsNone(_constexpr_module_missing_in_worker(src, nameless))
        # The root-name imports emitted for object-valued constexpr parameter
        # defaults must also trigger the in-process fallback.
        root_src = "from foo.bar import Cfg as Cfg\n"
        self.assertEqual(_constexpr_module_missing_in_worker(root_src, err), "foo.bar")
        self.assertIsNone(_constexpr_module_missing_in_worker(root_src, other))
        aliased = "from foo.bar import Cfg as Renamed\n"
        self.assertIsNone(_constexpr_module_missing_in_worker(aliased, err))
        # `import foo` can fail because foo/__init__.py imports a submodule the
        # worker lacks; the reported missing name is then a descendant of the
        # constexpr module and must still be attributed to it.
        parent_src = "import foo as __inductor_constexpr_module_0\n"
        descendant = ModuleNotFoundError(
            "No module named 'foo.helpers'", name="foo.helpers"
        )
        self.assertEqual(
            _constexpr_module_missing_in_worker(parent_src, descendant), "foo"
        )
        lookalike = ModuleNotFoundError("No module named 'foobar'", name="foobar")
        self.assertIsNone(_constexpr_module_missing_in_worker(parent_src, lookalike))
        # torch.dtype / tl.dtype constexprs import the library roots themselves;
        # a missing submodule under those is a real error in the worker, not a
        # stale search path, so it must not trigger the in-process fallback.
        for root in ("torch", "triton.language"):
            root_src = f"import {root} as __inductor_constexpr_module_0\n"
            missing_sub = ModuleNotFoundError(
                f"No module named '{root}.zzz'", name=f"{root}.zzz"
            )
            self.assertIsNone(
                _constexpr_module_missing_in_worker(root_src, missing_sub)
            )
        helper_src = "from triton.language import store as store\n"
        self.assertIsNone(
            _constexpr_module_missing_in_worker(
                helper_src,
                ModuleNotFoundError(
                    "No module named 'triton.language.extra.x'",
                    name="triton.language.extra.x",
                ),
            )
        )
        # A formatted worker traceback may chain the constexpr import failure
        # before an unrelated secondary one; any reported missing module that
        # matches a constexpr import is attributed regardless of chain order,
        # so the in-process fallback still fires.
        chained = (
            "ModuleNotFoundError: No module named 'foo'\n\n"
            "During handling of the above exception, another exception occurred:\n\n"
            "ModuleNotFoundError: No module named 'baz'\n"
        )
        self.assertEqual(_constexpr_module_missing_in_worker(src, chained), "foo.bar")
        reversed_chain = (
            "ModuleNotFoundError: No module named 'baz'\n\n"
            "During handling of the above exception, another exception occurred:\n\n"
            "ModuleNotFoundError: No module named 'foo'\n"
        )
        self.assertEqual(
            _constexpr_module_missing_in_worker(src, reversed_chain), "foo.bar"
        )
        # A traceback with only unrelated failures matches no constexpr import.
        unrelated_chain = (
            "ModuleNotFoundError: No module named 'baz'\n\n"
            "During handling of the above exception, another exception occurred:\n\n"
            "ModuleNotFoundError: No module named 'qux'\n"
        )
        self.assertIsNone(_constexpr_module_missing_in_worker(src, unrelated_chain))

    @unittest.skipUnless(has_triton_package(), "requires Triton")
    def test_constexpr_fallback_catches_raw_module_not_found(self):
        # With TORCHINDUCTOR_WORKER_START=spawn/fork the worker's
        # ModuleNotFoundError propagates raw from future.result(); the fallback
        # must fire for constexpr imports and re-raise anything else unchanged.
        import os
        from unittest.mock import Mock

        from torch._inductor.async_compile import AsyncCompile, CompiledTritonKernels

        # The raw-raise path never calls remove_future, so without this a
        # Mock-holding LambdaFuture would sit in the process-global kernel
        # cache until the next compile_fx clears it.
        self.addCleanup(CompiledTritonKernels.cache_clear)
        module_name = f"inductor_spawn_probe_{os.getpid()}"
        source_code = f"import {module_name} as __inductor_constexpr_module_0\n"
        task = Mock()
        task.result.side_effect = ModuleNotFoundError(
            f"No module named '{module_name}'", name=module_name
        )
        pool = Mock()
        pool.submit.return_value = task
        sentinel_kernel = object()
        with (
            patch.object(AsyncCompile, "use_process_pool", return_value=True),
            patch.object(AsyncCompile, "process_pool", return_value=pool),
            patch.object(
                AsyncCompile, "_compile_triton_in_process", return_value=sentinel_kernel
            ),
        ):
            future = AsyncCompile().triton("probe_kernel", source_code)
            logger_name = "torch._inductor.async_compile"
            with self.assertLogs(logger_name, level="WARNING") as logs:
                self.assertIs(future.result(), sentinel_kernel)
            self.assertTrue(any("in-process compilation" in msg for msg in logs.output))
            # An unrelated missing module must propagate unchanged.
            unrelated = f"import {module_name}x as __inductor_constexpr_module_0\n"
            future = AsyncCompile().triton("probe_kernel", unrelated)
            with self.assertRaisesRegex(
                ModuleNotFoundError, f"No module named '{module_name}'"
            ):
                future.result()

    @unittest.skipUnless(HAS_GPU_AND_TRITON, "requires GPU and Triton")
    def test_unspellable_enum_constexpr_errors_clearly(self):
        import triton
        import triton.language as tl

        class Mode(Enum):
            ADD = 1

        @triton.jit
        def enum_kernel(
            in_ptr, out_ptr, numel, MODE: tl.constexpr, BLOCK_SIZE: tl.constexpr
        ):
            offsets = tl.arange(0, BLOCK_SIZE)
            mask = offsets < numel
            x = tl.load(in_ptr + offsets, mask=mask)
            output = tl.where(MODE == 1, x + 1, x * 2)
            tl.store(out_ptr + offsets, output, mask=mask)

        def fn(x):
            output = torch.empty_like(x)
            enum_kernel[(1,)](x, output, x.numel(), Mode.ADD, 256)
            return output

        x = torch.randn(128, device=GPU_TYPE)
        with self.assertRaisesRegex(
            BackendCompilerFailed,
            r"cannot be written into.*Mode is defined inside a function",
        ):
            torch.compile(fn)(x)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_CPU or HAS_GPU:
        run_tests("sympy")
