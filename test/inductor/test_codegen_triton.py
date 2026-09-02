# Owner(s): ["module: inductor"]
import ast
import contextlib
import dataclasses
import plistlib
import sys
import types
import unittest
from collections import Counter, defaultdict, deque, namedtuple, OrderedDict
from enum import Enum, IntEnum, IntFlag
from types import SimpleNamespace
from unittest.mock import patch

import sympy

import torch
import torch._inductor.config as inductor_config
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
from torch._inductor.codegen.wrapper import (
    _constexpr_constant,
    _ConstexprRenderer,
    _escape_triton_kernel_source_for_wrapper,
    _render_constexpr_mappings,
)
from torch._inductor.dtype_propagation import DtypePropagationOpsHandler, promote_types
from torch._inductor.exc import InductorError
from torch._inductor.graph import GraphLowering
from torch._inductor.runtime.hints import DeviceProperties
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import (
    is_triton_fp8_dtype_supported,
    run_and_get_code,
    run_and_get_kernels,
)
from torch._inductor.virtualized import V
from torch._logging import warning_once
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
        UserDefinedPydanticLikeConfig,
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
        UserDefinedTritonKernelPermission,
        UserDefinedTritonKernelPlainMode,
        UserDefinedTritonKernelPlainReprConfig,
        UserDefinedTritonKernelSelfReferentialConfig,
    )
except ImportError:
    from test.inductor.triton_constexpr_configs import (
        runner as LauncherScopeShadowConfig,
        tl as TritonLanguageShadowConfig,
        UserDefinedAttrsLikeConfig,
        UserDefinedPydanticLikeConfig,
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
        UserDefinedTritonKernelPermission,
        UserDefinedTritonKernelPlainMode,
        UserDefinedTritonKernelPlainReprConfig,
        UserDefinedTritonKernelSelfReferentialConfig,
    )


WRAPPER_LOGGER = "torch._inductor.codegen.wrapper"


def render_constexpr(value, name="X"):
    """Render one constexpr value through the production entry point; returns
    the generated source expression and the module imports it needs."""
    renderer = _ConstexprRenderer()
    (rendered,) = _render_constexpr_mappings(renderer, [{name: value}], "test_kernel")
    return repr(rendered[name]), renderer.imports


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

    def test_constexpr_plain_repr_class_rendering(self):
        # A plain class (identity equality) with a constructor-style repr
        # renders through its module alias and verifies by instance state.
        value = UserDefinedTritonKernelPlainReprConfig(offset=3)
        source, imports = render_constexpr(value)
        alias = "__inductor_constexpr_module_0"
        self.assertEqual(
            source, f"{alias}.UserDefinedTritonKernelPlainReprConfig(offset=3)"
        )
        self.assertEqual(imports, [f"import {type(value).__module__} as {alias}"])
        scope = {}
        exec("\n".join(imports), scope)
        rebuilt = eval(source, scope)
        self.assertIs(type(rebuilt), UserDefinedTritonKernelPlainReprConfig)
        self.assertEqual(vars(rebuilt), vars(value))

        class LocalRepr:
            def __init__(self, offset):
                self.offset = offset

            def __repr__(self):
                return f"LocalRepr(offset={self.offset!r})"

        class NoRepr:
            pass

        with self.assertRaisesRegex(ImportError, "LocalRepr cannot be imported"):
            render_constexpr(LocalRepr(1))
        with self.assertRaisesRegex(RuntimeError, "NoRepr has no constructor-style"):
            render_constexpr(NoRepr())

    def test_constexpr_mapping_and_sequence_subclass_rendering(self):
        alias = "__inductor_constexpr_module_0"
        # dict/list subclasses whose repr is a constructor call render as that
        # call over the rendered items (interchangeable Enums unwrapped).
        mapping = OrderedDict(
            [("mode", UserDefinedTritonKernelConfigMode.FAST), ("block", 64)]
        )
        source, imports = render_constexpr(mapping)
        self.assertEqual(source, f"{alias}.OrderedDict({{'mode': 1, 'block': 64}})")
        self.assertEqual(imports, [f"import collections as {alias}"])
        scope = {}
        exec("\n".join(imports), scope)
        rebuilt = eval(source, scope)
        self.assertIs(type(rebuilt), OrderedDict)
        self.assertEqual(list(rebuilt.items()), [("mode", 1), ("block", 64)])
        self.assertEqual(
            render_constexpr(Counter({"a": 1}))[0], f"{alias}.Counter({{'a': 1}})"
        )
        self.assertEqual(render_constexpr(deque([1, 2]))[0], f"{alias}.deque([1, 2])")
        # defaultdict's repr is not a call over its items (the factory comes
        # first), so evaluating the rendering fails and it declines.
        with self.assertRaisesRegex(RuntimeError, "defaultdict.*raised TypeError"):
            render_constexpr(defaultdict(int, {"a": 1}))

    def test_constexpr_torch_device_and_flag_rendering(self):
        alias = "__inductor_constexpr_module_0"
        self.assertEqual(
            render_constexpr(torch.device("cuda", 0)),
            (f"{alias}.device(type='cuda', index=0)", [f"import torch as {alias}"]),
        )
        self.assertEqual(
            render_constexpr(torch.device("cpu"))[0], f"{alias}.device(type='cpu')"
        )
        self.assertEqual(render_constexpr(torch.strided)[0], f"{alias}.strided")
        perm = UserDefinedTritonKernelPermission
        with patch(f"{WRAPPER_LOGGER}.warning_once"):
            self.assertEqual(
                render_constexpr(perm.READ | perm.WRITE)[0],
                f"{alias}.UserDefinedTritonKernelPermission(3)",
            )
            self.assertEqual(
                render_constexpr(perm.READ)[0],
                f"{alias}.UserDefinedTritonKernelPermission['READ']",
            )

    def test_constexpr_spec_less_module_types_render(self):
        # Types living in a module registered in sys.modules by hand (no
        # __spec__) import fine in-process, so they render.
        module = types.ModuleType("inductor_constexpr_specless_probe")

        class Mode(Enum):
            A = 1

        @dataclasses.dataclass(frozen=True)
        class Cfg:
            offset: int

        for cls in (Mode, Cfg):
            cls.__module__ = module.__name__
            cls.__qualname__ = cls.__name__
            setattr(module, cls.__name__, cls)
        self.assertIsNone(module.__spec__)
        with patch.dict(sys.modules, {module.__name__: module}):
            with patch(f"{WRAPPER_LOGGER}.warning_once"):
                source, imports = render_constexpr(Mode.A)
            self.assertEqual(source, "__inductor_constexpr_module_0.Mode['A']")
            self.assertEqual(
                imports, [f"import {module.__name__} as __inductor_constexpr_module_0"]
            )
            self.assertEqual(
                render_constexpr(Cfg(1))[0],
                "__inductor_constexpr_module_0.Cfg(offset=1)",
            )

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
    def test_namedtuple_constexpr_launcher_does_not_reimport_types(self):
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

        def make_op(name, cfg):
            @torch.library.triton_op(f"test_codegen_triton::{name}", mutates_args={})
            def op(x: torch.Tensor) -> torch.Tensor:
                out = torch.empty_like(x)
                n_elements = x.numel()

                def grid(meta):
                    return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

                torch.library.wrap_triton(add_offset_kernel)[grid](
                    x, out, n_elements, cfg=cfg, BLOCK_SIZE=128
                )
                return out

            return op

        cases = (
            ("nested_point", UserDefinedTritonKernelConfigNamespace.Point(offset=2)),
            ("launcher_shadow", LauncherScopeShadowConfig(offset=3)),
        )
        device = GPU_TYPE if HAS_GPU_AND_TRITON else "cpu"
        x = torch.randn(1024, device=device)
        for name, cfg in cases:
            with self.subTest(config=type(cfg).__qualname__):
                op = make_op(name, cfg)
                compiled = torch.compile(lambda x, op=op: op(x), fullgraph=True)
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
        with self.assertRaisesRegex(
            InductorError, "RuntimeError: .*does not reproduce the original"
        ):
            compiled(x)

    @unittest.skipUnless(
        HAS_GPU_AND_TRITON or (HAS_CPU and has_triton_package()),
        "requires CPU or GPU Triton",
    )
    def test_object_constexpr_default_in_user_defined_triton_kernel(self):
        # The def-time default expression is evaluated when the generated
        # module re-execs the spliced kernel def (even when the caller passes
        # cfg explicitly), and Triton's code generator re-evaluates it with its
        # own semantics, so the default is rewritten to a module-level name
        # bound to the rendered value.
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
        alias = "__inductor_constexpr_module_0"
        binding = "__inductor_constexpr_default_0"
        self.assertIn(
            f"{binding} = {alias}.UserDefinedTritonKernelDefaultArgConfig(offset=1)",
            code[0],
        )
        self.assertIn(f"cfg: tl.constexpr = {binding}", code[0])

    @unittest.skipUnless(HAS_GPU_AND_TRITON, "requires GPU and Triton")
    def test_enum_constexpr_default_in_user_defined_triton_kernel(self):
        # A plain Enum member as a constexpr default: the def-time expression
        # `Mode.ADD` names nothing the generated module binds, so the default
        # is routed through the same renderer as constexpr values.
        import triton
        import triton.language as tl

        Mode = UserDefinedTritonKernelPlainMode

        @triton.jit
        def default_enum_kernel(
            in_ptr,
            out_ptr,
            numel,
            BLOCK_SIZE: tl.constexpr,
            MODE: tl.constexpr = Mode.ADD,
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
            default_enum_kernel[(1,)](
                x, output, x.numel(), BLOCK_SIZE=256, MODE=Mode.MUL
            )
            return output

        x = torch.randn(128, device=GPU_TYPE)
        with patch(f"{WRAPPER_LOGGER}.warning_once"):
            actual, code = run_and_get_code(torch.compile(fn), x)
        self.assertEqual(actual, x * 2)
        alias = "__inductor_constexpr_module_0"
        binding = "__inductor_constexpr_default_0"
        self.assertIn(
            f"{binding} = {alias}.UserDefinedTritonKernelPlainMode['ADD']", code[0]
        )
        self.assertIn(f"MODE: tl.constexpr = {binding}", code[0])
        self.assertIn(
            f"'MODE': {alias}.UserDefinedTritonKernelPlainMode['MUL']", code[0]
        )

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

    def test_constexpr_module_enum_rendering(self):
        with patch(f"{WRAPPER_LOGGER}.warning_once"):
            self.assertEqual(
                render_constexpr(plistlib.FMT_XML),
                (
                    "__inductor_constexpr_module_0.PlistFormat['FMT_XML']",
                    ["import plistlib as __inductor_constexpr_module_0"],
                ),
            )

    def test_constexpr_constructor_repr_config(self):
        config = UserDefinedTritonKernelNestedConfig(
            nested=UserDefinedTritonKernelConfigNamespace.Nested(offset=2)
        )
        source, imports = render_constexpr(config)
        alias = "__inductor_constexpr_module_0"
        self.assertEqual(
            source,
            f"{alias}.UserDefinedTritonKernelNestedConfig(nested={alias}.UserDefinedTritonKernelConfigNamespace.Nested(offset=2))",
        )
        self.assertEqual(imports, [f"import {type(config).__module__} as {alias}"])
        scope = {}
        exec("\n".join(imports), scope)
        self.assertEqual(eval(source, scope), config)
        # A nested class whose repr uses its bare name renders through its
        # qualified path, and a root named like a generated-module binding
        # (``tl``) is reached through the module alias, shadowing nothing.
        bare = UserDefinedTritonKernelConfigNamespace.BareNested(offset=2)
        self.assertEqual(
            render_constexpr(bare)[0],
            f"{alias}.UserDefinedTritonKernelConfigNamespace.BareNested(offset=2)",
        )
        self.assertEqual(
            render_constexpr(TritonLanguageShadowConfig(offset=2))[0],
            f"{alias}.tl(offset=2)",
        )

    def test_constexpr_constructor_repr_enum_field(self):
        config = UserDefinedTritonKernelEnumConfig(
            mode=UserDefinedTritonKernelConfigMode.FAST
        )
        # The interchangeable IntEnum field goes through the stack's enum
        # normalization, not the field's raw repr.
        self.assertEqual(
            render_constexpr(config)[0],
            "__inductor_constexpr_module_0.UserDefinedTritonKernelEnumConfig(mode=1)",
        )

    def test_constexpr_constructor_repr_protocols(self):
        nested = UserDefinedTritonKernelConfigNamespace.Nested(offset=2)
        for config_type in (UserDefinedAttrsLikeConfig, UserDefinedPydanticLikeConfig):
            with self.subTest(config_type=config_type.__name__):
                source, imports = render_constexpr(config_type(nested=nested))
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

    def test_constexpr_constructor_repr_declines(self):
        @dataclasses.dataclass(frozen=True)
        class LocalConfig:
            offset: int

        # A local (unimportable) config type declines with ImportError naming
        # the kernel, the constexpr and the type.
        with self.assertRaisesRegex(
            ImportError,
            r"Triton kernel test_kernel constexpr argument 'CFG' has value "
            r".*LocalConfig\(offset=2\) of type .*LocalConfig, which cannot be "
            r"written into the generated kernel: its type .*LocalConfig cannot be "
            r"imported by the generated module; define it at module scope",
        ):
            render_constexpr(LocalConfig(offset=2), name="CFG")
        # A repr-visible init=False field cannot be passed as a constructor
        # argument; a hidden required init parameter makes the constructor
        # call fail. Both decline instead of crashing the generated module.
        with self.assertRaisesRegex(RuntimeError, "raised TypeError"):
            render_constexpr(UserDefinedTritonKernelNonInitConfig(offset=2))
        with self.assertRaisesRegex(RuntimeError, "raised TypeError"):
            render_constexpr(UserDefinedTritonKernelHiddenConfig(2, object()))

    def test_constexpr_declines_value_unfaithful_configs(self):
        # The repr omits scale, so evaluating it would silently rebuild the
        # config with the default scale=1 instead of 7; rendering must decline
        # into the loud error rather than compile with wrong constants.
        unfaithful = UserDefinedTritonKernelHiddenDefaultConfig(offset=3, scale=7)
        with self.assertRaisesRegex(RuntimeError, "does not reproduce the original"):
            render_constexpr(unfaithful)
        # __post_init__ coerces again on rebuild: repr shows offset=6, but
        # evaluating Cfg(offset=6) produces offset=12.
        coercing = UserDefinedTritonKernelCoercingConfig(offset=3)
        self.assertEqual(coercing.offset, 6)
        with self.assertRaisesRegex(RuntimeError, "does not reproduce the original"):
            render_constexpr(coercing)
        # A hidden field still holding its default rebuilds faithfully.
        faithful = UserDefinedTritonKernelHiddenDefaultConfig(offset=3)
        source, imports = render_constexpr(faithful)
        scope = {}
        exec("\n".join(imports), scope)
        self.assertEqual(eval(source, scope), faithful)

    def test_constexpr_declines_self_referential_config(self):
        cfg = UserDefinedTritonKernelSelfReferentialConfig()
        cfg.child = cfg
        with self.assertRaisesRegex(RuntimeError, "self-referential"):
            render_constexpr(cfg)

    def test_constexpr_constructs_each_object_once(self):
        inner = UserDefinedTritonKernelCountingConfig()
        cfg = UserDefinedTritonKernelCountingConfig(
            UserDefinedTritonKernelCountingConfig(inner)
        )
        UserDefinedTritonKernelCountingConfig.constructed = 0
        render_constexpr(cfg)
        # Rendering verifies by evaluating the source once at the top level, so
        # compile-time constructor work stays linear in the number of objects.
        self.assertEqual(UserDefinedTritonKernelCountingConfig.constructed, 3)

    def test_constexpr_constant_enum_interchange(self):
        class Mode(IntEnum):
            EVEN = 2

        cases = (
            ("int_enum", Mode.EVEN, 2),
            ("int_flag", IntFlag("Flags", {"A": 1}).A, 1),
            ("str_enum", Enum("TextMode", {"A": "a"}, type=str).A, "a"),
            ("float_enum", Enum("FloatMode", {"A": 1.5}, type=float).A, 1.5),
        )
        for name, value, expected in cases:
            with self.subTest(case=name):
                self.assertEqual(_constexpr_constant(value), expected)
        # Plain members stay members (they are what the kernel bakes) unless
        # their class cannot be imported by the generated module.
        self.assertIs(_constexpr_constant(plistlib.FMT_XML), plistlib.FMT_XML)
        with patch.object(plistlib.PlistFormat, "__module__", "__main__"):
            self.assertEqual(_constexpr_constant(plistlib.FMT_XML), 1)

    def test_constexpr_constant_namedtuple_recursion(self):
        class Mode(IntEnum):
            EVEN = 2

        Pair = namedtuple("Pair", ("left", "right"))
        nested = {Mode.EVEN: [Mode.EVEN, (Mode.EVEN,), Pair(Mode.EVEN, 3)]}
        self.assertEqual(_constexpr_constant(nested), {2: [2, (2,), Pair(2, 3)]})

    def test_constexpr_constant_namedtuple_with_shifting_new(self):
        class Shifted(namedtuple("ShiftedBase", ("value",))):
            __slots__ = ()

            def __new__(cls, value):
                return super().__new__(cls, value + 1)

        shifted = Shifted(1)
        self.assertEqual(_constexpr_constant(shifted).value, 2)

    def test_unimportable_plain_enum_constexpr_unwraps_with_warning(self):
        # A plain Enum the generated module cannot import (defined in a local
        # scope here; __main__ is the script case) bakes its value, as before
        # Enum members rendered faithfully, and warns once naming the kernel,
        # the constexpr and the class.
        class Local(Enum):
            VALUE = 1

        warning_once.cache_clear()
        with self.assertLogs(WRAPPER_LOGGER, level="WARNING") as logs:
            self.assertEqual(render_constexpr(Local.VALUE, name="MODE"), ("1", []))
            self.assertEqual(render_constexpr(Local.VALUE, name="MODE"), ("1", []))
        (message,) = logs.output
        self.assertIn("test_kernel constexpr 'MODE'", message)
        self.assertIn(f"Enum class {Local.__module__}.{Local.__qualname__}", message)
        self.assertIn("value 1 is baked in instead", message)
        with patch.object(plistlib.PlistFormat, "__module__", "__main__"):
            with self.assertLogs(WRAPPER_LOGGER, level="WARNING") as logs:
                self.assertEqual(render_constexpr(plistlib.FMT_XML), ("1", []))
        self.assertIn("__main__.PlistFormat", logs.output[0])

        # A member whose value has no spelling either still errors, naming the
        # Enum type and the value's problem.
        class Opaque(Enum):
            VALUE = object()

        with self.assertRaisesRegex(
            RuntimeError, r"of type .*Opaque, which cannot .*object objects have no"
        ):
            render_constexpr(Opaque.VALUE)

    def test_plain_enum_member_constexpr_warns_once(self):
        warning_once.cache_clear()
        with self.assertLogs(WRAPPER_LOGGER, level="WARNING") as logs:
            for _ in range(2):
                render_constexpr(plistlib.FMT_XML, name="MODE")
        (message,) = logs.output
        self.assertIn("test_kernel constexpr 'MODE' is the plain Enum member", message)
        self.assertIn("<PlistFormat.FMT_XML: 1> of plistlib.PlistFormat", message)
        self.assertIn("MODE == 1", message)
        self.assertIn("triton.unwrap_plain_enum_constexpr=True", message)

    def test_unwrap_plain_enum_constexpr_knob(self):
        # The escape hatch restores value-unwrapping for every plain Enum, in
        # the rendered source and in the kernel metas, without the BC warning.
        with (
            inductor_config.patch({"triton.unwrap_plain_enum_constexpr": True}),
            patch(f"{WRAPPER_LOGGER}.warning_once") as warn,
        ):
            self.assertEqual(render_constexpr(plistlib.FMT_XML), ("1", []))
            self.assertEqual(_constexpr_constant(plistlib.FMT_XML), 1)
            perm = UserDefinedTritonKernelPermission
            self.assertEqual(render_constexpr(perm.READ | perm.WRITE), ("3", []))
        warn.assert_not_called()

    def test_constexpr_ordered_set_round_trip(self):
        from torch.utils._ordered_set import OrderedSet

        # A set subclass's type and iteration order are semantic: the source
        # must reconstruct the exact type in order, not a sorted builtin set.
        source, imports = render_constexpr(OrderedSet([2, 1]))
        scope = {}
        exec("\n".join(imports), scope)
        reconstructed = eval(source, scope)
        self.assertIs(type(reconstructed), OrderedSet)
        self.assertEqual(list(reconstructed), [2, 1])

    def test_constexpr_declines_set_subclasses(self):
        from torch.utils._ordered_set import OrderedSet

        class LocalSet(set):
            pass

        with self.assertRaisesRegex(RuntimeError, "no deterministic spelling"):
            render_constexpr(LocalSet({1}))

        # Only OrderedSet exactly reconstructs: other subclasses -- even fully
        # importable ones -- decline rather than emit hash-order-nondeterministic
        # or constructor-assuming source.
        import torch.utils._ordered_set as ordered_set_module

        class OrderedSubSet(OrderedSet):
            pass

        OrderedSubSet.__module__ = "torch.utils._ordered_set"
        OrderedSubSet.__qualname__ = "OrderedSubSet"
        with patch.object(
            ordered_set_module, "OrderedSubSet", OrderedSubSet, create=True
        ):
            with self.assertRaisesRegex(RuntimeError, "no deterministic spelling"):
                render_constexpr(OrderedSubSet([1]))

    def test_constexpr_container_subclass_declines(self):
        # Emitting a container display for a dict/list/tuple subclass would
        # silently drop its type, so without a constructor-style repr these
        # decline with the clear error instead of coercing to plain builtins.
        class LocalDict(dict):
            pass

        class LocalList(list):
            pass

        class LocalTuple(tuple):
            __slots__ = ()

        for value in (LocalDict({"a": 1}), LocalList([1]), LocalTuple((1,))):
            with self.subTest(type=type(value).__name__):
                self.assertIs(_constexpr_constant(value), value)
                with self.assertRaisesRegex(RuntimeError, "no constructor-style repr"):
                    render_constexpr(value)

    @parametrize(
        "value, expected",
        (
            subtest((64, ("64", [])), name="int"),
            subtest((float("inf"), ("float('inf')", [])), name="positive_inf"),
            subtest((float("-inf"), ("float('-inf')", [])), name="negative_inf"),
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
            subtest(((), ("()", [])), name="empty_tuple"),
        ),
    )
    def test_constexpr_builtin_source(self, value, expected):
        self.assertEqual(render_constexpr(value), expected)

    def test_constexpr_nan_declines(self):
        # NaN != NaN would break the ==-based config matching that consumes
        # these constants (autotune-cache lookup, precomputed-grid selection).
        for value in (float("nan"), -float("nan")):
            with self.assertRaisesRegex(RuntimeError, "NaN never compares equal"):
                render_constexpr(value)

    def test_constexpr_bytearray_source(self):
        value = bytearray(b"a\x00'\"b")
        source, imports = render_constexpr(value)
        self.assertEqual(imports, [])
        rebuilt = eval(source)
        self.assertIs(type(rebuilt), bytearray)
        self.assertEqual(rebuilt, value)

    @unittest.skipUnless(has_triton_package(), "requires Triton")
    def test_constexpr_triton_dtype_source(self):
        import triton.language as tl

        alias = "__inductor_constexpr_module_0"
        imports = [f"import triton.language as {alias}"]
        self.assertEqual(render_constexpr(tl.float32), (f"{alias}.float32", imports))
        self.assertEqual(
            render_constexpr((tl.float32,)), (f"({alias}.float32,)", imports)
        )
        self.assertEqual(
            render_constexpr(tl.constexpr(4)), (f"{alias}.constexpr(4)", imports)
        )

    def test_constexpr_enum_imports_do_not_collide(self):
        import uuid

        values = (plistlib.FMT_XML, uuid.SafeUUID.safe)
        renderer = _ConstexprRenderer()
        with patch(f"{WRAPPER_LOGGER}.warning_once"):
            rendered = _render_constexpr_mappings(
                renderer, [{"LEFT": values[0]}, {"RIGHT": values[1]}], "k"
            )
        scope = {}
        exec("\n".join(renderer.imports), scope)
        left, right = (eval(repr(mapping), scope) for mapping in rendered)
        self.assertIs(left["LEFT"], values[0])
        self.assertIs(right["RIGHT"], values[1])

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
            output = tl.where(MODE == 1, x + 1, x + 1)
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

    @unittest.skipUnless(HAS_GPU_AND_TRITON, "requires GPU and Triton")
    @parametrize("unwrap", (False, True))
    def test_plain_enum_constexpr_in_user_defined_triton_kernel(self, unwrap):
        # A plain Enum member is baked as the member, so `MODE == 1` is False
        # in the kernel as in a direct eager launch (plain Enum __eq__ is
        # identity); earlier releases baked its value and took the `MODE == 1`
        # branch. config.triton.unwrap_plain_enum_constexpr restores that.
        import triton
        import triton.language as tl

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
            output = torch.empty_like(x)
            enum_kernel[(1,)](
                x,
                output,
                x.numel(),
                UserDefinedTritonKernelPlainMode.ADD,
                BLOCK_SIZE=256,
            )
            return output

        x = torch.randn(128, device=GPU_TYPE)
        self.assertEqual(fn(x), x * 2)
        with (
            inductor_config.patch({"triton.unwrap_plain_enum_constexpr": unwrap}),
            patch(f"{WRAPPER_LOGGER}.warning_once") as warn,
        ):
            actual, code = run_and_get_code(torch.compile(fn), x)
        if unwrap:
            self.assertEqual(actual, x + 1)
            self.assertIn("'MODE': 1", code[0])
            warn.assert_not_called()
        else:
            self.assertEqual(actual, x * 2)
            self.assertIn("UserDefinedTritonKernelPlainMode['ADD']", code[0])
            warn.assert_called_once()
            message = warn.call_args.args[1] % warn.call_args.args[2:]
            self.assertIn("enum_kernel constexpr 'MODE'", message)
            self.assertIn("branching on MODE == 1", message)

    @unittest.skipUnless(HAS_GPU_AND_TRITON, "requires GPU and Triton")
    def test_main_module_enum_constexpr_unwraps_with_warning(self):
        # A plain Enum defined in __main__ (the top of a user script) cannot be
        # imported by the generated module, so its value is baked (the kernel
        # takes the `MODE == 1` branch, unlike eager) and one warning names the
        # kernel, the constexpr and the class instead of failing the compile.
        import triton
        import triton.language as tl

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
            output = torch.empty_like(x)
            enum_kernel[(1,)](
                x,
                output,
                x.numel(),
                UserDefinedTritonKernelPlainMode.ADD,
                BLOCK_SIZE=256,
            )
            return output

        x = torch.randn(128, device=GPU_TYPE)
        with (
            patch.object(UserDefinedTritonKernelPlainMode, "__module__", "__main__"),
            patch(f"{WRAPPER_LOGGER}.warning_once") as warn,
        ):
            actual, code = run_and_get_code(torch.compile(fn), x)
        self.assertEqual(actual, x + 1)
        self.assertIn("'MODE': 1", code[0])
        self.assertNotIn("__inductor_constexpr_module", code[0])
        warn.assert_called_once()
        message = warn.call_args.args[1] % warn.call_args.args[2:]
        self.assertIn("enum_kernel constexpr 'MODE'", message)
        self.assertIn("Enum class __main__.UserDefinedTritonKernelPlainMode", message)
        self.assertIn("value 1 is baked in instead", message)

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

    @unittest.skipUnless(has_triton_package(), "requires Triton")
    def test_constexpr_fallback_result_is_memoized(self):
        # LambdaFuture.result() re-runs its result_fn on every call and the
        # worker task re-raises the same SubprocException, so without
        # memoization every re-entry of the fallback path would recompile the
        # kernel in-process again.
        import os
        from unittest.mock import Mock

        from torch._inductor import async_compile
        from torch._inductor.async_compile import AsyncCompile
        from torch._inductor.compile_worker.subproc_pool import SubprocException
        from torch.utils._ordered_set import OrderedSet

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
            patch.object(async_compile, "_worker_unimportable_modules", OrderedSet()),
            patch.object(AsyncCompile, "use_process_pool", return_value=True),
            patch.object(AsyncCompile, "process_pool", return_value=pool),
            patch.object(
                AsyncCompile, "_compile_triton_in_process", return_value=sentinel_kernel
            ) as compile_in_process,
        ):
            future = AsyncCompile().triton("probe_kernel", source_code)
            logger_name = "torch._inductor.async_compile"
            warning_once.cache_clear()
            with self.assertLogs(logger_name, level="WARNING") as logs:
                self.assertIs(future.result(), sentinel_kernel)
            fallback_warnings = [
                msg for msg in logs.output if "in-process compilation" in msg
            ]
            self.assertEqual(len(fallback_warnings), 1)
            self.assertIn(repr(module_name), fallback_warnings[0])
            with self.assertNoLogs(logger_name, level="WARNING"):
                self.assertIs(future.result(), sentinel_kernel)
            self.assertEqual(compile_in_process.call_count, 1)
        self.assertEqual(task.result.call_count, 1)

    @unittest.skipUnless(has_triton_package(), "requires Triton")
    def test_constexpr_worker_unimportable_module_skips_pool(self):
        # Once a worker fails to import a constexpr module, later kernels
        # referencing it compile in-process directly: one failed worker submit
        # and one warning per module, not per kernel.
        import os
        from unittest.mock import Mock

        from torch._inductor import async_compile
        from torch._inductor.async_compile import AsyncCompile
        from torch._inductor.compile_worker.subproc_pool import SubprocException
        from torch.utils._ordered_set import OrderedSet

        module_name = f"inductor_skip_pool_probe_{os.getpid()}"
        sources = [
            f"import {module_name} as __inductor_constexpr_module_0\n# kernel {i}\n"
            for i in range(2)
        ]
        task = Mock()
        task.result.side_effect = SubprocException(
            f"ModuleNotFoundError: No module named '{module_name}'"
        )
        pool = Mock()
        pool.submit.return_value = task
        sentinel_kernel = object()
        with (
            patch.object(async_compile, "_worker_unimportable_modules", OrderedSet()),
            patch.object(AsyncCompile, "use_process_pool", return_value=True),
            patch.object(AsyncCompile, "process_pool", return_value=pool),
            patch.object(
                AsyncCompile, "_compile_triton_in_process", return_value=sentinel_kernel
            ) as compile_in_process,
            patch("torch._inductor.async_compile.warning_once") as warn,
        ):
            first = AsyncCompile().triton("probe_kernel_0", sources[0])
            self.assertIs(first.result(), sentinel_kernel)
            # The second kernel never reaches the pool.
            self.assertIs(
                AsyncCompile().triton("probe_kernel_1", sources[1]), sentinel_kernel
            )
        self.assertEqual(pool.submit.call_count, 1)
        self.assertEqual(compile_in_process.call_count, 2)
        warn.assert_called_once()
        self.assertEqual(warn.call_args.args[2:], (module_name,))

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

    @unittest.skipUnless(has_triton_package(), "requires Triton")
    def test_constexpr_fallback_catches_raw_module_not_found(self):
        # With TORCHINDUCTOR_WORKER_START=spawn/fork the worker's
        # ModuleNotFoundError propagates raw from future.result(); the fallback
        # must fire for constexpr imports and re-raise anything else unchanged.
        import os
        from unittest.mock import Mock

        from torch._inductor import async_compile
        from torch._inductor.async_compile import AsyncCompile, CompiledTritonKernels
        from torch.utils._ordered_set import OrderedSet

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
            patch.object(async_compile, "_worker_unimportable_modules", OrderedSet()),
            patch.object(AsyncCompile, "use_process_pool", return_value=True),
            patch.object(AsyncCompile, "process_pool", return_value=pool),
            patch.object(
                AsyncCompile, "_compile_triton_in_process", return_value=sentinel_kernel
            ),
        ):
            future = AsyncCompile().triton("probe_kernel", source_code)
            logger_name = "torch._inductor.async_compile"
            warning_once.cache_clear()
            with self.assertLogs(logger_name, level="WARNING") as logs:
                self.assertIs(future.result(), sentinel_kernel)
            self.assertTrue(any("in-process compilation" in msg for msg in logs.output))
            # An unrelated missing module must propagate unchanged.
            unrelated = f"import {module_name}x as __inductor_constexpr_module_0\n"
            future = AsyncCompile().triton("probe_kernel", unrelated)
            with self.assertRaisesRegex(ModuleNotFoundError, module_name):
                future.result()

    @unittest.skipUnless(HAS_GPU_AND_TRITON, "requires GPU and Triton")
    def test_plain_repr_class_constexpr_in_user_defined_triton_kernel(self):
        # A plain (non-dataclass) config class with a constructor-style repr in
        # an importable module compiles end to end.
        import triton
        import triton.language as tl

        @triton.jit
        def plain_offset_kernel(
            x, out, n_elements, cfg: tl.constexpr, BLOCK_SIZE: tl.constexpr
        ):
            pid = tl.program_id(axis=0)
            offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            values = tl.load(x + offsets, mask=mask)
            tl.store(out + offsets, values + cfg.offset, mask=mask)

        cfg = UserDefinedTritonKernelPlainReprConfig(offset=3)

        @torch.library.triton_op("test_codegen_triton::plain_repr_cfg", mutates_args={})
        def plain_repr_cfg(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            n_elements = x.numel()

            def grid(meta):
                return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

            torch.library.wrap_triton(plain_offset_kernel)[grid](
                x, out, n_elements, cfg=cfg, BLOCK_SIZE=128
            )
            return out

        x = torch.randn(1024, device=GPU_TYPE)
        compiled = torch.compile(lambda t: plain_repr_cfg(t), fullgraph=True)
        actual, code = run_and_get_code(compiled, x)
        self.assertEqual(actual, x + 3)
        self.assertIn("UserDefinedTritonKernelPlainReprConfig(offset=3)", code[0])

    @unittest.skipUnless(HAS_GPU_AND_TRITON, "requires GPU and Triton")
    def test_unrenderable_constexpr_errors_clearly(self):
        # Values with no spelling at all (an object without a constructor-style
        # repr) and types the generated module cannot import fail codegen with
        # an error naming the kernel, the constexpr, the value's type and a
        # remedy; the latter keeps the ImportError type.
        import triton
        import triton.language as tl

        class NoRepr:
            offset = 3

        @dataclasses.dataclass(frozen=True)
        class LocalCfg:
            offset: int

        @triton.jit
        def offset_kernel(
            x, out, n_elements, cfg: tl.constexpr, BLOCK_SIZE: tl.constexpr
        ):
            pid = tl.program_id(axis=0)
            offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            values = tl.load(x + offsets, mask=mask)
            tl.store(out + offsets, values + cfg.offset, mask=mask)

        def make_op(name, cfg):
            @torch.library.triton_op(f"test_codegen_triton::{name}", mutates_args={})
            def op(x: torch.Tensor) -> torch.Tensor:
                out = torch.empty_like(x)
                n_elements = x.numel()

                def grid(meta):
                    return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

                torch.library.wrap_triton(offset_kernel)[grid](
                    x, out, n_elements, cfg=cfg, BLOCK_SIZE=128
                )
                return out

            return op

        x = torch.randn(1024, device=GPU_TYPE)
        op = make_op("no_repr_cfg", NoRepr())
        with self.assertRaisesRegex(
            InductorError,
            r"RuntimeError: Triton kernel offset_kernel constexpr argument 'cfg' has "
            r"value <.*NoRepr object at 0x[0-9a-f]+> of type .*NoRepr, which cannot be "
            r"written into the generated kernel: its type .*NoRepr has no "
            r"constructor-style repr",
        ):
            torch.compile(lambda t, op=op: op(t), fullgraph=True)(x)
        op = make_op("local_cfg", LocalCfg(offset=3))
        with self.assertRaisesRegex(
            InductorError,
            r"ImportError: Triton kernel offset_kernel constexpr argument 'cfg' has "
            r"value .*LocalCfg\(offset=3\) of type .*LocalCfg, which cannot be written "
            r"into the generated kernel: its type .*LocalCfg cannot be imported by "
            r"the generated module; define it at module scope",
        ):
            torch.compile(lambda t, op=op: op(t), fullgraph=True)(x)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_CPU or HAS_GPU:
        run_tests("sympy")
