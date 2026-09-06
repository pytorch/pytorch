# Owner(s): ["module: inductor"]
import ast
import contextlib
import dataclasses
import unittest
from collections import namedtuple, OrderedDict
from enum import Enum, IntEnum
from types import SimpleNamespace
from unittest.mock import Mock, patch

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
from torch.testing._internal.inductor_utils import (
    GPU_TYPE,
    HAS_CPU,
    HAS_GPU,
    HAS_GPU_AND_TRITON,
    TRITON_HAS_CPU,
)
from torch.utils._sympy.functions import FloorDiv, TruncToFloat, TruncToInt
from torch.utils._sympy.symbol import make_symbol, SymT
from torch.utils._sympy.value_ranges import ValueRanges
from torch.utils._triton import has_triton_package


if has_triton_package():
    import triton
    import triton as triton_alias
    import triton.language as tl
    from triton import jit as triton_jit

    @triton.jit(
        noinline=True,
        debug=True,
        do_not_specialize=["x"],
    )
    def noinline_helper_for_codegen(x, out, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        values = tl.load(x + offsets, mask=mask)
        tl.store(out + offsets, values + 1, mask=mask)

    @triton.jit
    def root_for_noinline_helper(x, out, n_elements, BLOCK_SIZE: tl.constexpr):
        noinline_helper_for_codegen(x, out, n_elements, BLOCK_SIZE)

    @triton.jit
    def root_decorator_for_codegen(x, out, n_elements, BLOCK_SIZE: tl.constexpr):
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        values = tl.load(x + offsets, mask=mask)
        tl.store(out + offsets, values + 1, mask=mask)

    @triton.jit(
        do_not_specialize=["one"],
        do_not_specialize_on_alignment=["multiple_of_16"],
    )
    def root_specialization_for_codegen(
        x,
        out,
        one,
        multiple_of_16,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        values = tl.load(x + offsets, mask=mask)
        tl.store(out + offsets, values + one + multiple_of_16, mask=mask)

    @triton_jit(noinline=True, debug=True)
    def aliased_jit_helper_for_codegen(x):
        return x + 1

    @triton.jit
    def root_for_aliased_jit_helper(x, out, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        values = aliased_jit_helper_for_codegen(tl.load(x + offsets, mask=mask))
        tl.store(out + offsets, values, mask=mask)

    @triton_alias.jit(noinline=True, debug=True)
    def module_aliased_jit_helper_for_codegen(x):
        return x + 1

    @triton.jit
    def root_for_module_aliased_jit_helper(
        x, out, n_elements, BLOCK_SIZE: tl.constexpr
    ):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        values = module_aliased_jit_helper_for_codegen(tl.load(x + offsets, mask=mask))
        tl.store(out + offsets, values, mask=mask)

    def repr_for_codegen(*args, **kwargs):
        return "repr"

    @triton.jit(repr=repr_for_codegen)
    def global_option_jit_helper_for_codegen(x):
        return x + 1

    @triton.jit
    def root_for_global_option_jit_helper(x, out, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        values = global_option_jit_helper_for_codegen(tl.load(x + offsets, mask=mask))
        tl.store(out + offsets, values, mask=mask)


try:
    from .triton_constexpr_configs import (
        tl as TritonLanguageShadowConfig,
        UserDefinedAttrsLikeConfig,
        UserDefinedPydanticLikeConfig,
        UserDefinedTritonKernelConfigMode,
        UserDefinedTritonKernelConfigNamespace,
        UserDefinedTritonKernelEnumConfig,
        UserDefinedTritonKernelHiddenConfig,
        UserDefinedTritonKernelNestedConfig,
        UserDefinedTritonKernelNonInitConfig,
    )
except ImportError:
    from triton_constexpr_configs import (
        tl as TritonLanguageShadowConfig,
        UserDefinedAttrsLikeConfig,
        UserDefinedPydanticLikeConfig,
        UserDefinedTritonKernelConfigMode,
        UserDefinedTritonKernelConfigNamespace,
        UserDefinedTritonKernelEnumConfig,
        UserDefinedTritonKernelHiddenConfig,
        UserDefinedTritonKernelNestedConfig,
        UserDefinedTritonKernelNonInitConfig,
    )


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

    def test_importable_constexpr_types_bare_nested_class_repr(self):
        nested_type = UserDefinedTritonKernelConfigNamespace.BareNested
        value = nested_type(offset=2)
        with self.assertRaisesRegex(ImportError, "uses the bare name BareNested"):
            get_importable_constexpr_types([value])

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

    def test_importable_constexpr_types_non_init_dataclass_field_error(self):
        value = UserDefinedTritonKernelNonInitConfig(offset=2)
        with self.assertRaisesRegex(
            ImportError, "repr-visible field derived with init=False"
        ):
            get_importable_constexpr_types([value])

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

        _check_divisibility(
            (0,),
            triton_utils.config_of(
                [SizeArg("A", sixteen), SizeArg("B", sixteen)],
                divisible_by_16_exclusions=(1,),
            ),
        )
        self.assertEqual(
            triton_utils.equal_1_arg_indices(
                [SizeArg("A", sympy.S.One), SizeArg("B", sympy.S.One)],
                exclusions=(1,),
            ),
            (0,),
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

    @unittest.skipUnless(has_triton_package(), "requires Triton")
    def test_user_defined_triton_kernel_uses_bare_root_jit_decorator(self):
        from torch._inductor.codegen.wrapper import (
            user_defined_triton_kernel_transitive_closure_source_code,
        )

        decorator_src = "@triton.jit(noinline=True, debug=True)"
        raw_src = "".join(root_decorator_for_codegen.raw_src).replace(
            "@triton.jit", decorator_src, 1
        )

        bare_source = user_defined_triton_kernel_transitive_closure_source_code(
            root_decorator_for_codegen
        )
        with patch.object(root_decorator_for_codegen, "raw_src", raw_src):
            decorated_source = (
                user_defined_triton_kernel_transitive_closure_source_code(
                    root_decorator_for_codegen
                )
            )
        self.assertEqual(bare_source, decorated_source)
        self.assertIn("@triton.jit\ndef root_decorator_for_codegen", decorated_source)
        self.assertNotIn("noinline=True", decorated_source)
        self.assertNotIn("debug=True", decorated_source)

    @unittest.skipUnless(
        HAS_GPU_AND_TRITON or (HAS_CPU and TRITON_HAS_CPU),
        "requires CPU or GPU Triton",
    )
    def test_user_defined_triton_kernel_honors_root_specialization_options(self):
        from torch._inductor.codegen import wrapper

        def fn(x):
            out = torch.empty_like(x)
            root_specialization_for_codegen[(1,)](
                x,
                out,
                1,
                16,
                x.numel(),
                BLOCK_SIZE=128,
            )
            return out

        device = GPU_TYPE if HAS_GPU_AND_TRITON else "cpu"
        x = torch.randn(64, device=device)
        with patch.object(wrapper, "config_of", wraps=wrapper.config_of) as config_of:
            result, _ = run_and_get_code(torch.compile(fn), x)

        self.assertEqual(result, x + 17)
        user_config_call = next(
            call
            for call in config_of.call_args_list
            if call.kwargs.get("pointer_range_override") == ()
        )
        self.assertEqual(user_config_call.kwargs["equal_to_1_exclusions"], (2,))
        self.assertEqual(user_config_call.kwargs["divisible_by_16_exclusions"], (3,))

    @unittest.skipUnless(has_triton_package(), "requires Triton")
    def test_user_defined_triton_cache_keys_include_root_specialization(self):
        from torch._functorch._aot_autograd.autograd_cache import (
            AOTAutogradCacheDetails,
        )
        from torch._higher_order_ops.triton_kernel_wrap import (
            kernel_side_table,
            triton_kernel_wrapper_mutation,
        )
        from torch._inductor.codecache import FxGraphHashDetails

        graph = torch.fx.Graph()
        kernel_idx = kernel_side_table.add_kernel(root_specialization_for_codegen)
        constant_args_idx = kernel_side_table.add_constant_args({})
        graph.call_function(
            triton_kernel_wrapper_mutation,
            kwargs={
                "kernel_idx": kernel_idx,
                "constant_args_idx": constant_args_idx,
                "grid": [(1,)],
                "tma_descriptor_metadata": {},
                "kwargs": {},
            },
        )
        graph.output(None)
        gm = torch.fx.GraphModule({}, graph)

        fx_details = FxGraphHashDetails(gm, [], {}, [])
        self.assertEqual(fx_details.user_defined_triton_source[0][1:3], ((2,), (3,)))

        aot_details = object.__new__(AOTAutogradCacheDetails)
        with patch.object(
            AOTAutogradCacheDetails,
            "_iter_triton_kernels_from_node",
            return_value=[root_specialization_for_codegen],
        ):
            aot_sources = aot_details.get_triton_source_codes_from_gm(gm)
        self.assertEqual(aot_sources[0][1:], ((2,), (3,)))

    @unittest.skipUnless(
        HAS_GPU_AND_TRITON or (HAS_CPU and TRITON_HAS_CPU),
        "requires CPU or GPU Triton",
    )
    def test_user_defined_triton_kernel_without_alignment_specialization_metadata(
        self,
    ):
        from torch._inductor.codegen.wrapper import PythonWrapperCodegen

        def fn(x):
            out = torch.empty_like(x)
            root_decorator_for_codegen[(1,)](
                x,
                out,
                x.numel(),
                BLOCK_SIZE=128,
            )
            return out

        define_kernel = PythonWrapperCodegen.define_user_defined_triton_kernel

        def define_kernel_without_alignment_metadata(self, kernel, *args, **kwargs):
            # Older Triton KernelParam instances predate this attribute. Remove
            # it only during Inductor codegen so current Triton can capture the HOP.
            alignment_options = [
                param.do_not_specialize_on_alignment for param in kernel.params
            ]
            try:
                for param in kernel.params:
                    del param.do_not_specialize_on_alignment
                return define_kernel(self, kernel, *args, **kwargs)
            finally:
                for param, alignment_option in zip(kernel.params, alignment_options):
                    param.do_not_specialize_on_alignment = alignment_option

        device = GPU_TYPE if HAS_GPU_AND_TRITON else "cpu"
        x = torch.randn(64, device=device)
        with patch.object(
            PythonWrapperCodegen,
            "define_user_defined_triton_kernel",
            define_kernel_without_alignment_metadata,
        ):
            result, _ = run_and_get_code(torch.compile(fn), x)

        self.assertEqual(result, x + 1)

    @unittest.skipUnless(
        HAS_GPU_AND_TRITON or (HAS_CPU and TRITON_HAS_CPU),
        "requires CPU or GPU Triton",
    )
    def test_user_defined_triton_kernel_preserves_jit_decorator(self):
        from torch._inductor.codegen.wrapper import (
            user_defined_triton_kernel_transitive_closure_source_code,
        )

        source = user_defined_triton_kernel_transitive_closure_source_code(
            root_for_noinline_helper
        )
        decorator_idx = source.index("@triton.jit(")
        helper_idx = source.index("def noinline_helper_for_codegen")
        self.assertLess(decorator_idx, helper_idx)
        self.assertIn("noinline=True", source)
        self.assertIn("debug=True", source)
        self.assertIn('do_not_specialize=["x"]', source)

        def fn(x):
            out = torch.empty_like(x)
            n_elements = x.numel()
            root_for_noinline_helper[(1,)](x, out, n_elements, BLOCK_SIZE=128)
            return out

        device = GPU_TYPE if HAS_GPU_AND_TRITON else "cpu"
        x = torch.randn(64, device=device)
        result, _ = run_and_get_code(torch.compile(fn), x)
        self.assertEqual(result, x + 1)

    @unittest.skipUnless(has_triton_package(), "requires Triton")
    def test_user_defined_triton_kernel_preserves_aliased_jit_decorator(self):
        from torch._inductor.codegen.wrapper import (
            user_defined_triton_kernel_transitive_closure_source_code,
        )

        test_cases = (
            (root_for_aliased_jit_helper, "def aliased_jit_helper_for_codegen"),
            (
                root_for_module_aliased_jit_helper,
                "def module_aliased_jit_helper_for_codegen",
            ),
        )
        for root, helper_def in test_cases:
            source = user_defined_triton_kernel_transitive_closure_source_code(root)
            decorator_idx = source.index("@triton.jit(")
            helper_idx = source.index(helper_def)
            self.assertLess(decorator_idx, helper_idx)
            self.assertIn("noinline=True", source)
            self.assertIn("debug=True", source)

    def test_user_defined_triton_kernel_jit_decorator_parse_failure_falls_back(self):
        from torch._inductor.codegen.wrapper import _triton_jit_decorator_from_source

        for raw_src in (None, "", "@triton.jit(\n", ["@triton.jit(\n"]):
            self.assertEqual(
                _triton_jit_decorator_from_source(Mock(raw_src=raw_src)),
                "@triton.jit",
            )

    @unittest.skipUnless(has_triton_package(), "requires Triton")
    def test_user_defined_triton_kernel_jit_decorator_is_memoized(self):
        from torch._inductor.codegen.wrapper import _triton_jit_decorator_from_source

        _triton_jit_decorator_from_source.cache_clear()
        self.addCleanup(_triton_jit_decorator_from_source.cache_clear)
        with patch(
            "torch._inductor.codegen.wrapper.ast.parse", wraps=ast.parse
        ) as parse:
            first = _triton_jit_decorator_from_source(noinline_helper_for_codegen)
            first_call_count = parse.call_count
            second = _triton_jit_decorator_from_source(noinline_helper_for_codegen)

        self.assertEqual(first, second)
        self.assertGreater(first_call_count, 0)
        self.assertEqual(parse.call_count, first_call_count)

    @unittest.skipUnless(has_triton_package(), "requires Triton")
    def test_user_defined_triton_kernel_rejects_global_jit_decorator_option(self):
        from torch._inductor.codegen.wrapper import (
            user_defined_triton_kernel_transitive_closure_source_code,
        )

        msg = (
            "global_option_jit_helper_for_codegen: @triton.jit decorator options "
            "must be Python literals for Inductor codegen; non-literal options "
            "are not supported: repr="
        )
        with self.assertRaisesRegex(RuntimeError, msg):
            user_defined_triton_kernel_transitive_closure_source_code(
                root_for_global_option_jit_helper
            )

    @unittest.skipUnless(has_triton_package(), "requires Triton")
    def test_user_defined_triton_kernel_literal_eval_type_error(self):
        from torch._inductor.codegen.wrapper import _triton_jit_decorator_from_source

        msg = "non-literal options are not supported: noinline="
        _triton_jit_decorator_from_source.cache_clear()
        with (
            patch(
                "torch._inductor.codegen.wrapper.ast.literal_eval",
                side_effect=TypeError,
            ),
            self.assertRaisesRegex(RuntimeError, msg),
        ):
            _triton_jit_decorator_from_source(noinline_helper_for_codegen)

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

        def fn(x):
            out = torch.empty_like(x)
            n_elements = x.numel()

            def grid(meta):
                return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

            add_constexpr_kernel[grid](
                x,
                out,
                n_elements,
                cfg=UserDefinedTritonKernelNestedConfig(
                    nested=UserDefinedTritonKernelConfigNamespace.Nested(offset=2)
                ),
                BLOCK_SIZE=128,
            )
            return out

        device = GPU_TYPE if HAS_GPU_AND_TRITON else "cpu"
        x = torch.randn(1024, device=device)
        actual = torch.compile(fn)(x)
        self.assertEqual(actual, x + 2)

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

    def test_sanitize_for_repr(self):
        from torch._inductor.codegen.wrapper import _sanitize_for_repr

        class Color(Enum):
            RED = "red"
            BLUE = "blue"

        class Priority(IntEnum):
            LOW = 0
            HIGH = 1

        # Enum -> value
        self.assertEqual(_sanitize_for_repr(Color.RED), "red")
        self.assertEqual(_sanitize_for_repr(Priority.HIGH), 1)

        # Recursion into containers
        self.assertEqual(
            _sanitize_for_repr({"a": Color.RED, "b": [Priority.LOW, 42]}),
            {"a": "red", "b": [0, 42]},
        )

        # Tuples
        self.assertEqual(
            _sanitize_for_repr((Color.BLUE, 1)),
            ("blue", 1),
        )

        # Namedtuples
        Pair = namedtuple("Pair", ["x", "y"])
        result = _sanitize_for_repr(Pair(Color.RED, Priority.HIGH))
        self.assertIsInstance(result, Pair)
        self.assertEqual(result, Pair("red", 1))

        # Enum as dict key
        self.assertEqual(
            _sanitize_for_repr({Color.RED: 1}),
            {"red": 1},
        )

        # Nested enum value
        class Outer(Enum):
            INNER = Color.RED

        self.assertEqual(_sanitize_for_repr(Outer.INNER), "red")

        config = UserDefinedTritonKernelEnumConfig(
            UserDefinedTritonKernelConfigMode.FAST
        )
        self.assertEqual(len(get_importable_constexpr_types([config])), 1)
        sanitized_config = _sanitize_for_repr(config)
        self.assertIsInstance(sanitized_config, UserDefinedTritonKernelEnumConfig)
        self.assertEqual(sanitized_config.mode, 1)
        compile(repr(sanitized_config), "<sanitized-constexpr>", "eval")

        for config_type in (UserDefinedAttrsLikeConfig, UserDefinedPydanticLikeConfig):
            config = config_type(UserDefinedTritonKernelConfigMode.FAST, "hidden")
            self.assertEqual(len(get_importable_constexpr_types([config])), 1)
            sanitized_config = _sanitize_for_repr(config)
            self.assertIsInstance(sanitized_config.nested, int)
            compile(repr(sanitized_config), "<sanitized-constexpr>", "eval")

            unchanged_config = config_type(1, "hidden")
            self.assertIs(_sanitize_for_repr(unchanged_config), unchanged_config)

        sanitized_set = _sanitize_for_repr({UserDefinedTritonKernelConfigMode.FAST})
        self.assertIsInstance(next(iter(sanitized_set)), int)
        unchanged_set = {1}
        self.assertIs(_sanitize_for_repr(unchanged_set), unchanged_set)
        sanitized_frozenset = _sanitize_for_repr(
            frozenset({UserDefinedTritonKernelConfigMode.FAST})
        )
        self.assertIsInstance(next(iter(sanitized_frozenset)), int)

        mapping = OrderedDict([("mode", UserDefinedTritonKernelConfigMode.FAST)])
        sanitized_mapping = _sanitize_for_repr(mapping)
        self.assertIsInstance(sanitized_mapping, OrderedDict)
        self.assertIsInstance(sanitized_mapping["mode"], int)

        unchanged_mapping = OrderedDict([("mode", 1)])
        self.assertIs(_sanitize_for_repr(unchanged_mapping), unchanged_mapping)

        class ComputedReprArgs:
            @property
            def computed(self):
                return 1

            def __repr_args__(self):
                return (("computed", self.computed),)

        computed = ComputedReprArgs()
        self.assertIs(_sanitize_for_repr(computed), computed)

        class PositionalReprArgs:
            def __repr_args__(self):
                return ((None, 1),)

        positional = PositionalReprArgs()
        self.assertIs(_sanitize_for_repr(positional), positional)

        class LabelledMapping(dict):
            def __init__(self, label, values):
                super().__init__(values)
                self.label = label

        labelled_mapping = LabelledMapping("config", {"mode": 1})
        self.assertIs(_sanitize_for_repr(labelled_mapping), labelled_mapping)

        # Non-enum passthrough
        self.assertEqual(_sanitize_for_repr(42), 42)
        self.assertEqual(_sanitize_for_repr("hello"), "hello")

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


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_CPU or HAS_GPU:
        run_tests("sympy")
