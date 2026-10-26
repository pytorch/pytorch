# Owner(s): ["module: inductor"]
import ast
import contextlib
import dataclasses
import re
import unittest
from collections import namedtuple, OrderedDict
from enum import Enum, IntEnum
from types import SimpleNamespace
from unittest.mock import Mock, patch

import sympy

import torch
import torch._inductor.config as inductor_config
from torch._inductor import ir
from torch._inductor.choices import GridPolicy, InductorChoices
from torch._inductor.codegen import triton_utils
from torch._inductor.codegen.common import ArgName, CSEVariable, SizeArg, TensorArg
from torch._inductor.codegen.cpp_wrapper_cpu import CppWrapperCpu
from torch._inductor.codegen.simd import IterationRangesRoot
from torch._inductor.codegen.simd_kernel_features import SIMDKernelFeatures
from torch._inductor.codegen.triton import (
    _materialize_trunc_to_float_expr,
    BlockParameters,
    FixedTritonConfig,
    get_triton_reduction_function,
    IndexingOptions,
    TMACompatibilityChecker,
    TritonCSEVariable,
    TritonKernel,
    TritonKernelOverrides,
    TritonScheduling,
    TritonSymbols,
)
from torch._inductor.codegen.wrapper import (
    _escape_triton_kernel_source_for_wrapper,
    render_user_defined_triton_kernel_transitive_closure,
    user_defined_triton_kernel_transitive_closure,
    user_defined_triton_kernel_transitive_closure_source_code,
)
from torch._inductor.dtype_propagation import DtypePropagationOpsHandler, promote_types
from torch._inductor.graph import GraphLowering
from torch._inductor.runtime import hints
from torch._inductor.runtime.hints import AutotuneHint, DeviceProperties
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import (
    get_importable_constexpr_types,
    is_triton_fp8_dtype_supported,
    run_and_get_code,
    run_and_get_kernels,
)
from torch._inductor.virtualized import ops, V
from torch.testing._internal.common_device_type import largeTensorTest
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    skipIfRocm,
)
from torch.testing._internal.inductor_utils import (
    GPU_TYPE,
    HAS_CPU,
    HAS_CUDA_AND_TRITON,
    HAS_GPU,
    HAS_GPU_AND_TRITON,
    TRITON_HAS_CPU,
)
from torch.utils._sympy.functions import FloorDiv, TruncToFloat, TruncToInt
from torch.utils._sympy.symbol import make_symbol, SymT
from torch.utils._sympy.value_ranges import ValueRanges
from torch.utils._triton import has_triton_package


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


if has_triton_package():
    import triton

    # triton.constexpr_function was introduced in Triton 3.5.
    if hasattr(triton, "constexpr_function"):
        try:
            from .triton_kernel_helpers import (
                maximum as transitive_user_maximum,
                transitive_constexpr_helper as transitive_constexpr_alias,
                transitive_jit_helper as transitive_jit_alias,
                transitive_triton_alias_kernel,
            )
        except ImportError:
            from triton_kernel_helpers import (
                maximum as transitive_user_maximum,
                transitive_constexpr_helper as transitive_constexpr_alias,
                transitive_jit_helper as transitive_jit_alias,
                transitive_triton_alias_kernel,
            )

    @triton.jit
    def _test_transitive_helper(x):
        return x + 1

    @triton.jit
    def _test_transitive_kernel(x):
        return _test_transitive_helper(x)

    if hasattr(triton, "constexpr_function"):
        # This test exercises a constexpr alias. The constexpr is used with a
        # different symbol name than its definition.
        @triton.jit
        def _test_transitive_alias_kernel(x):
            return transitive_constexpr_alias(x)

        @triton.jit
        def _test_transitive_jit_alias_kernel(x):
            return transitive_jit_alias(x)


class _FixedConfigChoices(InductorChoices):
    def __init__(self, fixed_config):
        self.fixed_config = fixed_config

    def triton_kernel_kwargs(self, kernel_cls, features, groups, kernel_kwargs):
        return {
            **kernel_kwargs,
            "fixed_config": FixedTritonConfig(self.fixed_config),
        }


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

    def test_tma_store_can_use_descriptor_local_int32_indexing(self):
        layout = SimpleNamespace(offset=0, stride=[512, 1])
        buffer = Mock(spec=ir.Buffer)
        buffer.get_layout.return_value = layout
        buffer.get_name.return_value = "buf0"
        buffer.has_tensor_output.return_value = True
        kernel = SimpleNamespace(index_dtype="tl.int64", output_node=None)
        checker = TMACompatibilityChecker(
            kernel, torch.bfloat16, for_store=True, force=False, buffer_name="buf0"
        )

        with (
            patch.object(self._graph, "try_get_buffer", return_value=buffer),
            patch.object(
                self._graph,
                "get_current_device_or_throw",
                return_value=torch.device("cuda"),
            ),
            patch.object(self._graph, "get_allocation_size", return_value=[128, 512]),
            patch.object(
                self._graph, "get_allocation_storage_size", return_value=128 * 512
            ),
        ):
            self.assertTrue(checker.can_use_32bit_indexing())

        kernel.output_node = buffer
        with (
            patch.object(self._graph, "try_get_buffer", return_value=None),
            patch.object(
                self._graph,
                "get_current_device_or_throw",
                return_value=torch.device("cuda"),
            ),
            patch.object(self._graph, "get_allocation_size", return_value=[128, 512]),
            patch.object(
                self._graph, "get_allocation_storage_size", return_value=128 * 512
            ),
        ):
            self.assertTrue(checker.can_use_32bit_indexing())

    def test_tma_descriptor_local_int32_indexing_fails_closed(self):
        int32_max = torch.iinfo(torch.int32).max
        size = sympy.Symbol("s", positive=True, integer=True)
        kernel = SimpleNamespace(index_dtype="tl.int64", output_node=None)

        def can_use(
            *,
            for_store=True,
            device="cuda",
            offset=0,
            stride=(1,),
            shape=(1,),
            storage=1,
        ):
            layout = SimpleNamespace(offset=offset, stride=stride)
            buffer = Mock(spec=ir.Buffer)
            buffer.get_layout.return_value = layout
            buffer.has_tensor_output.return_value = True
            checker = TMACompatibilityChecker(
                kernel,
                torch.bfloat16,
                for_store=for_store,
                force=False,
                buffer_name="buf0",
            )
            with (
                patch.object(self._graph, "try_get_buffer", return_value=buffer),
                patch.object(
                    self._graph,
                    "get_current_device_or_throw",
                    return_value=torch.device(device),
                ),
                patch.object(self._graph, "get_allocation_size", return_value=shape),
                patch.object(
                    self._graph,
                    "get_allocation_storage_size",
                    return_value=storage,
                ),
            ):
                return checker.can_use_32bit_indexing()

        self.assertFalse(can_use(for_store=False))
        self.assertFalse(can_use(device="cpu"))
        self.assertFalse(can_use(offset=-1))
        self.assertFalse(can_use(stride=(-1,)))
        self.assertFalse(can_use(shape=(size,), storage=size))
        self.assertFalse(can_use(shape=(int32_max + 1,), storage=int32_max + 1))
        self.assertFalse(can_use(shape=(1,), storage=int32_max + 1))

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

    def test_gen_driver_setup_default(self):
        setup = TritonKernel.gen_driver_setup
        self.assertEqual(setup("cpu"), "triton_helpers.set_driver_to_cpu()")
        self.assertEqual(setup(GPU_TYPE), "triton_helpers.set_driver_to_gpu()")

    @unittest.skipUnless(HAS_GPU_AND_TRITON, "requires GPU and Triton")
    @inductor_config.patch(force_disable_caches=True)
    def test_gen_driver_setup_override(self):
        def fn(x):
            return x.sin() + 1

        x = torch.randn(16, device=GPU_TYPE)
        _, code = run_and_get_code(torch.compile(fn, fullgraph=True), x)
        self.assertIn("triton_helpers.set_driver_to_gpu()", " ".join(code))

        torch._dynamo.reset()
        no_setup = classmethod(lambda cls, device_type: "")
        with patch.object(TritonKernel, "gen_driver_setup", no_setup):
            out, code = run_and_get_code(torch.compile(fn, fullgraph=True), x)
        self.assertEqual(out, fn(x))
        self.assertNotIn("set_driver_to_gpu", " ".join(code))

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

    def test_importable_constexpr_types_skip_namedtuple(self):
        # NamedTuples go through namedtuple_helpers, not user-module imports.
        Pair = namedtuple("Pair", ["x", "y"])
        self.assertEqual(get_importable_constexpr_types([Pair(1, 2)]), [])

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

    @unittest.skipUnless(has_triton_package(), "requires Triton")
    def test_user_defined_triton_kernel_transitive_closure(self):
        source_modules = user_defined_triton_kernel_transitive_closure(
            _test_transitive_kernel
        )
        source = user_defined_triton_kernel_transitive_closure_source_code(
            _test_transitive_kernel
        )

        # The root kernel is always retained, independent of imported modules.
        kernel_source = source_modules[0]
        self.assertIsNone(kernel_source.module_name)
        self.assertIn("def _test_transitive_kernel", kernel_source.source)

        # Dependencies retain their defining module for filtering.
        self.assertTrue(
            any(
                source_module.module_name == _test_transitive_helper.fn.__module__
                and "def _test_transitive_helper" in source_module.source
                for source_module in source_modules
            )
        )

        # The compatibility API composes collection and formatting.
        self.assertEqual(
            source,
            render_user_defined_triton_kernel_transitive_closure(source_modules),
        )

    @unittest.skipUnless(
        has_triton_package() and hasattr(triton, "constexpr_function"),
        "requires triton.constexpr_function",
    )
    def test_user_defined_triton_kernel_with_alias(self):
        module_name = transitive_constexpr_alias.fn.__module__
        source_modules = user_defined_triton_kernel_transitive_closure(
            _test_transitive_alias_kernel
        )
        # We want to simulate substituting an import for a module's source code.
        # The tricky part is how to handle aliases.
        # `from helper import *` defines `transitive_constexpr_helper`, not the
        # `transitive_constexpr_alias` referenced by the kernel.
        source_modules = [
            dataclasses.replace(
                source_module,
                source=f"from {module_name} import *\n",
            )
            if source_module.module_name == module_name
            else source_module
            for source_module in source_modules
        ]
        source = render_user_defined_triton_kernel_transitive_closure(source_modules)
        rendered_globals: dict[str, object] = {}
        exec(source, rendered_globals)

        # The renderer must emit
        # `transitive_constexpr_alias = transitive_constexpr_helper`.
        self.assertIs(
            rendered_globals["transitive_constexpr_alias"],
            transitive_constexpr_alias,
        )

    @unittest.skipUnless(
        has_triton_package() and hasattr(triton, "constexpr_function"),
        "requires triton.constexpr_function",
    )
    def test_user_defined_triton_kernel_with_jit_alias(self):
        module_name = transitive_jit_alias.fn.__module__
        source_modules = user_defined_triton_kernel_transitive_closure(
            _test_transitive_jit_alias_kernel
        )
        source_modules = [
            dataclasses.replace(
                source_module,
                source=f"from {module_name} import *\n",
            )
            if source_module.module_name == module_name
            else source_module
            for source_module in source_modules
        ]
        rendered_globals: dict[str, object] = {}
        exec(
            render_user_defined_triton_kernel_transitive_closure(source_modules),
            rendered_globals,
        )

        self.assertIs(
            rendered_globals["transitive_jit_alias"],
            transitive_jit_alias,
        )

    @unittest.skipUnless(
        has_triton_package() and hasattr(triton, "constexpr_function"),
        "requires triton.constexpr_function",
    )
    def test_user_defined_triton_kernel_with_imported_module_alias(self):
        module_name = transitive_jit_alias.fn.__module__
        source_modules = user_defined_triton_kernel_transitive_closure(
            _test_transitive_jit_alias_kernel
        )
        source = render_user_defined_triton_kernel_transitive_closure(
            source_modules,
            imported_modules=(module_name,),
        )
        self.assertNotIn("def transitive_jit_helper", source)
        self.assertIn("def _test_transitive_jit_alias_kernel", source)
        self.assertIn("transitive_jit_alias = transitive_jit_helper", source)

        rendered_globals: dict[str, object] = {}
        exec(f"from {module_name} import *\n" + source, rendered_globals)
        self.assertIs(rendered_globals["transitive_jit_alias"], transitive_jit_alias)

    @unittest.skipUnless(
        has_triton_package() and hasattr(triton, "constexpr_function"),
        "requires triton.constexpr_function",
    )
    def test_user_defined_triton_kernel_with_triton_alias(self):
        """Avoid overwriting a user helper with an aliased Triton import.

        This simulates replacing a helper module's inline source with an import.

        Unsafe:
            def maximum(...): ...
            from triton.language import maximum
            tl_maximum = maximum  # `maximum` no longer refers to the helper.

        Safe:
            from triton.language import maximum as tl_maximum
        """
        module_name = transitive_triton_alias_kernel.fn.__module__
        source_modules = user_defined_triton_kernel_transitive_closure(
            transitive_triton_alias_kernel
        )
        source_modules = [
            source_module
            for source_module in source_modules
            if source_module.module_name != module_name
        ]
        source = f"from {module_name} import maximum\n"
        source += render_user_defined_triton_kernel_transitive_closure(source_modules)
        rendered_globals: dict[str, object] = {}
        exec(source, rendered_globals)

        # The aliased Triton import must not replace the user-defined helper.
        self.assertIs(rendered_globals["tl_maximum"], triton.language.maximum)
        self.assertIs(rendered_globals["maximum"], transitive_user_maximum)

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

    def _capped_choices(self, policy, seen=None):
        class CappedChoices(InductorChoices):
            def triton_grid_policy(self, features):
                if seen is not None:
                    seen.append(features)
                return policy

            def uuid(self):
                return f"capped {policy}"

        return CappedChoices()

    def _capped_kernel(self, policy, xnumel, rnumel=None, seen=None, **kwargs):
        self._stack.enter_context(self._graph.set_current_device(torch.device("cuda")))
        numels = {"x": sympy.Integer(xnumel)}
        if rnumel is not None:
            numels["r0_"] = sympy.Integer(rnumel)
            kwargs["override_persistent_reduction"] = False
            kwargs["override_cooperative_reduction"] = False
        features = SIMDKernelFeatures([], *numels.values())
        with V.set_choices_handler(self._capped_choices(policy, seen)):
            return TritonKernel(numels, features=features, **kwargs)

    @parametrize(
        "max_xblock,xnumel,dropped",
        (
            (65536, 1 << 24, True),
            (65536, (1 << 24) + hints.TRITON_MAX_BLOCK["X"], False),
            (None, (1 << 24) + hints.CAPPED_GRID_DEFAULT_MAX_XBLOCK, True),
            # Coordinate descent can raise XBLOCK to the stock max block
            (1024, 3 * 1024, False),
            # With XBLOCK 1 the bound turns int64 once xnumel + 44 programs leave int32
            (1, 2**31 - 1 - 44, False),
            (1, 2**31 - 44, False),
            (1, 2**31, True),
        ),
    )
    @skipIfRocm(msg="TritonKernel asks for capped grids only on CUDA outside ROCm")
    def test_grid_policy_loops_over_x_blocks(self, max_xblock, xnumel, dropped):
        seen, scores, policy = [], {"x": sympy.Integer(2)}, GridPolicy(44, max_xblock)
        kernel = self._capped_kernel(policy, xnumel, seen=seen, tiling_scores=scores)
        self.assertEqual([f.tiling_scores for f in seen], [scores])
        index = "tl.int32" if xnumel < 2**31 else "tl.int64"
        self.assertEqual(kernel.index_dtype, index)
        int64_bound = xnumel + 44 > 2**31 - 1
        bound = "xnumel.to(tl.int64)" if int64_bound else "xnumel"
        pid = f"x_block.to({index})" if int64_bound else "x_block"
        self.assertIn(
            f"for x_block in tl.range(tl.program_id(0), ({bound} - 1) // XBLOCK + 1, "
            f"tl.num_programs(0)):\n    xoffset = {pid} * XBLOCK\n    xindex = ",
            kernel.body.getvalue(),
        )
        # The mask drops at multiples of the larger of max_xblock and the stock block
        mask = "tl.full([XBLOCK], True, tl.int1)" if dropped else "xindex < xnumel"
        self.assertIn(f"\n    xmask = {mask}", kernel.body.getvalue())
        limit = max_xblock or hints.CAPPED_GRID_DEFAULT_MAX_XBLOCK
        self.assertEqual(kernel.max_block("x"), max(limit, hints.TRITON_MAX_BLOCK["X"]))
        self.assertEqual(kernel._get_grid_type().__name__, "CappedGrid1D")
        meta = kernel.inductor_meta_per_kernel()
        self.assertEqual(meta["grid_cap"], 44)
        self.assertEqual(meta.get("grid_max_xblock"), max_xblock)

    @skipIfRocm(msg="TritonKernel asks for capped grids only on CUDA outside ROCm")
    def test_grid_policy_looped_reduction_runs_inside_the_x_loop(self):
        kernel = self._capped_kernel(GridPolicy(66), 4096, 65536)
        self.assertEqual(kernel._get_grid_type().__name__, "CappedGrid1D")
        # A looped reduction keeps the stock max block, so 4,096 rows drop the x mask
        self.assertEqual(kernel.max_block("x"), hints.TRITON_MAX_BLOCK["X"])
        self.assertIn("    xmask = tl.full([XBLOCK], True", kernel.body.getvalue())
        # Two passes over the row, as when a second pass reads the first's statistics
        with kernel:
            kernel.compute.writeline("tmp0 = 1.0")
            kernel.codegen_body()
            kernel.compute.writeline("tmp1 = 2.0")
            kernel.codegen_body()
        lines = kernel.body.getvalue().splitlines()
        header = [line.startswith("for x_block") for line in lines].index(True)
        passes = [i for i, line in enumerate(lines) if "for r0_offset in" in line]
        self.assertEqual(len(passes), 2)
        for i in passes:
            self.assertTrue(i > header and lines[i].startswith("    for r0_offset"))

    @parametrize(
        "case",
        [
            "persistent",
            "mix_order",
            "no_x_dim",
            "cooperative",
            "two_reduction_dims",
            "two_dimensional",
            "staged",
            "native_matmul",
            "combo",
            "fixed_config",
            "hint_override",
            "one_x_element",
            "subclass",
            "grandchild",
            "block_ptr",
            "tensor_descriptor",
            "multi_kernel",
            "combo_autotune",
            "rocm",
            "xpu",
            "cpu",
            "no_device",
            "pointwise",
            "reduction",
            "opted_in",
        ],
    )
    @skipIfRocm(msg="TritonKernel asks for capped grids only on CUDA outside ROCm")
    def test_grid_policy_asked_only_for_eligible_kernels(self, case):
        # One excluding condition per case; the last three cases are asked
        x, r, y = sympy.Integer(1024), sympy.Integer(512), sympy.Integer(64)
        opted_in = type("OptedIn", (TritonKernel,), {"supports_grid_policy": True})

        def pointwise(cls=TritonKernel, numel=x, schedule=None, **kwargs):
            features = SIMDKernelFeatures([], numel, indexing_node_schedule=schedule)
            return cls({"x": numel}, features=features, **kwargs)

        def reduction(persistent, cooperative=False, rnumels=("r0_",), **kwargs):
            # Without mask optimization a persistent reduction needs no GPU here
            return TritonKernel(
                {"x": x, **dict.fromkeys(rnumels, r)},
                features=SIMDKernelFeatures([], x, r ** len(rnumels)),
                optimize_mask=False,
                override_persistent_reduction=persistent,
                override_cooperative_reduction=cooperative,
                **kwargs,
            )

        def native_matmul():
            matmul = property(lambda self: True, lambda self, value: None)
            with patch.object(opted_in, "is_native_matmul", matmul, create=True):
                return pointwise(opted_in)

        factories = {
            "persistent": lambda: reduction(True),
            "mix_order": lambda: reduction(True, mix_order_reduction=True),
            # Dropping the x dimension needs a fixed config (XBLOCK 1), itself excluded
            "no_x_dim": lambda: reduction(
                True, fixed_config=FixedTritonConfig({"XBLOCK": 1, "R0_BLOCK": 512})
            ),
            "cooperative": lambda: reduction(False, cooperative=True),
            "two_reduction_dims": lambda: reduction(False, rnumels=("r0_", "r1_")),
            "two_dimensional": lambda: TritonKernel(
                {"y": y, "x": x}, features=SIMDKernelFeatures([], x * y)
            ),
            "staged": lambda: pointwise(schedule=[]),
            "native_matmul": native_matmul,
            "combo": lambda: pointwise(is_combo_kernel=True),
            "fixed_config": lambda: pointwise(
                fixed_config=FixedTritonConfig({"XBLOCK": 128})
            ),
            "hint_override": lambda: pointwise(hint_override=1024),
            "one_x_element": lambda: pointwise(numel=sympy.S.One),
            "subclass": lambda: pointwise(type("Subclass", (TritonKernel,), {})),
            "grandchild": lambda: pointwise(type("Grandchild", (opted_in,), {})),
            "reduction": lambda: reduction(False),
            "opted_in": lambda: pointwise(opted_in),
        }
        configs = {
            "block_ptr": {"triton.use_block_ptr": True},
            "tensor_descriptor": {"triton.use_tensor_descriptor": True},
            "multi_kernel": {"triton.multi_kernel": 1},
            "combo_autotune": {
                "combo_kernels": True,
                "combo_kernel_compile_time_autotune": True,
            },
        }
        device = torch.device(case if case in ("xpu", "cpu") else "cuda")
        seen = []
        with (
            patch.object(torch.version, "hip", "6.4" if case == "rocm" else None),
            patch.object(
                self._graph, "current_device", None if case == "no_device" else device
            ),
            inductor_config.patch(configs.get(case, {})),
            V.set_choices_handler(self._capped_choices(GridPolicy(44), seen)),
        ):
            kernel = factories.get(case, pointwise)()
        asked = case in ("pointwise", "reduction", "opted_in")
        self.assertEqual(len(seen), int(asked))
        self.assertEqual(kernel.grid_policy, GridPolicy(44) if asked else None)
        self.assertEqual(bool(kernel.no_x_dim), case == "no_x_dim")

    @skipIfRocm(msg="TritonKernel asks for capped grids only on CUDA outside ROCm")
    def test_grid_policy_hook_must_return_a_policy_or_none(self):
        with self.assertRaisesRegex(
            TypeError, r"CappedChoices\.triton_grid_policy must return .+, got 44"
        ):
            self._capped_kernel(44, 1 << 24)

    @skipIfRocm(msg="TritonKernel asks for capped grids only on CUDA outside ROCm")
    def test_grid_policy_none_keeps_the_stock_grid_and_pdl(self):
        stock = self._capped_kernel(None, 1 << 24)
        self.assertIn("xoffset = tl.program_id(0) * XBLOCK", stock.body.getvalue())
        self.assertEqual(stock._get_grid_type().__name__, "Grid1D")
        capped = self._capped_kernel(GridPolicy(44, 65536), 1 << 24)
        # With PDL on and compute capability 9, only the stock kernel uses PDL
        with (
            inductor_config.patch({"triton.enable_pdl": True}),
            patch("torch.cuda.get_device_capability", return_value=(9, 0)),
        ):
            for kernel, pdl in ((stock, True), (capped, False)):
                with kernel:
                    self.assertEqual(bool(TritonKernel._enable_pdl_codegen()), pdl)
                    launch_pdl = TritonKernel.triton_meta_common()["launch_pdl"]
                    self.assertEqual(bool(launch_pdl), pdl)

    def test_grid_policy_validates_its_fields(self):
        for programs in (0, True, 44.0):
            with self.assertRaisesRegex(ValueError, "an int of at least 1"):
                GridPolicy(programs)
        self.assertEqual(GridPolicy(1).programs, 1)
        for max_xblock in (0, 3000, True, 4096.0):
            with self.assertRaisesRegex(ValueError, "an int power of 2"):
                GridPolicy(44, max_xblock)
        self.assertIsNone(GridPolicy(44).max_xblock)

    def _compile_capped(self, fn, *args, policy=None, **compile_kwargs):
        # The FX graph caches do not key on a policy set only on this thread
        handler = self._capped_choices(policy or GridPolicy(44, 65536))
        with (
            inductor_config.patch(fx_graph_cache=False, fx_graph_remote_cache=False),
            V.set_choices_handler(handler),
        ):
            compiled = torch.compile(fn, **compile_kwargs)
            result, code = run_and_get_code(compiled, *args)
        return compiled, result, "\n".join(code)

    @parametrize(
        "xnumel,coordesc", ((1 << 24, False), (1 << 24, True), (2**31 - 10, False))
    )
    @unittest.skipUnless(HAS_CUDA_AND_TRITON, "requires CUDA and Triton")
    # Twice the buffer must be free, so the 2**24 case runs on busy GPUs too
    @largeTensorTest(
        lambda self, xnumel, **_: 2 * xnumel, device=GPU_TYPE, inductor=True
    )
    @inductor_config.patch({"triton.autotune_pointwise": False})
    @skipIfRocm(msg="TritonKernel asks for capped grids only on CUDA outside ROCm")
    def test_grid_policy_in_place_kernel_runs_each_block_once(self, xnumel, coordesc):
        # One config doubles an int8 view in place: each element reads 2 (4 if its block
        # ran twice) and the guard after it keeps -1. 2**24 drops the x mask, also
        # with the XBLOCK that coordinate descent picks from the capped candidate
        def double(x):
            x.mul_(2)

        buf = torch.ones(xnumel + (1 << 17), device=GPU_TYPE, dtype=torch.int8)
        buf[xnumel:] = -1
        x, guard = buf[:xnumel], buf[xnumel:]
        with inductor_config.patch(coordinate_descent_tuning=coordesc):
            _, _, code = self._compile_capped(double, x, dynamic=False)
        dropped = xnumel == 1 << 24
        mask = "tl.full([XBLOCK], True" if dropped else "xindex < xnumel"
        pid = "x_block" if dropped else "x_block.to(tl.int32)"
        self.assertIn(f"xmask = {mask}", code)
        self.assertIn(f"xoffset = {pid} * XBLOCK", code)
        self.assertEqual((x.min().item(), x.max().item()), (2, 2))
        self.assertEqual((guard.min().item(), guard.max().item()), (-1, -1))

    @parametrize("rows", (4096, 4096 + 5))
    @unittest.skipUnless(HAS_CUDA_AND_TRITON, "requires CUDA and Triton")
    @inductor_config.patch({"split_reductions": False})
    @skipIfRocm(msg="TritonKernel asks for capped grids only on CUDA outside ROCm")
    def test_grid_policy_in_place_reduction_runs_each_block_once(self, rows):
        # One kernel adds each row's sum to the row in place in a second pass (a block
        # run twice adds again). 4,096 rows, a multiple of the stock max block, drop the
        # x mask, and 4,101 keep it for a masked last block; guard rows keep -1, and
        # integer values keep fp32 sums exact.
        def add_row_sums(x):
            x.add_(x.sum(1, keepdim=True))

        cols = 4096
        buf = torch.randint(-8, 8, (rows + 64, cols), device=GPU_TYPE).float()
        buf[rows:] = -1
        x = buf[:rows]
        expected = x + x.sum(1, keepdim=True)
        _, _, code = self._compile_capped(add_row_sums, x, policy=GridPolicy(66))
        self.assertEqual(code.count("@triton.jit"), 1)
        self.assertNotIn(f"empty_strided_cuda(({rows}, {cols})", code)
        self.assertGreaterEqual(code.count("\n        for r0_offset in tl.range("), 2)
        dropped = rows % hints.TRITON_MAX_BLOCK["X"] == 0
        mask = "tl.full([XBLOCK], True" if dropped else "xindex < xnumel"
        self.assertIn(f"xmask = {mask}", code)
        self.assertEqual(x, expected, atol=0, rtol=0)
        self.assertTrue((buf[rows:] == -1).all())

    @parametrize("cpp_wrapper", (False, True))
    @unittest.skipUnless(HAS_CUDA_AND_TRITON, "requires CUDA and Triton")
    @skipIfRocm(msg="TritonKernel asks for capped grids only on CUDA outside ROCm")
    def test_grid_policy_dynamic_size_below_and_above_its_hint(self, cpp_wrapper):
        x = torch.randn(1 << 22, device=GPU_TYPE)
        torch._dynamo.mark_dynamic(x, 0)
        with inductor_config.patch(cpp_wrapper=cpp_wrapper):
            compiled, actual, code = self._compile_capped(lambda t: t * 2, x)
        # The C++ wrapper computes the capped grid at each launch
        self.assertEqual("std::min<int64_t>(" in code, cpp_wrapper)
        # An int64 bound, an int32 index, and the runtime told the size hint may vary
        self.assertIn("(xnumel.to(tl.int64) - 1) // XBLOCK + 1", code)
        self.assertIn("    xoffset = x_block.to(tl.int32) * XBLOCK", code)
        self.assertIn("'grid_dynamic_xnumel': True", code)
        self.assertEqual(actual, x * 2, atol=0, rtol=0)
        # The same kernel, without a recompile, below the cap, uneven and far above
        with torch._dynamo.config.patch(error_on_recompile=True):
            for numel in (5000, (1 << 20) + 3, (1 << 24) + 3):
                y = torch.randn(numel, device=GPU_TYPE)
                self.assertEqual(compiled(y), y * 2, atol=0, rtol=0)

    @unittest.skipUnless(HAS_CUDA_AND_TRITON, "requires CUDA and Triton")
    @skipIfRocm(msg="TritonKernel asks for capped grids only on CUDA outside ROCm")
    def test_grid_policy_from_a_registered_handler_keys_the_fx_graph_cache(self):
        # Registering a policy misses the FX graph cache, then hits the capped kernel
        from torch._dynamo.utils import counters
        from torch._inductor import choices, virtualized

        x = torch.randn(1 << 24, device=GPU_TYPE)
        key = virtualized._choices._key
        prior = virtualized.threadlocal.__dict__.get(key)

        def compile_and_run():
            torch._dynamo.reset()
            actual, code = run_and_get_code(torch.compile(lambda t: t * 2), x)
            self.assertEqual(actual, x * 2, atol=0, rtol=0)
            return "\n".join(code)

        with inductor_config.patch(fx_graph_cache=True, fx_graph_remote_cache=False):
            self.assertNotIn("for x_block", compile_and_run())
            policy = self._capped_choices(GridPolicy(44))
            choices.register_inductor_choices("policy", lambda: policy)
            try:
                for _ in range(2):
                    code = compile_and_run()
                    self.assertIn("for x_block in tl.range(tl.program_id(0), ", code)
            finally:
                choices.unregister_inductor_choices("policy")
                # Unregistering installs a newly built handler: restore the prior one
                virtualized.threadlocal.__dict__.pop(key, None)
                if prior is not None:
                    setattr(virtualized.threadlocal, key, prior)
        self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 2)
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)

    @unittest.skipUnless(HAS_CUDA_AND_TRITON, "requires CUDA and Triton")
    @skipIfRocm(msg="TritonKernel asks for capped grids only on CUDA outside ROCm")
    def test_grid_policy_fx_ir_wrapper_caps_a_symbolic_grid(self):
        from torch._inductor.codegen import common
        from torch._inductor.codegen.wrapper_fxir import WrapperFxCodegen

        gms = []

        class RecordingFxCodegen(WrapperFxCodegen):
            def compile_graph(self, gm):
                gms.append(gm)
                return super().compile_graph(gm)

        backend = common.DeviceCodegen(TritonScheduling, RecordingFxCodegen, None)
        x = torch.randn(1 << 22, device=GPU_TYPE)
        with patch.dict(common.device_codegens, {GPU_TYPE: backend}):
            _, actual, _ = self._compile_capped(lambda t: t * 2, x, dynamic=True)
        self.assertEqual(actual, x * 2, atol=0, rtol=0)
        (gm,) = gms
        (node,) = gm.graph.find_nodes(
            op="call_function",
            target=torch.ops.higher_order.triton_kernel_wrapper_mutation,
        )
        grid = node.kwargs["grid"][0][0]
        self.assertIs(grid.target, torch.sym_min)
        # One program for one element; the cap for 2**22 elements, 64 or more blocks
        expr = grid.meta["val"].node.expr
        (size,) = expr.free_symbols
        self.assertEqual([int(expr.subs(size, n)) for n in (1, 1 << 22)], [1, 44])

    @parametrize(
        "reduction_type,identity,reduction_fn",
        (("sum", "0", "tl.sum"), ("prod", "1", "triton_helpers.prod")),
    )
    @parametrize(
        "xnumel,optimize_mask",
        ((sympy.Integer(40961), False), (sympy.Integer(1), True)),
    )
    def test_mix_order_partial_accumulate_masks_x(
        self, reduction_type, identity, reduction_fn, xnumel, optimize_mask
    ):
        self._stack.enter_context(self._graph.set_current_device(torch.device("cpu")))
        rnumel = sympy.Integer(129)
        kernel = TritonKernel(
            {"x": xnumel, "r0_": rnumel},
            features=SIMDKernelFeatures([], xnumel, rnumel),
            mix_order_reduction=True,
            optimize_mask=optimize_mask,
            override_persistent_reduction=True,
            override_cooperative_reduction=False,
        )

        with kernel:
            x_tree, r_tree = kernel.range_trees
            xvalue = ops.index_expr(x_tree.full_range().symbol(), torch.float32)
            rvalue = ops.index_expr(r_tree.full_range().symbol(), torch.float32)
            value = ops.add(xvalue, rvalue)
            value = ops.add(value, ops.constant(0.25, torch.float32))
            ops.partial_accumulate("out", reduction_type, value, {})
            kernel.codegen_body()

        code = kernel.body.getvalue()
        masked_reduction = re.compile(
            rf"(?P<masked>tmp\d+) = tl\.where\(xmask, tmp\d+, {identity}\)\n"
            rf"\s+tmp\d+ = {re.escape(reduction_fn)}\((?P=masked), 0\)"
        )
        self.assertRegex(code, masked_reduction)

    def test_mix_order_rejects_tma_xblock_above_heuristic_limit(self):
        xnumel = sympy.Integer(8192)
        rnumel = sympy.Integer(8)
        with self._graph.set_current_device(torch.device("cpu")):
            kernel = TritonKernel(
                {"x": xnumel, "r0_": rnumel},
                features=SIMDKernelFeatures([], xnumel, rnumel),
                mix_order_reduction=True,
                override_persistent_reduction=True,
                override_cooperative_reduction=False,
            )
        kernel.rsplit_size = 64
        xblock = TritonSymbols.block_sizes[SymT.XBLOCK]
        block_params = BlockParameters(
            shape=[rnumel, xnumel],
            block_shape=[rnumel, FloorDiv(xblock, rnumel)],
            strides=[xnumel, sympy.Integer(1)],
            offsets=[sympy.Integer(0), sympy.Integer(0)],
        )

        checker = TMACompatibilityChecker(
            kernel, torch.float32, for_store=False, force=False
        )
        with self._graph.set_current_device(torch.device("cuda")):
            compatible = checker.are_block_parameters_compatible(block_params)
        self.assertFalse(compatible)
        self.assertEqual(kernel.tma_min_block_sizes, {})

    def test_tma_metadata_uses_own_removed_buffers(self):
        xnumel = sympy.Integer(4096)
        rnumel = sympy.Integer(128)
        with self._graph.set_current_device(torch.device("cpu")):
            kernel = TritonKernel(
                {"x": xnumel, "r0_": rnumel},
                features=SIMDKernelFeatures([], xnumel, rnumel),
                mix_order_reduction=True,
                override_persistent_reduction=True,
                override_cooperative_reduction=False,
            )
            ambient_kernel = TritonKernel(
                {"x": xnumel},
                features=SIMDKernelFeatures([], xnumel, sympy.Integer(1)),
                override_cooperative_reduction=False,
            )

        removed_name = "buf0"
        kernel._device_tma_buffers.add(removed_name)
        kernel._record_tma_min_block_size(removed_name, "XBLOCK", 4)
        kernel.removed_buffers.add(removed_name)
        with V.set_kernel_handler(ambient_kernel):
            metadata = kernel.inductor_meta_per_kernel()

        self.assertNotIn("uses_tma", metadata)
        self.assertNotIn("uses_device_tma", metadata)
        self.assertNotIn("tma_min_block_sizes", metadata)

    @parametrize(
        "split_size,fixed_config,error",
        (
            (
                48,
                {"XBLOCK": 32},
                "RSPLIT_SIZE=48 is incompatible with fixed XBLOCK=32",
            ),
            (
                18,
                {"XBLOCK": 2, "RSPLIT_SIZE": 32, "NUM_STAGES": 1},
                "fixed RSPLIT_SIZE=32 does not match scheduled RSPLIT_SIZE=18",
            ),
            (
                18,
                {"XBLOCK": 3, "NUM_STAGES": 1},
                "fixed XBLOCK=3 must be a positive power of two",
            ),
        ),
    )
    def test_mix_order_rejects_incompatible_fixed_config(
        self, split_size, fixed_config, error
    ):
        xnumel = sympy.Integer(40961)
        rnumel = sympy.Integer(129)
        with (
            self._graph.set_current_device(torch.device("cpu")),
            V.set_choices_handler(_FixedConfigChoices(fixed_config)),
            self.assertRaisesRegex(ValueError, error),
        ):
            TritonScheduling(None)._create_kernel_for_mix_order_reduction(
                SIMDKernelFeatures([], xnumel, rnumel),
                split_size=split_size,
            )

    @parametrize(
        "fixed_config",
        (
            {"XBLOCK": 1, "RSPLIT_SIZE": 18, "NUM_STAGES": 1},
            {"XBLOCK": 2, "NUM_STAGES": 1},
        ),
    )
    def test_mix_order_normalizes_fixed_config(self, fixed_config):
        xnumel = sympy.Integer(40961)
        rnumel = sympy.Integer(129)
        with (
            self._graph.set_current_device(torch.device("cpu")),
            V.set_choices_handler(_FixedConfigChoices(fixed_config)),
        ):
            kernel = TritonScheduling(None)._create_kernel_for_mix_order_reduction(
                SIMDKernelFeatures([], xnumel, rnumel),
                split_size=18,
            )

        self.assertFalse(kernel.no_x_dim)
        self.assertEqual(kernel.fixed_config["RSPLIT_SIZE"], 18)

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

    @parametrize("scheduled", [False, True])
    @parametrize(
        "shape, stride, expected",
        [
            ((32, 32), (32, 1), True),
            ((2**30 - 1,), (1,), True),
            ((2**30,), (1,), False),
            ((2,), (2**30,), False),
        ],
    )
    def test_constant_pointer_range_survives_scheduling(
        self, scheduled, shape, stride, expected
    ):
        self._graph.constants["weight"] = torch.empty_strided(
            shape, stride, dtype=torch.float16, device="meta"
        )
        scheduler = Mock()
        scheduler.get_buffer_layout.side_effect = KeyError("weight")
        arg = TensorArg(name="in_ptr0", buffer="weight", dtype=torch.float16)

        with patch.object(self._graph, "scheduler", scheduler if scheduled else None):
            self.assertEqual(triton_utils._is_tensor_within_2gb(arg), expected)

    @parametrize("scheduled", [False, True])
    @parametrize(
        "storage_numel, expected",
        [(2**30 - 1, True), (2**30, False), (2**30 + 1, False)],
    )
    def test_constant_pointer_range_checks_backing_storage(
        self, scheduled, storage_numel, expected
    ):
        self._graph.constants["weight"] = torch.empty(
            storage_numel, dtype=torch.float16, device="meta"
        )[:32]
        scheduler = Mock()
        scheduler.get_buffer_layout.side_effect = KeyError("weight")
        arg = TensorArg(name="in_ptr0", buffer="weight", dtype=torch.float16)

        with patch.object(self._graph, "scheduler", scheduler if scheduled else None):
            self.assertEqual(triton_utils._is_tensor_within_2gb(arg), expected)

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

    def _skip_unless_annotation_is_literal(self):
        """Only the V4 descriptor puts ``tt.pointer_range`` in the generated code.

        Earlier versions carry it as a ``pointer_range_32`` field on a descriptor
        object, so asserting on the literal string would pass whatever the code did.
        """
        from torch._inductor.utils import (
            get_triton_attrs_descriptor_version,
            TritonAttrsDescriptorVersion,
        )

        if (
            get_triton_attrs_descriptor_version()
            != TritonAttrsDescriptorVersion.V4_DICT
        ):
            self.skipTest(
                "tt.pointer_range is only literal with the V4 attrs descriptor"
            )

    @unittest.skipUnless(torch.version.hip is not None, "pointer_range_32 is HIP-only")
    @unittest.skipUnless(HAS_GPU_AND_TRITON, "requires GPU and Triton")
    @inductor_config.patch({"freezing": True, "triton.emit_pointer_range_32": True})
    def test_constant_pointer_range_in_generated_code(self):
        self._skip_unless_annotation_is_literal()

        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer(
                    "weight", torch.randn(32, 32, device=GPU_TYPE, dtype=torch.float16)
                )

            def forward(self, x):
                return x + self.weight

        x = torch.randn(32, 32, device=GPU_TYPE, dtype=torch.float32)
        with torch.no_grad():
            _, kernels = run_and_get_kernels(
                torch.compile(Module().eval(), fullgraph=True), x, remove_quote=True
            )
        self.assertEqual(len(kernels), 1)
        meta = next(
            keyword.value
            for node in ast.walk(ast.parse(kernels[0]))
            if isinstance(node, ast.Call)
            for keyword in node.keywords
            if keyword.arg == "triton_meta"
        )
        meta = {key.value: value for key, value in zip(meta.keys, meta.values)}
        signature = ast.literal_eval(meta["signature"])
        configs = ast.literal_eval(meta["configs"])
        # Only the frozen weight has an fp16 pointer; input and output are fp32.
        weight_indices = [
            idx for idx, dtype in enumerate(signature.values()) if dtype == "*fp16"
        ]
        self.assertEqual(len(weight_indices), 1)
        self.assertIn(["tt.pointer_range", 32], configs[0][(weight_indices[0],)])

    @staticmethod
    def _flex_inputs(requires_grad=False):
        return [
            torch.randn(
                1,
                4,
                256,
                64,
                device=GPU_TYPE,
                dtype=torch.float16,
                requires_grad=requires_grad,
            )
            for _ in range(3)
        ]

    @unittest.skipUnless(torch.version.hip is not None, "pointer_range_32 is HIP-only")
    @unittest.skipUnless(HAS_GPU_AND_TRITON, "requires GPU and Triton")
    def test_pointer_range_applied_to_template_kernel(self):
        """A template kernel without atomics must still be tagged.

        The positive half of the two tests below: without it, a regression that
        dropped the annotation for every template kernel -- losing buffer ops on
        every matmul and attention kernel on ROCm -- would leave them all green.
        """
        self._skip_unless_annotation_is_literal()
        from torch.nn.attention.flex_attention import flex_attention

        q, k, v = self._flex_inputs()
        _, kernels = run_and_get_kernels(
            torch.compile(flex_attention, fullgraph=True), q, k, v, remove_quote=True
        )
        templates = [x for x in kernels if "triton_tem_" in x]
        self.assertTrue(templates, "no template kernel was generated")
        self.assertTrue(
            any("tt.pointer_range" in x for x in templates),
            "template kernel without atomics should carry tt.pointer_range",
        )

    @unittest.skipUnless(torch.version.hip is not None, "pointer_range_32 is HIP-only")
    @unittest.skipUnless(HAS_GPU_AND_TRITON, "requires GPU and Triton")
    @inductor_config.patch("triton.emit_pointer_range_32", False)
    def test_pointer_range_disabled_for_template_kernels(self):
        """The config flag must reach template kernels, not just the pointwise path.

        Template kernels build their own triton_meta in
        TritonTemplateKernel.jit_lines() rather than going through
        TritonKernel.codegen_kernel(), so the two can disagree.
        """
        self._skip_unless_annotation_is_literal()
        from torch.nn.attention.flex_attention import flex_attention

        q, k, v = self._flex_inputs()
        _, kernels = run_and_get_kernels(
            torch.compile(flex_attention, fullgraph=True), q, k, v, remove_quote=True
        )
        templates = [x for x in kernels if "triton_tem_" in x]
        self.assertTrue(templates, "no template kernel was generated")
        for kernel in templates:
            self.assertNotIn("tt.pointer_range", kernel)

    @unittest.skipUnless(torch.version.hip is not None, "pointer_range_32 is HIP-only")
    @unittest.skipUnless(HAS_GPU_AND_TRITON, "requires GPU and Triton")
    def test_pointer_range_not_applied_to_template_kernel_with_atomics(self):
        """Kernels using atomics must not be tagged, including template kernels.

        A score_mod capturing a tensor that requires grad makes the flex_attention
        backward accumulate into it with tl.atomic_add. Tagging that kernel lets the
        backend pick buffer atomics, which are far slower than global atomics when
        many lanes target the same address.
        """
        self._skip_unless_annotation_is_literal()
        from torch.nn.attention.flex_attention import flex_attention

        q, k, v = self._flex_inputs(requires_grad=True)
        bias = torch.randn(4, device=GPU_TYPE, dtype=torch.float16, requires_grad=True)

        def score_mod(score, b, h, q_idx, kv_idx):
            return score + bias[h]

        def fwd_bwd(q, k, v):
            out = torch.compile(flex_attention, fullgraph=True)(
                q, k, v, score_mod=score_mod
            )
            out.sum().backward()
            return out

        _, kernels = run_and_get_kernels(fwd_bwd, q, k, v, remove_quote=True)
        atomic = [x for x in kernels if re.search(r"tl\.atomic_\w+", x)]
        # the whole point is the *template* kernel, so a pointwise atomic kernel
        # alone would not exercise this
        self.assertTrue(
            any("triton_tem_" in x for x in atomic),
            "no template kernel using atomics was generated",
        )
        for kernel in atomic:
            self.assertNotIn("tt.pointer_range", kernel)

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

    def test_mtia_rank_zero_tensor_signature_uses_pointer(self):
        class FakeGraph:
            mutated_buffers = set()
            wrapper_code = SimpleNamespace(preserve_zero_dim_tensor_args=True)

            def is_unspec_arg(self, name):
                return True

            def get_current_device_or_throw(self):
                return torch.device("cpu")

        arg = TensorArg(name="in_ptr0", buffer="buf0", dtype=torch.float32)
        with V.set_graph_handler(FakeGraph()):
            self.assertFalse(triton_utils.should_unwrap_unspec_arg(arg.buffer))
            self.assertEqual(triton_utils.signature_of(arg, size_dtype=None), "*fp32")

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
        HAS_GPU_AND_TRITON or (HAS_CPU and TRITON_HAS_CPU),
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
        HAS_GPU_AND_TRITON or (HAS_CPU and TRITON_HAS_CPU),
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

    def test_autotune_hints_meta_is_ordered(self):
        # inductor_meta is rendered into the kernel source, so the hints must
        # not depend on set iteration order.
        xnumel, rnumel = sympy.Integer(64), sympy.Integer(8192)
        kernel = TritonKernel(
            {"x": xnumel, "r0_": rnumel},
            features=SIMDKernelFeatures([], xnumel, rnumel),
            override_persistent_reduction=False,
            override_cooperative_reduction=False,
        )
        kernel.autotune_hints.add(AutotuneHint.SCALAR_ACCUMULATORS)
        kernel.autotune_hints.add(AutotuneHint.ONE_ELEMENT_PER_THREAD)
        with V.set_kernel_handler(kernel):
            hints = kernel.inductor_meta_per_kernel()["autotune_hints"]
        self.assertEqual(
            hints,
            (AutotuneHint.ONE_ELEMENT_PER_THREAD, AutotuneHint.SCALAR_ACCUMULATORS),
        )
        self.assertEqual(
            repr(hints),
            "(AutotuneHint.ONE_ELEMENT_PER_THREAD, AutotuneHint.SCALAR_ACCUMULATORS)",
        )


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_CPU or HAS_GPU:
        run_tests("sympy")
