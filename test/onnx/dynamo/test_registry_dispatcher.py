# Owner(s): ["module: onnx"]
"""Unit tests for the internal registration wrapper module."""

import logging
import operator
from typing import Sequence

import torch
import torch.fx
from onnxscript.function_libs.torch_lib import ops  # type: ignore[import]
from torch.onnx._internal.diagnostics import infra
from torch.onnx._internal.fx import onnxfunction_dispatcher, registration
from torch.testing._internal import common_utils


class TestRegistrationDecorators(common_utils.TestCase):
    def setUp(self) -> None:
        self.registry = registration.OnnxRegistry()

    def tearDown(self) -> None:
        self.registry._registry.pop("test::test_op", None)

    def test_onnx_symbolic_registers_function(self):
        self.assertFalse(self.registry.is_registered_op("test::test_op", 9))

        def test(g, x):
            return g.op("test", x)

        self.registry.register("test::test_op", 9, test, custom=True)
        self.assertTrue(self.registry.is_registered_op("test::test_op", 9))
        function_group = self.registry.get_function_group("test::test_op")
        assert function_group is not None
        self.assertEqual(function_group.get(9), {test})

    def test_custom_onnx_symbolic_registers_custom_function(self):
        self.assertFalse(self.registry.is_registered_op("test::test_op", 9))

        def test(g, x):
            return g.op("test", x)

        self.registry.register("test::test_op", 9, test, custom=True)

        self.assertTrue(self.registry.is_registered_op("test::test_op", 9))
        function_group = self.registry.get_function_group("test::test_op")
        assert function_group is not None
        self.assertEqual(function_group.get(9), {test})

    def test_custom_onnx_symbolic_overrides_existing_function(self):
        self.assertFalse(self.registry.is_registered_op("test::test_op", 9))

        def test_original():
            return "original"

        self.registry.register("test::test_op", 9, test_original, custom=True)

        self.assertTrue(self.registry.is_registered_op("test::test_op", 9))

        def test_custom():
            return "custom"

        self.registry.register("test::test_op", 9, test_custom, custom=True)

        function_group = self.registry.get_function_group("test::test_op")
        assert function_group is not None
        self.assertEqual(function_group.get(9), {test_custom, test_original})


@common_utils.instantiate_parametrized_tests
class TestDispatcher(common_utils.TestCase):
    def setUp(self):
        self.registry = registration.OnnxRegistry()
        # TODO: remove this once we have a better way to do this
        logger = logging.getLogger("TestDispatcher")
        self.diagnostic_context = infra.DiagnosticContext(
            "torch.onnx.dynamo_export", torch.__version__, logger=logger
        )
        self.dispatcher = onnxfunction_dispatcher.OnnxFunctionDispatcher(
            self.registry, self.diagnostic_context
        )

        self.op_overload = torch.fx.Node(
            graph=torch.fx.Graph(),
            name="aten::add.Tensor",
            op="call_function",
            target=torch.ops.aten.add.Tensor,
            args=(torch.tensor(3), torch.tensor(4)),
            kwargs={},
        )
        self.op_overloadpacket = torch.fx.Node(
            graph=torch.fx.Graph(),
            name="aten::sym_size",
            op="call_function",
            target=torch.ops.aten.sym_size,
            args=(),
            kwargs={},
        )
        self.unsupported_op_overloadpacket = torch.fx.Node(
            graph=torch.fx.Graph(),
            name="aten::add",
            op="call_function",
            target=torch.ops.aten.add,
            args=(),
            kwargs={},
        )
        self.builtin_op = torch.fx.Node(
            graph=torch.fx.Graph(),
            name="builtin_add",
            op="call_function",
            target=operator.add,
            args=(1, 2),
            kwargs={},
        )
        self.unsupported_builtin_op = torch.fx.Node(
            graph=torch.fx.Graph(),
            name="builtin_add",
            op="call_function",
            target=operator.add,
            args=("A", "B"),
            kwargs={},
        )

        def made_up_test():
            return

        self.unsupported_op_node_target = torch.fx.Node(
            graph=torch.fx.Graph(),
            name="aten::made_up_node",
            op="call_function",
            target=made_up_test,
            args=(),
            kwargs={},
        )

    def test_get_aten_name(self):
        self.assertEqual(
            self.dispatcher.get_aten_name(self.op_overload, self.diagnostic_context),
            "aten::add.Tensor",
        )
        self.assertEqual(
            self.dispatcher.get_aten_name(
                self.op_overloadpacket, self.diagnostic_context
            ),
            "aten::sym_size",
        )
        self.assertEqual(
            self.dispatcher.get_aten_name(self.builtin_op, self.diagnostic_context),
            "aten::add",
        )
        with self.assertRaises(RuntimeError):
            self.dispatcher.get_aten_name(
                self.unsupported_op_overloadpacket, self.diagnostic_context
            )
        with self.assertRaises(RuntimeError):
            self.dispatcher.get_aten_name(
                self.unsupported_op_node_target, self.diagnostic_context
            )
        with self.assertRaises(RuntimeError):
            self.dispatcher.get_aten_name(
                self.unsupported_builtin_op, self.diagnostic_context
            )

    def test_get_function_overloads(self):
        # Test fall back to default op name
        self.assertEqual(
            self.dispatcher.get_function_overloads(
                self.op_overload, "aten::add.Tensor", self.diagnostic_context
            ),
            self.dispatcher.get_function_overloads(
                self.unsupported_op_overloadpacket, "aten::add", self.diagnostic_context
            ),
        )
        with self.assertRaises(RuntimeError):
            self.dispatcher.get_function_overloads(
                self.unsupported_op_node_target, "aten::test", self.diagnostic_context
            )

    def test_warnings_in_find_the_perfect_or_nearest_match_onnxfunction(self):
        with self.assertWarnsOnceRegex(
            UserWarning,
            "A perfect matched Opchema is not found in torchlib for aten::add",
        ):
            function_overloads = self.dispatcher.get_function_overloads(
                self.op_overload, "aten::add", self.diagnostic_context
            )
            self.dispatcher._find_the_perfect_or_nearest_match_onnxfunction(
                self.op_overload,
                "aten::add",
                function_overloads,
                self.op_overload.args,
                self.op_overload.kwargs,
                self.diagnostic_context,
            )

    @common_utils.parametrize(
        "available_opsets, target, expected",
        [
            ((7, 8, 9, 10, 11), 16, 11),
            ((7, 8, 9, 10, 11), 11, 11),
            ((7, 8, 9, 10, 11), 10, 10),
            ((7, 8, 9, 10, 11), 9, 9),
            ((7, 8, 9, 10, 11), 8, 8),
            ((7, 8, 9, 10, 11), 7, 7),
            ((9, 10, 16), 16, 16),
            ((9, 10, 16), 15, 10),
            ((9, 10, 16), 10, 10),
            ((9, 10, 16), 9, 9),
            ((9, 10, 16), 8, 9),
            ((9, 10, 16), 7, 9),
            ((7, 9, 10, 16), 16, 16),
            ((7, 9, 10, 16), 10, 10),
            ((7, 9, 10, 16), 9, 9),
            ((7, 9, 10, 16), 8, 9),
            ((7, 9, 10, 16), 7, 7),
            ([17], 16, None),  # New op added in 17
            ([9], 9, 9),
            ([9], 8, 9),
            ([], 16, None),
            ([], 9, None),
            ([], 8, None),
            # Ops registered at opset 1 found as a fallback when target >= 9
            ([1], 16, 1),
        ],
    )
    def test_dispatch_opset_version_returns_correct_version(
        self, available_opsets: Sequence[int], target: int, expected: int
    ):
        actual = onnxfunction_dispatcher._dispatch_opset_version(
            target, available_opsets
        )
        self.assertEqual(actual, expected)


class TestOpSchemaWrapper(common_utils.TestCase):
    def setUp(self):
        # OnnxFunction with default attributes
        self.onnx_function_add = ops.core.aten_add
        self.op_schema_wrapper_add = onnxfunction_dispatcher._OpSchemaWrapper(
            self.onnx_function_add.op_schema
        )

        # overload type: overloaded dtypes
        self.onnx_function_mul = ops.core.aten_mul
        self.onnx_function_mul_bool = ops.core.aten_mul_bool

        # overload type: optional dtype
        self.onnx_function_new_full = ops.core.aten_new_full
        self.onnx_function_new_full_dtype = ops.core.aten_new_full_dtype

    def test_perfect_match_inputs(self):
        inputs = [torch.randn(3, 4), torch.randn(3, 4)]
        wrong_type_inputs = ["A", "B"]
        wrong_number_of_inputs = [
            torch.randn(3, 4),
            torch.randn(3, 4),
            torch.randn(3, 4),
        ]
        self.assertTrue(
            self.op_schema_wrapper_add.perfect_match_inputs(inputs, {"alpha": 2.0})
        )
        # Even though the attributes has default, to pass perfect match, the attributes must be specified
        # Otherwise, we rely on the matching score mechanism
        self.assertFalse(
            self.op_schema_wrapper_add.perfect_match_inputs(wrong_type_inputs, {})
        )
        self.assertFalse(
            self.op_schema_wrapper_add.perfect_match_inputs(wrong_number_of_inputs, {})
        )
        # wrong kwargs
        self.assertFalse(
            self.op_schema_wrapper_add.perfect_match_inputs(
                inputs, {"wrong_kwargs": 2.0}
            )
        )

    def test_matching_score_system_on_overload_dtypes(self):
        inputs = [torch.randn(3, 4), torch.randn(3, 4)]
        inputs_bool = [
            torch.randint(0, 2, size=(3, 4), dtype=torch.int).bool(),
            torch.randint(0, 2, size=(3, 4), dtype=torch.int).bool(),
        ]

        op_schema_wrapper_mul = onnxfunction_dispatcher._OpSchemaWrapper(
            self.onnx_function_mul.op_schema
        )
        op_schema_wrapper_mul._record_matching_score(inputs, {})
        self.assertEqual(op_schema_wrapper_mul.match_score, 2)

        op_schema_wrapper_mul = onnxfunction_dispatcher._OpSchemaWrapper(
            self.onnx_function_mul.op_schema
        )
        op_schema_wrapper_mul._record_matching_score(inputs_bool, {})
        self.assertEqual(op_schema_wrapper_mul.match_score, 0)

        op_schema_wrapper_mul_bool = onnxfunction_dispatcher._OpSchemaWrapper(
            self.onnx_function_mul_bool.op_schema
        )
        op_schema_wrapper_mul_bool._record_matching_score(inputs, {})
        self.assertEqual(op_schema_wrapper_mul_bool.match_score, 0)

        op_schema_wrapper_mul_bool = onnxfunction_dispatcher._OpSchemaWrapper(
            self.onnx_function_mul_bool.op_schema
        )
        op_schema_wrapper_mul_bool._record_matching_score(inputs_bool, {})
        self.assertEqual(op_schema_wrapper_mul_bool.match_score, 2)

    def test_matching_score_system_on_optional_dtypes(self):
        inputs = [torch.randn(3, 4), torch.tensor(3)]

        op_schema_wrapper_new_full = onnxfunction_dispatcher._OpSchemaWrapper(
            self.onnx_function_new_full.op_schema
        )
        op_schema_wrapper_new_full._record_matching_score(inputs, {})
        self.assertEqual(op_schema_wrapper_new_full.match_score, 2)

        op_schema_wrapper_new_full = onnxfunction_dispatcher._OpSchemaWrapper(
            self.onnx_function_new_full.op_schema
        )
        op_schema_wrapper_new_full._record_matching_score(
            inputs, {"dtype": torch.float}
        )
        # subtract 1 for one mismatch kwargs
        self.assertEqual(op_schema_wrapper_new_full.match_score, 1)

        op_schema_wrapper_new_full_dtype = onnxfunction_dispatcher._OpSchemaWrapper(
            self.onnx_function_new_full_dtype.op_schema
        )
        op_schema_wrapper_new_full_dtype._record_matching_score(inputs, {})
        # subtract 1 for one mismatch kwargs
        self.assertEqual(op_schema_wrapper_new_full_dtype.match_score, 1)

        op_schema_wrapper_new_full_dtype = onnxfunction_dispatcher._OpSchemaWrapper(
            self.onnx_function_new_full_dtype.op_schema
        )
        op_schema_wrapper_new_full_dtype._record_matching_score(
            inputs, {"dtype": torch.float}
        )
        self.assertEqual(op_schema_wrapper_new_full_dtype.match_score, 2)

    def test_find_onnx_data_type(self):
        self.assertEqual(
            onnxfunction_dispatcher._find_onnx_data_type(1),
            {"tensor(int64)", "tensor(int16)", "tensor(int32)"},
        )
        self.assertEqual(
            onnxfunction_dispatcher._find_onnx_data_type(1.0),
            {"tensor(float)", "tensor(double)", "tensor(float16)"},
        )
        self.assertEqual(
            onnxfunction_dispatcher._find_onnx_data_type(torch.tensor([True])),
            {"tensor(bool)"},
        )
        self.assertEqual(
            onnxfunction_dispatcher._find_onnx_data_type(
                torch.tensor([1], dtype=torch.int64)
            ),
            {"tensor(int64)"},
        )
        self.assertEqual(
            onnxfunction_dispatcher._find_onnx_data_type(
                torch.tensor([1], dtype=torch.float16)
            ),
            {"tensor(float16)"},
        )
        self.assertEqual(
            onnxfunction_dispatcher._find_onnx_data_type(
                torch.tensor([1], dtype=torch.int32)
            ),
            {"tensor(int32)"},
        )
        self.assertEqual(
            onnxfunction_dispatcher._find_onnx_data_type(
                [torch.tensor([1], dtype=torch.int32)]
            ),
            {"seq(tensor(int32))"},
        )
        # None is no types allowed
        self.assertEqual(onnxfunction_dispatcher._find_onnx_data_type(None), set())
        self.assertEqual(onnxfunction_dispatcher._find_onnx_data_type([]), set())


if __name__ == "__main__":
    common_utils.run_tests()
