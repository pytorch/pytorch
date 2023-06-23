# Owner(s): ["module: onnx"]
"""Unit tests for the internal registration wrapper module."""
from __future__ import annotations

import logging
import operator
from typing import TypeVar, Union

import onnxscript  # type: ignore[import]

import torch
import torch.fx
from onnxscript import BFLOAT16, DOUBLE, FLOAT, FLOAT16  # type: ignore[import]
from onnxscript.function_libs.torch_lib import ops  # type: ignore[import]
from onnxscript.onnx_opset import opset15 as op  # type: ignore[import]
from parameterized import parameterized  # type: ignore[import]
from torch.onnx._internal.diagnostics import infra
from torch.onnx._internal.fx import onnxfunction_dispatcher, registration
from torch.testing._internal import common_utils

# TODO: this can only be global. https://github.com/microsoft/onnxscript/issues/805
TCustomFloat = TypeVar("TCustomFloat", bound=Union[FLOAT16, FLOAT, DOUBLE, BFLOAT16])


class TestRegistration(common_utils.TestCase):
    def setUp(self) -> None:
        self.registry = registration.OnnxRegistry()
        self.custom_domain = onnxscript.values.Opset(domain="custom", version=1)

    def tearDown(self) -> None:
        self.registry._registry.pop("test::test_op", None)

    def test_onnx_custom_symbolic_registers_function(self):
        self.assertFalse(self.registry.is_registered_op("test::test_op"))

        @onnxscript.script(self.custom_domain)
        def test(x, y):
            return op.Add(x, y)

        self.registry.register_custom_op("test::test_op", test)
        self.assertTrue(self.registry.is_registered_op("test::test_op"))

        function_group = self.registry.get_functions("test::test_op")
        assert function_group is not None
        self.assertEqual({func.onnx_function for func in function_group}, {test})

        function_group = self.registry._get_custom_functions("test::test_op")
        assert function_group is not None
        self.assertEqual({func.onnx_function for func in function_group}, {test})

    def test_custom_onnx_symbolic_joins_existing_function(self):
        self.assertFalse(self.registry.is_registered_op("test::test_op"))

        @onnxscript.script(self.custom_domain)
        def test_original(x, y):
            return op.Add(x, y)

        symbolic_fn = registration.SymbolicFunction(
            test_original, op_name="test::test_op"
        )
        self.registry._register(symbolic_fn)
        self.assertTrue(self.registry.is_registered_op("test::test_op"))

        @onnxscript.script(self.custom_domain)
        def test_custom(x, y):
            return op.Add(x, y)

        self.registry.register_custom_op("test::test_op", test_custom)

        function_group = self.registry.get_functions("test::test_op")
        assert function_group is not None
        self.assertEqual(
            {func.onnx_function for func in function_group},
            {test_custom, test_original},
        )

        function_group = self.registry._get_custom_functions("test::test_op")
        assert function_group is not None
        self.assertEqual({func.onnx_function for func in function_group}, {test_custom})


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

    @parameterized.expand(
        [
            (
                torch.fx.Node(
                    graph=torch.fx.Graph(),
                    name="aten::add.Tensor",
                    op="call_function",
                    target=torch.ops.aten.add.Tensor,  # type: ignore[attr-defined]
                    args=(torch.tensor(3), torch.tensor(4)),
                    kwargs={},
                ),
                "aten::add.Tensor",
            ),
            (
                torch.fx.Node(
                    graph=torch.fx.Graph(),
                    name="aten::sym_size",
                    op="call_function",
                    target=torch.ops.aten.sym_size,
                    args=(),
                    kwargs={},
                ),
                "aten::sym_size",
            ),
            (
                torch.fx.Node(
                    graph=torch.fx.Graph(),
                    name="builtin_add",
                    op="call_function",
                    target=operator.add,
                    args=(1, 2),
                    kwargs={},
                ),
                "aten::add",
            ),
        ]
    )
    def test_get_aten_name_on_supported_fx_node(self, node, expected_name):
        self.assertEqual(
            self.dispatcher.get_aten_name(node, self.diagnostic_context),
            expected_name,
        )

    @parameterized.expand(
        [
            (
                torch.fx.Node(
                    graph=torch.fx.Graph(),
                    name="aten::add",
                    op="call_function",
                    target=torch.ops.aten.add,
                    args=(),
                    kwargs={},
                ),
            ),
            (
                torch.fx.Node(
                    graph=torch.fx.Graph(),
                    name="builtin_add",
                    op="call_function",
                    target=operator.add,
                    args=("A", "B"),
                    kwargs={},
                ),
            ),
            (
                torch.fx.Node(
                    graph=torch.fx.Graph(),
                    name="aten::made_up_node",
                    op="call_function",
                    target=lambda: None,
                    args=(),
                    kwargs={},
                ),
            ),
        ]
    )
    def test_get_aten_name_on_unsupported_fx_node(self, node):
        with self.assertRaises(RuntimeError):
            self.dispatcher.get_aten_name(node, self.diagnostic_context)

    def test_get_function_overloads(self):
        # Test fall back to default op name
        node_overload = torch.fx.Node(
            graph=torch.fx.Graph(),
            name="aten::add.Tensor",
            op="call_function",
            target=torch.ops.aten.add.Tensor,  # type: ignore[attr-defined]
            args=(torch.tensor(3), torch.tensor(4)),
            kwargs={},
        )
        node_overloadpacket = torch.fx.Node(
            graph=torch.fx.Graph(),
            name="aten::add",
            op="call_function",
            target=torch.ops.aten.add,
            args=(),
            kwargs={},
        )
        self.assertEqual(
            self.dispatcher.get_function_overloads(
                node_overload, "aten::add.Tensor", self.diagnostic_context
            ),
            self.dispatcher.get_function_overloads(
                node_overloadpacket, "aten::add", self.diagnostic_context
            ),
        )
        unsupported_op_node = torch.fx.Node(
            graph=torch.fx.Graph(),
            name="aten::made_up_node",
            op="call_function",
            target=lambda: None,
            args=(),
            kwargs={},
        )
        with self.assertRaises(RuntimeError):
            self.dispatcher.get_function_overloads(
                unsupported_op_node, "aten::made_up_node", self.diagnostic_context
            )

    def test_warnings_in_find_the_perfect_or_nearest_match_onnxfunction(self):
        op_overload = torch.fx.Node(
            graph=torch.fx.Graph(),
            name="aten::add.Tensor",
            op="call_function",
            target=torch.ops.aten.add.Tensor,  # type: ignore[attr-defined]
            args=(torch.tensor(3), torch.tensor(4)),
            kwargs={},
        )
        with self.assertWarnsOnceRegex(
            UserWarning,
            "A perfect matched Opchema is not found in torchlib for aten::add",
        ):
            (
                function_overloads,
                custom_overloads,
            ) = self.dispatcher.get_function_overloads(
                op_overload, "aten::add", self.diagnostic_context
            )
            self.dispatcher._find_the_perfect_or_nearest_match_onnxfunction(
                op_overload,
                "aten::add",
                function_overloads,
                custom_overloads,  # custom function overloads
                op_overload.args,
                op_overload.kwargs,
                self.diagnostic_context,
            )

    @parameterized.expand(
        [
            (
                torch.fx.Node(
                    graph=torch.fx.Graph(),
                    name="aten::add.Tensor",
                    op="call_function",
                    target=torch.ops.aten.add.Tensor,  # type: ignore[attr-defined]
                    args=(torch.tensor(3), torch.tensor(4)),
                    kwargs={},
                ),
            ),
            (
                torch.fx.Node(
                    graph=torch.fx.Graph(),
                    name="aten::add.Tensor",
                    op="call_function",
                    target=torch.ops.aten.add.Tensor,  # type: ignore[attr-defined]
                    args=(torch.tensor(3), torch.tensor(4)),
                    kwargs={"alpha": 1},
                ),
            ),
        ]
    )
    def test_respect_custom_ops_in_find_the_perfect_or_nearest_match_onnxfunction(
        self, node
    ):
        custom_domain = onnxscript.values.Opset(domain="custom", version=1)

        @onnxscript.script(custom_domain)
        def test_custom_op(x: TCustomFloat, y: TCustomFloat) -> TCustomFloat:
            return op.Add(x, y)

        @onnxscript.script(custom_domain)
        def test_default_op(x: TCustomFloat, y: TCustomFloat) -> TCustomFloat:
            return op.Add(x, y)

        custom_overloads = set(
            {
                registration.SymbolicFunction(
                    test_custom_op, op_name="test::test_op", is_custom=True
                )
            }
        )
        function_overloads = (
            set(
                {
                    registration.SymbolicFunction(
                        test_default_op, op_name="test::test_op"
                    )
                }
            )
            | custom_overloads
        )

        symbolic_fn = self.dispatcher._find_the_perfect_or_nearest_match_onnxfunction(
            node,
            "aten::add",
            function_overloads,
            custom_overloads,  # custom function overloads
            node.args,
            node.kwargs,
            self.diagnostic_context,
        )
        self.assertEqual(symbolic_fn, test_custom_op)


class TestOpSchemaWrapper(common_utils.TestCase):
    def setUp(self):
        # overload type: optional dtype
        self.onnx_function_new_full = ops.core.aten_new_full
        self.onnx_function_new_full_dtype = ops.core.aten_new_full_dtype

    @parameterized.expand(
        [
            ([torch.randn(3, 4), torch.randn(3, 4)], {"alpha": 2.0}, True),
            (["A", "B"], {}, False),
            ([torch.randn(3, 4), torch.randn(3, 4), torch.randn(3, 4)], {}, False),
            ([torch.randn(3, 4), torch.randn(3, 4)], {"wrong_kwargs": 2.0}, False),
        ]
    )
    def test_perfect_match_inputs(self, inputs, attributes, assertion):
        # OnnxFunction with default attributes
        op_schema_wrapper_add = onnxfunction_dispatcher._OpSchemaWrapper(
            ops.core.aten_add.op_schema
        )

        if assertion:
            self.assertTrue(
                op_schema_wrapper_add.perfect_match_inputs(inputs, attributes)
            )
        else:
            self.assertFalse(
                op_schema_wrapper_add.perfect_match_inputs(inputs, attributes)
            )

    @parameterized.expand(
        [
            ([torch.randn(3, 4), torch.randn(3, 4)], {}, ops.core.aten_mul, 2),
            (
                [
                    torch.randint(0, 2, size=(3, 4), dtype=torch.int).bool(),
                    torch.randint(0, 2, size=(3, 4), dtype=torch.int).bool(),
                ],
                {},
                ops.core.aten_mul,
                0,
            ),
            ([torch.randn(3, 4), torch.randn(3, 4)], {}, ops.core.aten_mul_bool, 0),
            (
                [
                    torch.randint(0, 2, size=(3, 4), dtype=torch.int).bool(),
                    torch.randint(0, 2, size=(3, 4), dtype=torch.int).bool(),
                ],
                {},
                ops.core.aten_mul_bool,
                2,
            ),
        ]
    )
    def test_matching_score_system_on_overload_dtypes(self, inputs, kwargs, op, score):
        op_schema_wrapper = onnxfunction_dispatcher._OpSchemaWrapper(op.op_schema)
        op_schema_wrapper._record_matching_score(inputs, kwargs)
        self.assertEqual(op_schema_wrapper.match_score, score)

    @parameterized.expand(
        [
            ([torch.randn(3, 4), torch.tensor(3)], {}, ops.core.aten_new_full, 2),
            (
                [torch.randn(3, 4), torch.tensor(3)],
                {"dtype": torch.float},
                ops.core.aten_new_full,
                1,
            ),
            ([torch.randn(3, 4), torch.tensor(3)], {}, ops.core.aten_new_full_dtype, 1),
            (
                [torch.randn(3, 4), torch.tensor(3)],
                {"dtype": torch.float},
                ops.core.aten_new_full_dtype,
                2,
            ),
        ]
    )
    def test_matching_score_system_on_optional_dtypes(self, inputs, kwargs, op, score):
        op_schema_wrapper = onnxfunction_dispatcher._OpSchemaWrapper(op.op_schema)
        op_schema_wrapper._record_matching_score(inputs, kwargs)
        self.assertEqual(op_schema_wrapper.match_score, score)

    @parameterized.expand(
        [
            (1, {"tensor(int64)", "tensor(int16)", "tensor(int32)"}),
            (1.0, {"tensor(float)", "tensor(double)", "tensor(float16)"}),
            (torch.tensor([True]), {"tensor(bool)"}),
            (torch.tensor([1], dtype=torch.int64), {"tensor(int64)"}),
            (torch.tensor([1], dtype=torch.int32), {"tensor(int32)"}),
            (torch.tensor([1], dtype=torch.int16), {"tensor(int16)"}),
            (torch.tensor([1], dtype=torch.float), {"tensor(float)"}),
            (torch.tensor([1], dtype=torch.float16), {"tensor(float16)"}),
            (torch.tensor([1], dtype=torch.double), {"tensor(double)"}),
            (None, set()),  # None is not type allowed
            ([], set()),  # Empty list is not type allowed
        ]
    )
    def test_find_onnx_data_type(self, value, expected_onnx_str_dtype):
        self.assertEqual(
            onnxfunction_dispatcher._find_onnx_data_type(value), expected_onnx_str_dtype
        )


if __name__ == "__main__":
    common_utils.run_tests()
