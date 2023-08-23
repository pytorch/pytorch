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
from torch.onnx._internal.diagnostics import infra
from torch.onnx._internal.fx import onnxfunction_dispatcher, registration
from torch.testing._internal import common_utils

# TODO: this can only be global. https://github.com/microsoft/onnxscript/issues/805
TCustomFloat = TypeVar("TCustomFloat", bound=Union[FLOAT16, FLOAT, DOUBLE, BFLOAT16])


class TestRegistration(common_utils.TestCase):
    def setUp(self) -> None:
        self.registry = registration.OnnxRegistry(opset_version=18)
        self.custom_domain = onnxscript.values.Opset(domain="custom", version=1)

    def tearDown(self) -> None:
        internal_name_instance = registration.OpName.from_name_parts(
            namespace="test", op_name="test_op"
        )
        self.registry._registry.pop(internal_name_instance, None)

    def test_register_custom_op_registers_custom_function(self):
        self.assertFalse(self.registry.is_registered_op("test", "test_op", "default"))

        @onnxscript.script(self.custom_domain)
        def custom_add(x, y):
            return op.Add(x, y)

        self.registry.register_custom_op(custom_add, "test", "test_op", "default")
        self.assertTrue(self.registry.is_registered_op("test", "test_op", "default"))

        # Test on get_functions
        function_group = self.registry.get_functions("test", "test_op", "default")
        self.assertIsNotNone(function_group)
        self.assertEqual({func.onnx_function for func in function_group}, {custom_add})  # type: ignore[arg-type]

    def test_custom_onnx_symbolic_joins_existing_function(self):
        self.assertFalse(self.registry.is_registered_op("test", "test_op"))

        @onnxscript.script(self.custom_domain)
        def test_original(x, y):
            return op.Add(x, y)

        # default has to be specified, as we are not using the registration.OpName
        internal_name_instance = registration.OpName.from_name_parts(
            namespace="test", op_name="test_op", overload="default"
        )
        symbolic_fn = registration.SymbolicFunction(
            test_original, op_full_name=internal_name_instance.qualified_name()
        )
        self.registry._register(internal_name_instance, symbolic_fn)
        self.assertTrue(self.registry.is_registered_op("test", "test_op"))

        @onnxscript.script(self.custom_domain)
        def test_custom(x, y):
            return op.Add(x, y)

        self.registry.register_custom_op(test_custom, "test", "test_op")

        function_group = self.registry.get_functions("test", "test_op")
        assert function_group is not None
        # The order does matter (list)
        self.assertEqual(
            [func.onnx_function for func in function_group],
            [test_original, test_custom],
        )


@common_utils.instantiate_parametrized_tests
class TestDispatcher(common_utils.TestCase):
    def setUp(self):
        self.registry = registration.OnnxRegistry(opset_version=18)
        # TODO: remove this once we have a better way to do this
        logger = logging.getLogger("TestDispatcher")
        self.diagnostic_context = infra.DiagnosticContext(
            "torch.onnx.dynamo_export", torch.__version__, logger=logger
        )
        self.dispatcher = onnxfunction_dispatcher.OnnxFunctionDispatcher(
            self.registry, self.diagnostic_context
        )

    @common_utils.parametrize(
        "node, expected_name",
        [
            common_utils.subtest(
                (
                    torch.fx.Node(
                        graph=torch.fx.Graph(),
                        name="aten::add.Tensor",
                        op="call_function",
                        target=torch.ops.aten.add.Tensor,  # type: ignore[attr-defined]
                        args=(torch.tensor(3), torch.tensor(4)),
                        kwargs={},
                    ),
                    ("aten", "add", "Tensor"),
                ),
                name="get_Opoverload_name",
            ),
            common_utils.subtest(
                (
                    torch.fx.Node(
                        graph=torch.fx.Graph(),
                        name="aten::sym_size",
                        op="call_function",
                        target=torch.ops.aten.sym_size,
                        args=(),
                        kwargs={},
                    ),
                    ("aten", "sym_size", None),
                ),
                name="get_Opoverloadpacket_name",
            ),
            common_utils.subtest(
                (
                    torch.fx.Node(
                        graph=torch.fx.Graph(),
                        name="builtin_add",
                        op="call_function",
                        target=operator.add,
                        args=(1, 2),
                        kwargs={},
                    ),
                    ("aten", "add", None),
                ),
                name="get_builtin_op_name",
            ),
        ],
    )
    def test_get_aten_name_on_supported_fx_node(
        self, node: torch.fx.Node, expected_name: str
    ):
        expected_name_class = registration.OpName.from_name_parts(*expected_name)
        self.assertEqual(
            self.dispatcher._get_aten_name(node, self.diagnostic_context),
            expected_name_class,
        )

    @common_utils.parametrize(
        "node",
        [
            common_utils.subtest(
                torch.fx.Node(
                    graph=torch.fx.Graph(),
                    name="aten::add",
                    op="call_function",
                    target=torch.ops.aten.add,
                    args=(),
                    kwargs={},
                ),
                name="unsupported_Opoverloadpacket_name",
            ),
            common_utils.subtest(
                torch.fx.Node(
                    graph=torch.fx.Graph(),
                    name="builtin_add",
                    op="call_function",
                    target=operator.add,
                    args=("A", "B"),
                    kwargs={},
                ),
                name="unsupported_input_dtypes_for_builtin_op",
            ),
            common_utils.subtest(
                torch.fx.Node(
                    graph=torch.fx.Graph(),
                    name="aten::made_up_node",
                    op="call_function",
                    target=lambda: None,
                    args=(),
                    kwargs={},
                ),
                name="unsupported_target_function",
            ),
        ],
    )
    def test_get_aten_name_on_unsupported_fx_node(self, node: torch.fx.Node):
        with self.assertRaises(RuntimeError):
            self.dispatcher._get_aten_name(node, self.diagnostic_context)

    def test_get_function_overloads_gives_overload_fall_back_default(self):
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
            target=torch.ops.aten.add.Tensor,  # type: ignore[attr-defined]
            args=(),
            kwargs={},
        )

        self.assertEqual(
            self.dispatcher.get_function_overloads(
                node_overload, self.diagnostic_context
            ),
            self.dispatcher.get_function_overloads(
                node_overloadpacket,
                self.diagnostic_context,
            ),
        )

        # Non-registered op
        internal_opname_class_unsupported = registration.OpName.from_name_parts(
            namespace="aten", op_name="made_up_node", overload=None
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
                unsupported_op_node,
                self.diagnostic_context,
            )

    @common_utils.parametrize(
        "node",
        [
            common_utils.subtest(
                torch.fx.Node(
                    graph=torch.fx.Graph(),
                    name="aten::add.Tensor",
                    op="call_function",
                    target=torch.ops.aten.add.Tensor,  # type: ignore[attr-defined]
                    args=(torch.tensor(3), torch.tensor(4)),
                    kwargs={},
                ),
                name="nearest_match",
            ),
            common_utils.subtest(
                torch.fx.Node(
                    graph=torch.fx.Graph(),
                    name="aten::add.Tensor",
                    op="call_function",
                    target=torch.ops.aten.add.Tensor,  # type: ignore[attr-defined]
                    args=(torch.tensor(3), torch.tensor(4)),
                    kwargs={"alpha": 1},
                ),
                name="perfect_match_with_kwargs",
            ),
        ],
    )
    def test_find_the_perfect_or_nearest_match_onnxfunction_gives_custom_ops_precedence(
        self, node
    ):
        custom_domain = onnxscript.values.Opset(domain="custom", version=1)

        @onnxscript.script(custom_domain)
        def test_custom_op(x: TCustomFloat, y: TCustomFloat) -> TCustomFloat:
            return op.Add(x, y)

        @onnxscript.script(custom_domain)
        def test_default_op(x: TCustomFloat, y: TCustomFloat) -> TCustomFloat:
            return op.Add(x, y)

        op_full_name = "test::test_op"

        custom_overloads = [
            registration.SymbolicFunction(
                test_custom_op, op_full_name=op_full_name, is_custom=True
            )
        ]
        function_overloads = [
            registration.SymbolicFunction(test_default_op, op_full_name=op_full_name)
        ] + custom_overloads

        symbolic_fn = self.dispatcher._find_the_perfect_or_nearest_match_onnxfunction(
            node,
            function_overloads,
            node.args,
            node.kwargs,
            self.diagnostic_context,
        )
        self.assertEqual(symbolic_fn, test_custom_op)


@common_utils.instantiate_parametrized_tests
class TestOpSchemaWrapper(common_utils.TestCase):
    def setUp(self):
        # overload type: optional dtype
        self.onnx_function_new_full = ops.core.aten_new_full
        self.onnx_function_new_full_dtype = ops.core.aten_new_full_dtype

    @common_utils.parametrize(
        "inputs, attributes, assertion",
        [
            common_utils.subtest(
                ([torch.randn(3, 4), torch.randn(3, 4)], {"alpha": 2.0}, True),
                name="perfect_match_with_kwargs",
            ),
            common_utils.subtest(
                (["A", "B"], {}, False),
                name="non_perfect_match_due_to_non_tensor_inputs",
            ),
            common_utils.subtest(
                ([torch.randn(3, 4), torch.randn(3, 4)], {"wrong_kwargs": 2.0}, False),
                name="non_perfect_match_due_to_wrong_kwargs",
            ),
        ],
    )
    def test_perfect_match_inputs(self, inputs, attributes, assertion):
        # OnnxFunction with default attributes
        op_schema_wrapper_add = onnxfunction_dispatcher._OnnxSchemaChecker(
            ops.core.aten_add
        )
        self.assertEqual(
            op_schema_wrapper_add.perfect_match_inputs(inputs, attributes), assertion
        )

    @common_utils.parametrize(
        "inputs, kwargs, op, score",
        [
            common_utils.subtest(
                ([torch.randn(3, 4), torch.randn(3, 4)], {}, ops.core.aten_mul, 2),
                name="match_2_inputs",
            ),
            common_utils.subtest(
                (
                    [
                        torch.randint(0, 2, size=(3, 4), dtype=torch.int).bool(),
                        torch.randint(0, 2, size=(3, 4), dtype=torch.int).bool(),
                    ],
                    {},
                    ops.core.aten_mul,
                    0,
                ),
                name="match_0_inputs",
            ),
            common_utils.subtest(
                ([torch.randn(3, 4), torch.randn(3, 4)], {}, ops.core.aten_mul_bool, 0),
                name="match_0_inputs_bool",
            ),
            common_utils.subtest(
                (
                    [
                        torch.randint(0, 2, size=(3, 4), dtype=torch.int).bool(),
                        torch.randint(0, 2, size=(3, 4), dtype=torch.int).bool(),
                    ],
                    {},
                    ops.core.aten_mul_bool,
                    2,
                ),
                name="match_2_inputs_bool",
            ),
        ],
    )
    def test_matching_score_system_on_overload_dtypes(self, inputs, kwargs, op, score):
        op_schema_wrapper = onnxfunction_dispatcher._OnnxSchemaChecker(op)
        op_schema_wrapper._record_matching_score(inputs, kwargs)
        self.assertEqual(op_schema_wrapper.match_score, score)

    @common_utils.parametrize(
        "inputs, kwargs, op, score",
        [
            common_utils.subtest(
                ([torch.randn(3, 4), torch.tensor(3)], {}, ops.core.aten_new_full, 2),
                name="match_2_inputs",
            ),
            common_utils.subtest(
                (
                    [torch.randn(3, 4), torch.tensor(3)],
                    {"dtype": torch.float},
                    ops.core.aten_new_full,
                    1,
                ),
                name="match_2_inputs_and_mismatch_1_kwarg",
            ),
            common_utils.subtest(
                (
                    [torch.randn(3, 4), torch.tensor(3)],
                    {},
                    ops.core.aten_new_full_dtype,
                    1,
                ),
                name="match_2_input_and_mismatch_1_kwargs_optional",
            ),
            common_utils.subtest(
                (
                    [torch.randn(3, 4), torch.tensor(3)],
                    {"dtype": torch.float},
                    ops.core.aten_new_full_dtype,
                    2,
                ),
                name="match_2_input_and_match_1_kwargs_optional",
            ),
        ],
    )
    def test_matching_score_system_on_optional_dtypes(self, inputs, kwargs, op, score):
        op_schema_wrapper = onnxfunction_dispatcher._OnnxSchemaChecker(op)
        op_schema_wrapper._record_matching_score(inputs, kwargs)
        self.assertEqual(op_schema_wrapper.match_score, score)

    @common_utils.parametrize(
        "value, expected_onnx_str_dtype",
        [
            common_utils.subtest(
                (1, {"tensor(int64)", "tensor(int16)", "tensor(int32)"}),
                name="all_ints",
            ),
            common_utils.subtest(
                (1.0, {"tensor(float)", "tensor(double)", "tensor(float16)"}),
                name="all_floats",
            ),
            common_utils.subtest(
                (torch.tensor([True]), {"tensor(bool)"}),
                name="bool",
            ),
            common_utils.subtest(
                (torch.tensor([1], dtype=torch.int64), {"tensor(int64)"}),
                name="int64",
            ),
            common_utils.subtest(
                (torch.tensor([1], dtype=torch.int32), {"tensor(int32)"}),
                name="int32",
            ),
            common_utils.subtest(
                (torch.tensor([1], dtype=torch.int16), {"tensor(int16)"}),
                name="int16",
            ),
            common_utils.subtest(
                (torch.tensor([1], dtype=torch.float), {"tensor(float)"}),
                name="float",
            ),
            common_utils.subtest(
                (torch.tensor([1], dtype=torch.float16), {"tensor(float16)"}),
                name="float16",
            ),
            common_utils.subtest(
                (torch.tensor([1], dtype=torch.double), {"tensor(double)"}),
                name="double",
            ),
            common_utils.subtest((None, set()), name="None"),  # None allows no dtype
            common_utils.subtest(
                ([], set()), name="empaty_list"
            ),  # Empty list allows no dtype
        ],
    )
    def test_find_onnx_data_type(self, value, expected_onnx_str_dtype):
        self.assertEqual(
            onnxfunction_dispatcher._find_onnx_data_type(value), expected_onnx_str_dtype
        )


if __name__ == "__main__":
    common_utils.run_tests()
