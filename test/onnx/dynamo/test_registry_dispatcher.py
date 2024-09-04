# Owner(s): ["module: onnx"]
"""Unit tests for the internal registration wrapper module."""

from __future__ import annotations

import operator
from typing import TypeVar, Union

import onnxscript  # type: ignore[import]
from onnxscript import BFLOAT16, DOUBLE, FLOAT, FLOAT16  # type: ignore[import]
from onnxscript.onnx_opset import opset15 as op  # type: ignore[import]

import torch
import torch.fx
from torch.onnx._internal.fx import diagnostics, onnxfunction_dispatcher, registration
from torch.testing._internal import common_utils


# TODO: this can only be global. https://github.com/microsoft/onnxscript/issues/805
TCustomFloat = TypeVar("TCustomFloat", bound=Union[FLOAT16, FLOAT, DOUBLE, BFLOAT16])


class TestRegistration(common_utils.TestCase):
    def setUp(self) -> None:
        self.registry = torch.onnx.OnnxRegistry()
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

        self.registry.register_op(custom_add, "test", "test_op", "default")
        self.assertTrue(self.registry.is_registered_op("test", "test_op", "default"))

        # Test on get_ops
        function_group = self.registry.get_op_functions("test", "test_op", "default")
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
        symbolic_fn = registration.ONNXFunction(
            test_original, op_full_name=internal_name_instance.qualified_name()
        )
        self.registry._register(internal_name_instance, symbolic_fn)
        self.assertTrue(self.registry.is_registered_op("test", "test_op"))

        @onnxscript.script(self.custom_domain)
        def test_custom(x, y):
            return op.Add(x, y)

        self.registry.register_op(test_custom, "test", "test_op")

        function_group = self.registry.get_op_functions("test", "test_op")
        assert function_group is not None
        # The order does matter (list)
        self.assertEqual(
            [func.onnx_function for func in function_group],
            [test_original, test_custom],
        )


@common_utils.instantiate_parametrized_tests
class TestDispatcher(common_utils.TestCase):
    def setUp(self):
        self.registry = torch.onnx.OnnxRegistry()
        self.diagnostic_context = diagnostics.DiagnosticContext(
            "torch.onnx.dynamo_export", torch.__version__
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
                    ("_operator", "add", None),
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
                    args=(torch.tensor(3.0), torch.tensor(4.0)),
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
                    args=(torch.tensor(3.0), torch.tensor(4.0)),
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
        def test_custom_op(
            x: TCustomFloat, y: TCustomFloat, alpha: int = 1
        ) -> TCustomFloat:
            return op.Add(x, y)

        @onnxscript.script(custom_domain)
        def test_default_op(
            x: TCustomFloat, y: TCustomFloat, alpha: int = 1
        ) -> TCustomFloat:
            return op.Add(x, y)

        op_full_name = "test::test_op"

        custom_overloads = [
            registration.ONNXFunction(
                test_custom_op, op_full_name=op_full_name, is_custom=True
            )
        ]
        function_overloads = [
            registration.ONNXFunction(test_default_op, op_full_name=op_full_name)
        ] + custom_overloads

        symbolic_fn = self.dispatcher._find_the_perfect_or_nearest_match_onnxfunction(
            node,
            function_overloads,
            node.args,
            node.kwargs,
            self.diagnostic_context,
        )
        self.assertEqual(symbolic_fn, test_custom_op)

    @common_utils.parametrize(
        "node",
        [
            common_utils.subtest(
                torch.fx.Node(
                    graph=torch.fx.Graph(),
                    name="aten::add.Tensor",
                    op="call_function",
                    target=torch.ops.aten.add.Tensor,  # type: ignore[attr-defined]
                    args=(torch.tensor(3.0), torch.tensor(4.0)),
                    kwargs={"attr": None},
                ),
                name="perfect_match_with_ignoring_none_attribute",
            ),
            common_utils.subtest(
                torch.fx.Node(
                    graph=torch.fx.Graph(),
                    name="aten::add.Tensor",
                    op="call_function",
                    target=torch.ops.aten.add.Tensor,  # type: ignore[attr-defined]
                    args=(torch.tensor(3.0), torch.tensor(4.0)),
                    kwargs={"unrelated": None},
                ),
                name="perfect_match_with_ignoring_unrelated_none_attribute",
            ),
        ],
    )
    def test_find_the_perfect_or_nearest_match_onnxfunction_ignores_attribute_with_none(
        self, node
    ):
        custom_domain = onnxscript.values.Opset(domain="custom", version=1)

        @onnxscript.script(custom_domain)
        def test_op_attribute(
            x: TCustomFloat, y: TCustomFloat, attr: int
        ) -> TCustomFloat:
            return op.Add(x, y)

        @onnxscript.script(custom_domain)
        def test_op(x: TCustomFloat, y: TCustomFloat) -> TCustomFloat:
            return op.Add(x, y)

        op_full_name = "test::test_op"

        function_overloads = [
            registration.ONNXFunction(test_op_attribute, op_full_name=op_full_name),
            registration.ONNXFunction(test_op, op_full_name=op_full_name),
        ]

        symbolic_fn = self.dispatcher._find_the_perfect_or_nearest_match_onnxfunction(
            node,
            function_overloads,
            node.args,
            node.kwargs,
            self.diagnostic_context,
        )
        self.assertEqual(symbolic_fn, test_op)

    @common_utils.parametrize(
        "node",
        [
            common_utils.subtest(
                torch.fx.Node(
                    graph=torch.fx.Graph(),
                    name="aten::add.Tensor",
                    op="call_function",
                    target=torch.ops.aten.add.Tensor,  # type: ignore[attr-defined]
                    args=(torch.tensor(3.0), torch.tensor(4.0)),
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
                    args=(torch.tensor(3.0), torch.tensor(4.0)),
                    kwargs={"alpha": 1},
                ),
                name="perfect_match_with_kwargs",
            ),
        ],
    )
    def test_find_the_perfect_or_nearest_match_onnxfunction_gives_tie_breaks_to_registered_order(
        self, node
    ):
        custom_domain = onnxscript.values.Opset(domain="custom", version=1)

        @onnxscript.script(custom_domain)
        def test_second_custom_op(
            x: TCustomFloat, y: TCustomFloat, alpha: int = 1
        ) -> TCustomFloat:
            return op.Add(x, y)

        @onnxscript.script(custom_domain)
        def test_third_custom_op(
            x: TCustomFloat, y: TCustomFloat, alpha: int = 1
        ) -> TCustomFloat:
            return op.Add(x, y)

        @onnxscript.script(custom_domain)
        def test_first_custom_op(
            x: TCustomFloat, y: TCustomFloat, alpha: int = 1
        ) -> TCustomFloat:
            return op.Add(x, y)

        op_full_name = "aten::add"

        function_overloads = [
            registration.ONNXFunction(
                test_first_custom_op, op_full_name=op_full_name, is_custom=True
            ),
            registration.ONNXFunction(
                test_second_custom_op, op_full_name=op_full_name, is_custom=True
            ),
            registration.ONNXFunction(
                test_third_custom_op, op_full_name=op_full_name, is_custom=True
            ),
        ]

        symbolic_fn = self.dispatcher._find_the_perfect_or_nearest_match_onnxfunction(
            node,
            function_overloads,
            node.args,
            node.kwargs,
            self.diagnostic_context,
        )
        self.assertEqual(symbolic_fn, test_third_custom_op)


if __name__ == "__main__":
    common_utils.run_tests()
