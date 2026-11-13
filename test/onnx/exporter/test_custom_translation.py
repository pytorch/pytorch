# Owner(s): ["module: onnx"]
"""Global native registrations and their export-time isolation."""

from __future__ import annotations

import operator
import subprocess
import sys
import threading
from unittest import mock

import onnx

import onnx_ir as ir
from onnx.reference import ReferenceEvaluator
from onnxscript import FLOAT, opset18 as op, script

import torch
from torch.onnx._internal.exporter import (
    _core,
    _custom_translation,
    _dispatching,
    _registration,
)
from torch.testing._internal import common_utils
from torch.testing._internal.common_utils import HardwareClassification


def native_add(op, self: ir.Value, other: ir.Value, alpha: float = 1) -> ir.Value:
    if alpha != 1:
        other = op.Mul(other, alpha)
    return op.Add(self, other)


def native_sub(op, self: ir.Value, other: ir.Value, alpha: float = 1) -> ir.Value:
    return op.Sub(self, other)


class AddModel(torch.nn.Module):
    def forward(self, x, y):
        return torch.add(x, y, alpha=2).sin()


@common_utils.instantiate_parametrized_tests
class TestCustomTranslation(common_utils.TestCase):
    hw_classification = HardwareClassification.GENERIC

    def setUp(self):
        super().setUp()
        with _custom_translation._LOCK:
            saved = _custom_translation._TRANSLATIONS.copy()
            _custom_translation._TRANSLATIONS.clear()

        def restore():
            with _custom_translation._LOCK:
                _custom_translation._TRANSLATIONS.clear()
                _custom_translation._TRANSLATIONS.update(saved)

        self.addCleanup(restore)
        self.target = torch.ops.aten.add.Tensor

    def test_decorator_and_direct_registration_return_original(self):
        self.assertIs(
            torch.onnx.register_custom_translation(self.target)(native_add), native_add
        )
        self.assertIs(
            torch.onnx.register_custom_translation(operator.add, native_add), native_add
        )
        self.assertEqual(len(_custom_translation.snapshot()), 2)

    def test_duplicate_replace_and_exact_unregister(self):
        torch.onnx.register_custom_translation(self.target, native_add)
        torch.onnx.register_custom_translation(self.target, native_add, is_complex=True)
        torch.onnx.register_custom_translation(
            self.target, native_add, opset_version=19
        )
        with self.assertRaisesRegex(ValueError, "replace=True"):
            torch.onnx.register_custom_translation(self.target, native_sub)
        torch.onnx.register_custom_translation(self.target, native_sub, replace=True)
        entries = _custom_translation.snapshot()
        self.assertIs(entries[0].function, native_sub)
        self.assertIs(entries[1].function, native_add)
        torch.onnx.unregister_custom_translation(self.target)
        self.assertEqual(len(_custom_translation.snapshot()), 2)
        with self.assertRaisesRegex(KeyError, "metadata from registration"):
            torch.onnx.unregister_custom_translation(self.target)

    @common_utils.parametrize(
        "target,kwargs,error",
        [
            ("aten::add", {}, TypeError),
            (torch.ops.aten.add, {}, TypeError),
            ([], {}, TypeError),
            (operator.add, {"is_complex": 1}, TypeError),
            (operator.add, {"opset_version": 0}, ValueError),
            (operator.add, {"opset_version": True}, ValueError),
            (operator.add, {"opset_version": 18.5}, ValueError),
            (operator.add, {"replace": 1}, TypeError),
            (operator.add, {"signature": "FLOAT"}, TypeError),
        ],
    )
    def test_invalid_metadata(self, target, kwargs, error):
        with self.assertRaises(error):
            torch.onnx.register_custom_translation(target, native_add, **kwargs)
        self.assertEqual(_custom_translation.snapshot(), ())

    def test_invalid_function(self):
        with self.assertRaisesRegex(TypeError, "must be callable"):
            torch.onnx.register_custom_translation(self.target, 42)

    def test_builder_scope_error(self):
        with self.assertRaisesRegex(RuntimeError, "No ONNX IR builder is active"):
            torch.onnx.get_translation_builder()

    def test_registration_does_not_load_onnxscript(self):
        script = """
import importlib.util
import sys
import torch
before = set(sys.modules)
spec = importlib.util.spec_from_file_location('native_registration', sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
def translation(x):
    return x
module.register_custom_translation(abs, translation)
module.unregister_custom_translation(abs)
loaded = set(sys.modules) - before
if any(n == 'onnxscript' or n.startswith('onnxscript.') or 'torch_lib' in n for n in loaded):
    raise AssertionError(loaded)
"""
        subprocess.run(
            [sys.executable, "-c", script, _custom_translation.__file__],
            check=True,
            cwd=sys.prefix,
        )

    def test_versions_complex_and_snapshot(self):
        torch.onnx.register_custom_translation(
            self.target, native_add, opset_version=17
        )
        torch.onnx.register_custom_translation(
            self.target, native_sub, opset_version=19
        )
        torch.onnx.register_custom_translation(
            self.target, native_sub, is_complex=True, opset_version=18
        )
        snapshot = _custom_translation.snapshot()
        torch.onnx.unregister_custom_translation(self.target, opset_version=17)
        registry = _registration.ONNXRegistry.from_torchlib(
            18, custom_translations=snapshot
        )
        real = [m for m in registry.get_decomps(self.target) if not m.is_complex]
        complex_ = [m for m in registry.get_decomps(self.target) if m.is_complex]
        self.assertIs(real[0].onnx_function.function, native_add)
        self.assertIs(complex_[0].onnx_function.function, native_sub)
        registry19 = _registration.ONNXRegistry.from_torchlib(
            19, custom_translations=snapshot
        )
        self.assertIs(
            registry19.get_decomps(self.target)[0].onnx_function.function, native_sub
        )
        newer = _registration.ONNXRegistry.from_torchlib(18)
        self.assertFalse(
            any(
                m.is_custom and not m.is_complex for m in newer.get_decomps(self.target)
            )
        )

    def test_cleanup_keeps_custom_and_independent_complex_versions(self):
        registry = _registration.ONNXRegistry()
        registry._opset_version = 20
        for version, custom, complex_ in [
            (17, True, False),
            (20, False, False),
            (18, False, True),
        ]:
            registry._register(
                self.target,
                _registration.OnnxDecompMeta(
                    native_add,
                    self.target,
                    None,
                    is_custom=custom,
                    is_complex=complex_,
                    opset_introduced=version,
                    skip_signature_inference=True,
                ),
            )
        registry._cleanup_registry_based_on_opset_version()
        self.assertEqual(len(registry.get_decomps(self.target)), 3)

    def export(self, model=None, **kwargs):
        return torch.onnx.export(
            AddModel().eval() if model is None else model,
            (torch.ones(2), torch.ones(2)),
            optimize=kwargs.pop("optimize", False),
            verbose=False,
            **kwargs,
        )

    def test_native_export_uses_direct_builder_and_metadata(self):
        torch.onnx.register_custom_translation(self.target, native_add)
        with mock.patch(
            "torch.onnx._internal.exporter._core._call_legacy_translation",
            wraps=_core._call_legacy_translation,
        ) as legacy:
            program = self.export()
        types = [n.op_type for n in program.model.graph]
        self.assertEqual(types, ["Constant", "Mul", "Add", "Sin"])
        self.assertEqual(legacy.call_count, 0)
        self.assertEqual(len(set(program.model.graph)), len(types))
        self.assertEqual(program.model.graph.outputs[0].shape, ir.Shape([2]))
        for node in program.model.graph:
            self.assertIn("node", node.meta)
        onnx.checker.check_model(program.model_proto, full_check=True)
        x, y = torch.ones(2), torch.ones(2)
        result = ReferenceEvaluator(program.model_proto).run(
            None, {"x": x.numpy(), "y": y.numpy()}
        )[0]
        self.assertEqual(torch.from_numpy(result), AddModel()(x, y))

    def test_local_legacy_override_precedes_global(self):
        torch.onnx.register_custom_translation(self.target, native_add)

        def legacy_sub(self, other, alpha=1):
            return op.Sub(self, other)

        program = self.export(custom_translation_table={self.target: legacy_sub})
        types = [n.op_type for n in program.model.graph]
        self.assertIn("Sub", types)
        self.assertNotIn("Add", types)
        onnx.checker.check_model(program.model_proto, full_check=True)

    def test_mutation_during_translation_only_affects_next_export(self):
        def mutating(op, self, other, alpha=1):
            torch.onnx.register_custom_translation(
                self_target, native_sub, replace=True
            )
            return native_add(op, self, other, alpha)

        self_target = self.target
        torch.onnx.register_custom_translation(self.target, mutating)
        first = self.export()
        second = self.export()
        self.assertIn("Add", [n.op_type for n in first.model.graph])
        self.assertIn("Sub", [n.op_type for n in second.model.graph])

    def test_invalid_native_output_is_actionable(self):
        torch.onnx.register_custom_translation(self.target, lambda *args, **kwargs: 42)
        with self.assertRaisesRegex(Exception, "must return 1 ONNX IR value"):
            self.export()

    def test_complex_dispatch(self):
        torch.onnx.register_custom_translation(self.target, native_add)
        torch.onnx.register_custom_translation(self.target, native_sub, is_complex=True)
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        x.meta["val"] = torch.ones(2, dtype=torch.complex64)
        node = graph.call_function(self.target, (x, x))
        registry = _registration.ONNXRegistry.from_torchlib()
        function, _ = _dispatching.dispatch(node, registry)
        self.assertIs(function.function, native_sub)

    def test_native_signature_constraints_are_preserved(self):
        constraint = ir.schemas.TypeConstraintParam(
            "T", {ir.TensorType(ir.DataType.FLOAT), ir.TensorType(ir.DataType.DOUBLE)}
        )
        signature = ir.schemas.OpSignature(
            "",
            "custom_add",
            overload="",
            since_version=18,
            params=[
                ir.schemas.Parameter("self", constraint, required=True, variadic=False),
                ir.schemas.Parameter(
                    "other", constraint, required=True, variadic=False
                ),
            ],
            outputs=[
                ir.schemas.Parameter(
                    "result", constraint, required=True, variadic=False
                )
            ],
        )
        torch.onnx.register_custom_translation(
            self.target, native_add, signature=signature
        )
        entry = _custom_translation.snapshot()[0]
        self.assertIs(entry.signature, signature)
        float_ = ir.Value(type=ir.TensorType(ir.DataType.FLOAT))
        double = ir.Value(type=ir.TensorType(ir.DataType.DOUBLE))
        with self.assertRaisesRegex(TypeError, "shared constraint"):
            entry(None, float_, double)
        integer = ir.Value(type=ir.TensorType(ir.DataType.INT64))
        with self.assertRaisesRegex(TypeError, "outside constraint"):
            entry(None, integer, integer)
        program = self.export()
        onnx.checker.check_model(program.model_proto, full_check=True)

    def test_snapshot_is_independent_of_concurrent_updates(self):
        torch.onnx.register_custom_translation(self.target, native_add)
        old = _custom_translation.snapshot()
        with mock.patch.object(
            _registration._torchlib_registry, "get_torchlib_ops", return_value=[]
        ):
            registry = _registration.ONNXRegistry.from_torchlib(custom_translations=old)
        thread = threading.Thread(
            target=torch.onnx.register_custom_translation,
            args=(self.target, native_sub),
            kwargs={"replace": True},
        )
        thread.start()
        thread.join(timeout=10)
        self.assertFalse(thread.is_alive())
        self.assertIs(
            registry.get_decomps(self.target)[0].onnx_function.function, native_add
        )
        self.assertIs(_custom_translation.snapshot()[0].function, native_sub)

    def test_native_function_definitions_and_custom_opsets(self):
        x = ir.Value(
            name="x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([2])
        )
        graph = ir.Graph([x], [], nodes=[], name="negate", opset_imports={"": 18})
        from torch.onnx._internal.exporter._ir_builder import GraphBuilder

        builder = GraphBuilder(graph)
        graph.outputs.append(builder.call("Neg", x)[0])
        function = ir.Function("native.test", "Negate", "", graph=graph, attributes=[])

        def custom(op, self, other, alpha=1):
            builder = op.builder
            result = builder.call_function(function, self, version=1)[0]
            return builder.call("Add", result, other)[0]

        torch.onnx.register_custom_translation(self.target, custom)
        registry = _registration.ONNXRegistry.from_torchlib(18)
        program = _core.export(
            AddModel().eval(),
            (torch.ones(2), torch.ones(2)),
            registry=registry,
            optimize=False,
            verbose=False,
        )
        self.assertIn(function.identifier(), program.model.functions)
        self.assertEqual(program.model.graph.opset_imports["native.test"], 1)
        onnx.checker.check_model(program.model_proto, full_check=True)
        converted = self.export()
        self.assertIn("Neg", [n.op_type for n in converted.model.graph])
        onnx.checker.check_model(converted.model_proto, full_check=True)

    def test_same_export_uses_snapshot_after_translation_unregisters(self):
        target = self.target
        calls = []

        def custom(op, self, other, alpha=1):
            if not calls:
                torch.onnx.unregister_custom_translation(target)
            calls.append(1)
            return native_sub(op, self, other)

        class TwoAdds(torch.nn.Module):
            def forward(self, x, y):
                return (x + y) + y

        torch.onnx.register_custom_translation(target, custom)
        program = self.export(TwoAdds().eval())
        self.assertEqual(calls, [1, 1])
        self.assertEqual([n.op_type for n in program.model.graph], ["Sub", "Sub"])

    def test_snapshot_precedes_model_capture(self):
        target = self.target
        torch.onnx.register_custom_translation(target, native_add)

        class MutatingModel(torch.nn.Module):
            def forward(self, x, y):
                torch.onnx.register_custom_translation(target, native_sub, replace=True)
                return x + y

        program = self.export(MutatingModel().eval())
        self.assertIn("Add", [n.op_type for n in program.model.graph])
        self.assertIs(_custom_translation.snapshot()[0].function, native_sub)

    def test_native_complex_output_retains_packed_shape(self):
        torch.onnx.register_custom_translation(self.target, native_add, is_complex=True)

        class ComplexAdd(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        program = torch.onnx.export(
            ComplexAdd().eval(),
            (
                torch.ones(3, dtype=torch.complex64),
                torch.ones(3, dtype=torch.complex64),
            ),
            optimize=False,
            verbose=False,
        )
        output = program.model.graph.outputs[0]
        self.assertEqual(output.dtype, ir.DataType.FLOAT)
        self.assertEqual(output.shape, ir.Shape([3, 2]))
        onnx.checker.check_model(program.model_proto, full_check=True)

    def test_native_outputs_support_legacy_python_operators(self):
        torch.onnx.register_custom_translation(self.target, native_add)

        class Negate(torch.nn.Module):
            def forward(self, x, y):
                return -(x + y)

        def legacy_neg(self):
            return -self

        program = self.export(
            Negate().eval(),
            custom_translation_table={torch.ops.aten.neg.default: legacy_neg},
        )
        self.assertEqual(
            [n.op_type for n in program.model.graph], ["Add", "Identity", "Neg"]
        )
        onnx.checker.check_model(program.model_proto, full_check=True)

    def test_native_multiple_outputs(self):
        target = torch.ops.aten.topk.default

        @torch.onnx.register_custom_translation(target)
        def topk(op, self, k, dim=-1, largest=True, sorted=True):
            return op.TopK(
                self, [k], axis=dim, largest=int(largest), sorted=int(sorted)
            )

        class TopK(torch.nn.Module):
            def forward(self, x, y):
                return torch.topk(x, 1)

        program = self.export(TopK().eval())
        self.assertEqual(len(program.model.graph.outputs), 2)
        self.assertEqual(program.model.graph.outputs[0].dtype, ir.DataType.FLOAT)
        self.assertEqual(program.model.graph.outputs[1].dtype, ir.DataType.INT64)
        onnx.checker.check_model(program.model_proto, full_check=True)

    def test_native_translation_inside_subgraph(self):
        torch.onnx.register_custom_translation(self.target, native_add)

        class Conditional(torch.nn.Module):
            def forward(self, x, y):
                return torch.cond(
                    x.sum() > 0, lambda a, b: a + b, lambda a, b: a - b, (x, y)
                )

        program = self.export(Conditional().eval())
        onnx.checker.check_model(program.model_proto, full_check=True)
        if_node = next(n for n in program.model.graph if n.op_type == "If")
        then_graph = if_node.attributes["then_branch"].value
        self.assertIn("Add", [n.op_type for n in then_graph])
        for node in then_graph:
            self.assertIs(node.graph, then_graph)

    @common_utils.parametrize("opset_version", [18, 19, 20])
    def test_native_opset_selection_in_export(self, opset_version):
        torch.onnx.register_custom_translation(
            self.target, native_add, opset_version=17
        )
        torch.onnx.register_custom_translation(
            self.target, native_sub, opset_version=19
        )
        program = self.export(opset_version=opset_version, optimize=True)
        types = [n.op_type for n in program.model.graph]
        self.assertIn("Sub" if opset_version >= 19 else "Add", types)
        self.assertEqual(program.model.opset_imports[""], opset_version)
        onnx.checker.check_model(program.model_proto, full_check=True)

    def test_native_outputs_support_scripted_legacy_translations(self):
        torch.onnx.register_custom_translation(self.target, native_add)

        @script()
        def scripted_relu(self: FLOAT, negative_slope: float = 0.01) -> FLOAT:
            return op.LeakyRelu(self, alpha=negative_slope)

        class LeakyRelu(torch.nn.Module):
            def forward(self, x, y):
                return torch.nn.functional.leaky_relu(x + y, negative_slope=0.2)

        program = self.export(
            LeakyRelu().eval(),
            custom_translation_table={torch.ops.aten.leaky_relu.default: scripted_relu},
        )
        self.assertIn("LeakyRelu", [n.op_type for n in program.model.graph])
        onnx.checker.check_model(program.model_proto, full_check=True)


if __name__ == "__main__":
    common_utils.run_tests()
