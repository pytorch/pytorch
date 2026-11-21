# Owner(s): ["module: onnx"]
"""Global native registrations and their export-time isolation."""

from __future__ import annotations

import operator
import subprocess
import sys
from typing import TYPE_CHECKING

import torch
from torch.onnx._internal.exporter import _custom_translation
from torch.testing._internal import common_utils
from torch.testing._internal.common_utils import HardwareClassification


if TYPE_CHECKING:
    import onnx_ir as ir


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
            _custom_translation.register_custom_translation(self.target)(native_add),
            native_add,
        )
        self.assertIs(
            _custom_translation.register_custom_translation(operator.add, native_add),
            native_add,
        )
        self.assertEqual(len(_custom_translation.snapshot()), 2)

    def test_duplicate_replace_and_exact_unregister(self):
        _custom_translation.register_custom_translation(self.target, native_add)
        _custom_translation.register_custom_translation(
            self.target, native_add, is_complex=True
        )
        _custom_translation.register_custom_translation(
            self.target, native_add, opset_version=19
        )
        with self.assertRaisesRegex(ValueError, "replace=True"):
            _custom_translation.register_custom_translation(self.target, native_sub)
        _custom_translation.register_custom_translation(
            self.target, native_sub, replace=True
        )
        entries = _custom_translation.snapshot()
        self.assertIs(entries[0].function, native_sub)
        self.assertIs(entries[1].function, native_add)
        _custom_translation.unregister_custom_translation(self.target)
        self.assertEqual(len(_custom_translation.snapshot()), 2)
        with self.assertRaisesRegex(KeyError, "metadata from registration"):
            _custom_translation.unregister_custom_translation(self.target)

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
            _custom_translation.register_custom_translation(
                target, native_add, **kwargs
            )
        self.assertEqual(_custom_translation.snapshot(), ())

    def test_invalid_function(self):
        with self.assertRaisesRegex(TypeError, "must be callable"):
            _custom_translation.register_custom_translation(self.target, 42)

    def test_builder_scope_error(self):
        with self.assertRaisesRegex(RuntimeError, "No ONNX IR builder is active"):
            _custom_translation.get_translation_builder()

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


if __name__ == "__main__":
    common_utils.run_tests()
