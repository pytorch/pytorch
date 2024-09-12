# Owner(s): ["module: onnx"]
"""Unit tests for the _tensors module."""

from __future__ import annotations

import unittest

import onnxscript

from torch.onnx._internal.exporter import _tensors


class SymbolicTensorTest(unittest.TestCase):
    def test_it_is_hashable(self):
        tensor = _tensors.SymbolicTensor(
            opset=onnxscript.values.Opset(domain="test", version=1)
        )
        self.assertEqual(hash(tensor), hash(tensor))
        self.assertIn(tensor, {tensor})


if __name__ == "__main__":
    unittest.main()
