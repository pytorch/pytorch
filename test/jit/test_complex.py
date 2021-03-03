import torch
import os
import sys
import unittest
from torch.testing._internal.jit_utils import JitTestCase
from typing import List, Dict
from torch.testing._internal.common_utils import IS_MACOS

# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)

class TestComplex(JitTestCase):
    def test_script(self):
        def fn(a: complex):
            return a

        self.checkScript(fn, (3 + 5j,))

    def test_complexlist(self):
        def fn(a: List[complex], idx: int):
            return a[idx]

        input = [1j, 2, 3 + 4j, -5, -7j]
        self.checkScript(fn, (input, 2))

    def test_complexdict(self):
        def fn(a: Dict[complex, complex], key: complex) -> complex:
            return a[key]

        input = {2 + 3j : 2 - 3j, -4.3 - 2j: 3j}
        self.checkScript(fn, (input, -4.3 - 2j))

    def test_pickle(self):
        class ComplexModule(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                # initialization is done in python
                # JIT doesn't parse init
                self.a = 3 + 5j
                self.b = [2 + 3j, 3 + 4j, 0 - 3j, -4 + 0j]
                self.c = {2 + 3j : 2 - 3j, -4.3 - 2j: 3j}

            def forward(self, b: int):
                return b + 2j

        loaded = self.getExportImportCopy(ComplexModule())
        self.assertEqual(loaded.a, 3 + 5j)
        self.assertEqual(loaded.b, [2 + 3j, 3 + 4j, -3j, -4])
        self.assertEqual(loaded.c, {2 + 3j : 2 - 3j, -4.3 - 2j: 3j})

    @unittest.skipIf(IS_MACOS, "FIX ME: Tensors fail to compare equal on mac")
    def test_complex_parse(self):
        def fn1(a: int):
            return a + complex(1, 2) + complex(-2, 3.4) + complex(-2.3, 3.4) + complex(2.1, -3.5)

        def fn(a: int, b: torch.Tensor, dim: int):
            # verifies `emitValueToTensor()` 's behavior
            b[dim] = 2.4 + 0.5j
            return a + b + 5j - 7.4j - 4

        t1 = torch.tensor(1)
        t2 = torch.tensor([0.4, 1.4j, 2.35])
        scripted = torch.jit.script(fn)
        self.assertEqual(scripted(t1, t2, 2), fn(t1, t2, 2))

        scripted1 = torch.jit.script(fn1)
        self.assertEqual(scripted1(t1), fn1(t1))
